"""
Shared SPY regime / shock pipeline: load → HMM → features → walk-forward CV.
Used by predict.py and optimize.py (import-safe; no side effects at import).
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import duckdb
import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = str(_ROOT / 'Data' / 'quant.db')

LAGS = [1, 5]
BASE = [
    'log_return', 'rv_21d', 'vix', 'vrp', 'slope_2s10s', 'garch_var',
    'amihud_liquidity', 'hy_oas', 'icsa',
]
REGIME_NAMES = {0: 'Low vol', 1: 'Normal', 2: 'Elevated', 3: 'Crisis'}
TARGET_NAMES = {0: 'Queda', 1: 'Estável', 2: 'Alta'}
MIN_WINDOW = 63
MIN_GARCH = 252
HMM_FEATURES = ['log_return', 'vix']
HMM_VOL_IDX = HMM_FEATURES.index('vix')
MIN_HMM_DEFAULT = 756
FORWARD_HORIZON = 5
RV_SHOCK_LOW = -0.15
RV_SHOCK_HIGH = 0.15
# Só acata Queda (0) ou Alta (2) se p da classe direcional > limiar; caso contrário força Estável (1).
TRADE_CONFIDENCE_THRESHOLD = 0.65
LOAD_REQUIRED = ['close', 'log_return', 'rv_21d', 'vrp', 'vix', 'slope_2s10s']

XGB_PARAMS_BASE = dict(
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0,
)


def _macro_series_to_wide(long_df: pd.DataFrame, rename: dict) -> pd.DataFrame:
    if long_df.empty:
        cols = {'date': pd.Series(dtype='datetime64[ns]')}
        for col in rename.values():
            cols[col] = pd.Series(dtype='float64')
        return pd.DataFrame(cols)
    wide = (
        long_df.pivot_table(index='date', columns='series_id', values='value', aggfunc='last')
        .rename(columns=rename)
        .reset_index()
    )
    wide['date'] = pd.to_datetime(wide['date'])
    return wide.sort_values('date')


def load_merged_df(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    conn = duckdb.connect(db_path)
    df = conn.execute("""
        SELECT
            date,
            close,
            log_return,
            rv_21d,
            vrp,
            amihud_liquidity
        FROM prices_daily
        WHERE ticker = 'SPY'
        ORDER BY date
    """).df()
    conn.close()

    conn = duckdb.connect(db_path)
    macro_vol = conn.execute("""
        SELECT date, series_id, value
        FROM macro_daily
        WHERE series_id IN ('VIX', '2S10S')
        ORDER BY date
    """).df()
    macro_credit = conn.execute("""
        SELECT date, series_id, value
        FROM macro_daily
        WHERE series_id IN ('HY_OAS', 'ICSA')
        ORDER BY date
    """).df()
    conn.close()

    macro_vol_w = _macro_series_to_wide(macro_vol, {'VIX': 'vix', '2S10S': 'slope_2s10s'})
    macro_cred_w = _macro_series_to_wide(macro_credit, {'HY_OAS': 'hy_oas', 'ICSA': 'icsa'})
    macro_wide = pd.merge(macro_vol_w, macro_cred_w, on='date', how='outer').sort_values('date')
    for col in ('vix', 'slope_2s10s', 'hy_oas', 'icsa'):
        if col not in macro_wide.columns:
            macro_wide[col] = np.nan
    macro_wide[['vix', 'slope_2s10s', 'hy_oas', 'icsa']] = macro_wide[
        ['vix', 'slope_2s10s', 'hy_oas', 'icsa']
    ].ffill()
    macro_wide = macro_wide[['date', 'vix', 'slope_2s10s', 'hy_oas', 'icsa']].drop_duplicates(
        'date', keep='last'
    )
    macro_wide = macro_wide.sort_values('date', kind='mergesort').reset_index(drop=True)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = pd.merge_asof(df, macro_wide, on='date', direction='backward')
    df['hy_oas'] = df['hy_oas'].ffill().bfill()
    df['icsa'] = df['icsa'].ffill().bfill()
    df['icsa'] = df['icsa'].fillna(0.0)
    return df.dropna(subset=LOAD_REQUIRED).reset_index(drop=True)


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rv_pct = df['rv_21d'].expanding(min_periods=MIN_WINDOW).rank(pct=True)
    df['regime'] = pd.cut(
        rv_pct, bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=[0, 1, 2, 3], include_lowest=True,
    )
    df = df.dropna(subset=['regime']).reset_index(drop=True)
    df['regime'] = df['regime'].astype(int)
    return df


def add_garch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lr = df['log_return'].values
    n = len(df)
    garch_var = np.full(n, np.nan)
    for t in range(MIN_GARCH - 1, n):
        window = lr[t - MIN_GARCH + 1: t + 1] * 100.0
        try:
            res = arch_model(window, vol='Garch', p=1, q=1, dist='normal').fit(disp='off')
            fcst = res.forecast(horizon=1, reindex=False)
            garch_var[t] = float(fcst.variance.values[-1, 0])
        except Exception:
            garch_var[t] = np.nan
    df['garch_var'] = garch_var
    return df


def _emission_full(model, x_vec, n_states, n_dim, eps=1e-6):
    out = np.empty(n_states)
    for s in range(n_states):
        cov = np.asarray(model.covars_[s], dtype=float)
        cov = cov + eps * np.eye(n_dim)
        out[s] = multivariate_normal.pdf(x_vec, mean=model.means_[s], cov=cov)
    return out


def add_hmm_forward_proba(
    df: pd.DataFrame,
    n_hmm_states: int,
    hmm_refit: int,
    *,
    min_hmm: int = MIN_HMM_DEFAULT,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    n_dim = len(HMM_FEATURES)
    hmm_cols = [f'p_hmm{s}_tmrw' for s in range(n_hmm_states)]
    hmm_proba = np.full((len(df), n_hmm_states), np.nan)
    current = {}
    n_steps = len(df) - min_hmm

    for t in range(min_hmm, len(df)):
        refit = (not current) or ((t - min_hmm) % hmm_refit == 0)

        if refit:
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(df[HMM_FEATURES].iloc[:t + 1].values)
            model = GaussianHMM(
                n_components=n_hmm_states,
                covariance_type='full',
                n_iter=100,
                random_state=42,
            )
            model.fit(X_fit)

            hard = model.predict(X_fit)
            mean_vix = [
                X_fit[hard == s, HMM_VOL_IDX].mean() if (hard == s).any() else 0.0
                for s in range(n_hmm_states)
            ]
            order = np.argsort(mean_vix)

            p_filtered = model.predict_proba(X_fit)[-1]
            current = {'model': model, 'scaler': scaler, 'order': order, 'filtered': p_filtered}

            if verbose and (t - min_hmm) % (hmm_refit * 10) == 0:
                progress = (t - min_hmm) / max(n_steps, 1) * 100
                print(f"  HMM {progress:4.0f}%  (t={t})")
        else:
            model = current['model']
            x_t = current['scaler'].transform(df[HMM_FEATURES].iloc[[t]].values)[0]

            p_predict = current['filtered'] @ model.transmat_
            emission = _emission_full(model, x_t, n_hmm_states, n_dim)

            unnorm = p_predict * emission
            p_filtered = unnorm / (unnorm.sum() or 1.0)
            current['filtered'] = p_filtered

        p_tomorrow_orig = current['filtered'] @ current['model'].transmat_
        hmm_proba[t] = p_tomorrow_orig[current['order']]

    for i, col in enumerate(hmm_cols):
        df[col] = hmm_proba[:, i]
    return df, hmm_cols


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lr = df['log_return'].values
    n = len(df)
    h = FORWARD_HORIZON
    sumsq_trailing = np.full(n, np.nan)
    sumsq_fwd = np.full(n, np.nan)
    for t in range(h - 1, n):
        sumsq_trailing[t] = np.sum(lr[t - (h - 1): t + 1] ** 2)
    for t in range(n - h):
        sumsq_fwd[t] = np.sum(lr[t + 1:t + 1 + h] ** 2)

    scale = 252.0 / FORWARD_HORIZON
    bwd_rv = np.sqrt(scale * sumsq_trailing)
    fwd_rv = np.sqrt(scale * sumsq_fwd)
    ratio = np.where(
        (bwd_rv > 1e-12) & np.isfinite(fwd_rv) & np.isfinite(bwd_rv),
        fwd_rv / bwd_rv - 1.0,
        np.nan,
    )
    df['vol_shock_ratio'] = ratio
    df['target'] = np.where(
        ratio < RV_SHOCK_LOW,
        0,
        np.where(ratio > RV_SHOCK_HIGH, 2, 1),
    ).astype(float)
    df.loc[~np.isfinite(ratio), 'target'] = np.nan
    return df


def apply_trade_confidence_filter(preds: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Converte previsões direcionais em Estável (1) quando a confiança na classe 0 ou 2 é baixa."""
    preds = np.asarray(preds, dtype=int).ravel()
    proba = np.asarray(proba, dtype=float)
    out = preds.copy()
    low_conf_down = (preds == 0) & (proba[:, 0] <= TRADE_CONFIDENCE_THRESHOLD)
    low_conf_up = (preds == 2) & (proba[:, 2] <= TRADE_CONFIDENCE_THRESHOLD)
    out[low_conf_down | low_conf_up] = 1
    return out


def add_feature_matrix(df: pd.DataFrame, hmm_cols: list[str]) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    df = df.copy()
    for col in BASE:
        for lag in LAGS:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df['regime_lag1'] = df['regime'].shift(1)
    df = df.dropna().reset_index(drop=True)
    df['target'] = df['target'].astype(int)
    features = BASE + [f'{c}_lag{lag}' for lag in LAGS for c in BASE] + ['regime_lag1'] + hmm_cols
    X = df[features].values
    y = df['target'].values
    return df, features, X, y


def build_xy(
    db_path: str,
    n_hmm_states: int,
    hmm_refit: int,
    *,
    min_hmm: int = MIN_HMM_DEFAULT,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    df = load_merged_df(db_path)
    if verbose:
        print(f"Loaded {len(df)} rows  |  {df.date.min().date()} → {df.date.max().date()}")
    df = add_regime(df)
    df = add_garch(df)
    if verbose:
        valid_g = np.isfinite(df['garch_var']).sum()
        print(f"GARCH forecasts: {valid_g}")
    if verbose:
        print(f"Rolling HMM (n_states={n_hmm_states}, refit={hmm_refit})...")
    df, hmm_cols = add_hmm_forward_proba(df, n_hmm_states, hmm_refit, min_hmm=min_hmm, verbose=verbose)
    df = add_target(df)
    return add_feature_matrix(df, hmm_cols)


def walk_forward_cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    xgb_params: dict,
    *,
    max_train_size: int | None = 756,
    n_splits: int = 5,
) -> dict:
    tscv_kw = dict(n_splits=n_splits)
    if max_train_size is not None:
        tscv_kw['max_train_size'] = max_train_size
    tscv = TimeSeriesSplit(**tscv_kw)
    fold_preds, fold_proba, fold_true, fold_test_idx = [], [], [], []
    for train_idx, test_idx in tscv.split(X):
        clf = XGBClassifier(**xgb_params)
        weights = compute_sample_weight(class_weight='balanced', y=y[train_idx])
        clf.fit(X[train_idx], y[train_idx], sample_weight=weights)
        preds = clf.predict(X[test_idx])
        proba = clf.predict_proba(X[test_idx])
        fold_preds.append(preds)
        fold_proba.append(proba)
        fold_true.append(y[test_idx])
        fold_test_idx.append(test_idx)
    all_preds = np.concatenate(fold_preds)
    all_proba = np.vstack(fold_proba)
    all_preds_trade = apply_trade_confidence_filter(all_preds, all_proba)
    all_true = np.concatenate(fold_true)
    cv_test_idx = np.concatenate(fold_test_idx)
    target_report_names = [f'{i}-{TARGET_NAMES[i]}' for i in range(3)]
    report_dict = classification_report(
        all_true, all_preds_trade,
        target_names=target_report_names,
        zero_division=0,
        output_dict=True,
    )
    report_dict_model = classification_report(
        all_true, all_preds,
        target_names=target_report_names,
        zero_division=0,
        output_dict=True,
    )
    f1_stable = float(report_dict[target_report_names[1]]['f1-score'])
    f1_weighted = float(report_dict['weighted avg']['f1-score'])
    accuracy = float(report_dict['accuracy'])
    return {
        'f1_stable': f1_stable,
        'f1_weighted': f1_weighted,
        'accuracy': accuracy,
        'all_true': all_true,
        'all_preds': all_preds,
        'all_preds_trade': all_preds_trade,
        'all_proba': all_proba,
        'cv_test_idx': cv_test_idx,
        'report_dict': report_dict,
        'report_dict_model': report_dict_model,
        'target_report_names': target_report_names,
    }
