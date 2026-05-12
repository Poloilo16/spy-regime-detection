import warnings
warnings.filterwarnings('ignore')

import duckdb
import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix

DB_PATH      = '/Users/lucaszelmanovits/Desktop/Quant/Research/Data/quant.db'
LAGS         = [1, 5]
BASE         = ['log_return', 'rv_21d', 'vix', 'vrp', 'slope_2s10s', 'garch_var']
REGIME_NAMES = {0: 'Low vol', 1: 'Normal', 2: 'Elevated', 3: 'Crisis'}
MIN_WINDOW   = 63    # minimum days before rv_21d expanding percentile is meaningful

# Rolling HMM config
HMM_FEATURES = ['log_return', 'rv_21d', 'vix', 'vrp', 'slope_2s10s']
N_HMM_STATES = 4
MIN_HMM      = 252   # 1 year minimum — need enough data to identify 4 states reliably
HMM_REFIT    = 21    # refit monthly; HMM parameters are stable day-to-day
HMM_COLS     = [f'p_hmm{s}_tmrw' for s in range(N_HMM_STATES)]

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0,
)

# ── 1. Load ───────────────────────────────────────────────────────────────────
conn = duckdb.connect(DB_PATH)
df = conn.execute("""
    SELECT
        p.date,
        p.close,
        p.log_return,
        p.rv_21d,
        p.vrp,
        MAX(CASE WHEN m.series_id = 'VIX'   THEN m.value END) AS vix,
        MAX(CASE WHEN m.series_id = '2S10S' THEN m.value END) AS slope_2s10s
    FROM prices_daily p
    LEFT JOIN macro_daily m ON p.date = m.date
    WHERE p.ticker = 'SPY'
    GROUP BY p.date, p.close, p.log_return, p.rv_21d, p.vrp
    ORDER BY p.date
""").df()
conn.close()

df = df.dropna().reset_index(drop=True)
print(f"Loaded {len(df)} rows  |  {df.date.min().date()} → {df.date.max().date()}")

# ── 2. Expanding-window rv_21d percentile regime labels (no lookahead) ───────
# At each time t, regime = quartile of rv_21d[t] relative to all rv_21d[0..t].
# Decouples labels from VIX, forces genuine prediction of future realized vol state.
rv_pct = df['rv_21d'].expanding(min_periods=MIN_WINDOW).rank(pct=True)
df['regime'] = pd.cut(rv_pct, bins=[0, 0.25, 0.50, 0.75, 1.0],
                      labels=[0, 1, 2, 3], include_lowest=True)
df = df.dropna(subset=['regime']).reset_index(drop=True)
df['regime'] = df['regime'].astype(int)

print("\nRegime distribution (rv_21d expanding quartiles):")
print(df['regime'].value_counts().sort_index().rename(REGIME_NAMES))

# ── 3. GARCH(1,1) conditional variance ───────────────────────────────────────
garch_fit = arch_model(df['log_return'] * 100, vol='Garch', p=1, q=1, dist='normal').fit(disp='off')
df['garch_var'] = garch_fit.conditional_volatility ** 2
print(f"\nGARCH(1,1) fitted  |  AIC: {garch_fit.aic:.1f}  BIC: {garch_fit.bic:.1f}")

# ── 4. Rolling HMM — forward probabilities (no lookahead) ────────────────────
# At each day T:
#   - Every HMM_REFIT days: refit HMM on expanding window, resolve label switching
#     by reordering states to ascending rv_21d mean.
#   - Between refits: propagate filtered P(state_T | obs_0:T) via one-step forward
#     algorithm (predict step via transmat_, update step via emission probability).
#   - Store P(state_T+1 | obs_0:T) = filtered_T @ transmat_ as features.
#
# Using the terminal step of predict_proba at each refit gives filtered = smoothed
# (no future information at the last observation). Subsequent incremental steps
# use the Bayesian filter to avoid reprocessing the full window each day.
hmm_proba = np.full((len(df), N_HMM_STATES), np.nan)
current   = {}
n_steps   = len(df) - MIN_HMM
print(f"\nBuilding rolling HMM forward probabilities ({n_steps} steps, refit every {HMM_REFIT} days)...")

for t in range(MIN_HMM, len(df)):
    refit = (not current) or ((t - MIN_HMM) % HMM_REFIT == 0)

    if refit:
        scaler = StandardScaler()
        X_fit  = scaler.fit_transform(df[HMM_FEATURES].iloc[:t + 1].values)
        model  = GaussianHMM(n_components=N_HMM_STATES, covariance_type='diag',
                             n_iter=100, random_state=42)
        model.fit(X_fit)

        # Resolve label switching: state 0 = lowest rv_21d mean
        hard    = model.predict(X_fit)
        rv_col  = HMM_FEATURES.index('rv_21d')
        mean_rv = [
            X_fit[hard == s, rv_col].mean() if (hard == s).any() else 0.0
            for s in range(N_HMM_STATES)
        ]
        order = np.argsort(mean_rv)  # order[new_state] = original_state

        # Terminal filtered proba (smoothed == filtered at last observation)
        p_filtered = model.predict_proba(X_fit)[-1]
        current = {'model': model, 'scaler': scaler, 'order': order, 'filtered': p_filtered}

        progress = (t - MIN_HMM) / n_steps * 100
        if (t - MIN_HMM) % (HMM_REFIT * 10) == 0:
            print(f"  {progress:4.0f}%  (t={t})")
    else:
        # Incremental forward step with new observation x_t
        model = current['model']
        x_t   = current['scaler'].transform(df[HMM_FEATURES].iloc[[t]].values)[0]

        # Prediction step: P(state_t | obs_0:t-1) = filtered_{t-1} @ transmat_
        p_predict = current['filtered'] @ model.transmat_

        # Emission: P(x_t | state=s) for each s
        # covars_ is (n_states, n_features) for 'diag' — build diagonal matrix
        model = current['model']
        emission = np.array([
            multivariate_normal.pdf(x_t, mean=model.means_[s],
                                    cov=np.diag(model.covars_[s]))
            for s in range(N_HMM_STATES)
        ])

        # Update step: normalize to get P(state_t | obs_0:t)
        unnorm = p_predict * emission
        p_filtered = unnorm / (unnorm.sum() or 1.0)
        current['filtered'] = p_filtered

    # P(state_T+1 | obs_0:T) = filtered_T @ transmat_, reordered to named states
    p_tomorrow_orig = current['filtered'] @ current['model'].transmat_
    hmm_proba[t]    = p_tomorrow_orig[current['order']]

for i, col in enumerate(HMM_COLS):
    df[col] = hmm_proba[:, i]

print(f"  100%  — {(~np.isnan(hmm_proba[:, 0])).sum()} rows have HMM features")

# ── 5. Feature engineering ────────────────────────────────────────────────────
for col in BASE:
    for lag in LAGS:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

df['regime_lag1'] = df['regime'].shift(1)  # strong persistence signal
df['target']      = df['regime'].shift(-1)

df = df.dropna().reset_index(drop=True)

# HMM_COLS are not lagged — they already represent P(regime_T+1 | data_0:T)
FEATURES = BASE + [f'{c}_lag{lag}' for lag in LAGS for c in BASE] + ['regime_lag1'] + HMM_COLS
X = df[FEATURES].values
y = df['target'].astype(int).values

print(f"\nFeature matrix: {X.shape[0]} rows × {X.shape[1]} features")

# ── 6. Walk-forward cross-validation ─────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
fold_preds, fold_true = [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    clf = XGBClassifier(**XGB_PARAMS)
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])
    fold_preds.append(preds)
    fold_true.append(y[test_idx])
    print(f"  Fold {fold+1}: test rows={len(test_idx):4d}  accuracy={(preds == y[test_idx]).mean():.3f}")

all_preds = np.concatenate(fold_preds)
all_true  = np.concatenate(fold_true)

target_names = [f'{i}-{name}' for i, name in REGIME_NAMES.items()]
print("\nWalk-forward CV — Classification Report:")
print(classification_report(all_true, all_preds, target_names=target_names, zero_division=0))

print("Confusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(all_true, all_preds))

# ── 7. Final model — fit on all data ─────────────────────────────────────────
final_clf = XGBClassifier(**XGB_PARAMS)
final_clf.fit(X, y)

# ── 8. Feature importances ────────────────────────────────────────────────────
importance = pd.Series(final_clf.feature_importances_, index=FEATURES)
print("\nTop 15 feature importances:")
print(importance.nlargest(15).round(4).to_string())

# ── 9. Predict tomorrow's regime ─────────────────────────────────────────────
x_latest     = df[FEATURES].iloc[[-1]].values
pred_proba   = final_clf.predict_proba(x_latest)[0]
pred_regime  = int(np.argmax(pred_proba))
latest_date  = df['date'].iloc[-1]
today_regime = int(df['regime_lag1'].iloc[-1])

print(f"\n{'─'*45}")
print(f"Latest date  : {latest_date.date()}")
print(f"Today regime : {today_regime} ({REGIME_NAMES[today_regime]})")
print(f"Predicted    : Regime {pred_regime} ({REGIME_NAMES[pred_regime]})")
print(f"Probabilities: " + "  ".join(f"R{i}={p:.2f}" for i, p in enumerate(pred_proba)))
print(f"{'─'*45}")
