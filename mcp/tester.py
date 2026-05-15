"""
Monte Carlo por bootstrap: testa se o timing dos sinais supera permutações i.i.d.
da mesma distribuição marginal de sinais contra retornos de mercado fixos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Alinhado ao horizonte de 5 sessões do pipeline SPY (annualização).
_PERIODS_PER_YEAR = 252.0 / 5.0


def _compute_metrics(returns: pd.Series) -> dict[str, float]:
    """
    Métricas sobre série de retornos por período (ex.: janelas ~5d).
    Sharpe e vol anualizados com sqrt(252/5); CAGR compondo períodos de 5d.
    """
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 2:
        return {'sharpe': float('nan'), 'cagr': float('nan'), 'volatility': float('nan')}

    mu = float(r.mean())
    vol = float(r.std(ddof=0))
    ann_sqrt = np.sqrt(_PERIODS_PER_YEAR)
    sharpe = (mu / vol) * ann_sqrt if vol > 0 else float('nan')
    vol_ann = vol * ann_sqrt

    wealth = float((1.0 + r).prod())
    years = (n * 5.0) / 252.0
    if years <= 0 or wealth <= 0:
        cagr = float('nan')
    else:
        cagr = float(wealth ** (1.0 / years) - 1.0)

    return {'sharpe': sharpe, 'cagr': cagr, 'volatility': vol_ann}


def bootstrap_test(
    signals: pd.Series,
    market_returns: pd.Series,
    metric: str = 'sharpe',
    n_bootstraps: int = 2000,
) -> dict[str, float]:
    """
    Hipótese nula: sinais i.i.d. amostrados com reposição da distribuição empírica
    dos sinais, multiplicados pelos mesmos `market_returns` (timing aleatório).

    p-value = fração de replicações em que a métrica do bootstrap >= métrica real
    (cauda superior: "macacos" batendo ou igualando o modelo).
    """
    if metric not in ('sharpe', 'cagr', 'volatility'):
        raise ValueError("metric must be one of: 'sharpe', 'cagr', 'volatility'")

    aligned = pd.concat([signals.rename('sig'), market_returns.rename('mkt')], axis=1).dropna()
    if aligned.empty:
        return {'real': float('nan'), 'pvalue': float('nan')}

    sig = aligned['sig'].astype(float)
    mkt = aligned['mkt'].astype(float)
    real_ret = sig * mkt
    metrics = _compute_metrics(real_ret)
    real_score = float(metrics[metric])
    if not np.isfinite(real_score):
        return {'real': real_score, 'pvalue': float('nan')}

    sig_vals = sig.to_numpy()
    mkt_vals = mkt.to_numpy()
    k = len(sig_vals)
    # Sample all bootstraps at once to avoid per-iteration pd.Series construction.
    all_samples = np.random.choice(sig_vals, size=(n_bootstraps, k), replace=True)
    all_rets = all_samples * mkt_vals  # (n_bootstraps, k)
    mu = all_rets.mean(axis=1)
    vol = all_rets.std(axis=1, ddof=0)
    ann_sqrt = np.sqrt(_PERIODS_PER_YEAR)
    if metric == 'sharpe':
        boot_scores = np.where(vol > 0, mu / vol * ann_sqrt, np.nan)
    elif metric == 'volatility':
        boot_scores = vol * ann_sqrt
    else:  # cagr
        wealth = (1.0 + all_rets).prod(axis=1)
        years = float(k * 5.0 / 252.0)
        boot_scores = np.where(
            (wealth > 0) & (years > 0), wealth ** (1.0 / years) - 1.0, np.nan,
        )
    boot_scores = np.where(np.isfinite(boot_scores), boot_scores, -np.inf)

    pvalue = float(np.mean(boot_scores >= real_score))
    return {'real': real_score, 'pvalue': pvalue}
