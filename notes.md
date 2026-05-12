# Project Status & Robustness Notes

## What We Have

### Pipeline
1. **Data** — 10 years of SPY daily OHLCV, VIX, and 2s10s yield curve slope in DuckDB. Derived features: log returns, 21-day realized vol (`rv_21d`), VRP, and GARCH(1,1) conditional variance.
2. **HMM regime detection** (`main.py`) — Gaussian HMM fitted on the full sample, BIC-selected 4 states ordered by volatility. Used for visualization only.
3. **Regime prediction** (`predict.py`) — `rv_21d` expanding-window quartile labels, GARCH feature, rolling HMM forward probabilities, XGBoost classifier with walk-forward CV (5 splits). 69% overall accuracy.

### Current Results (rv_21d expanding quartile labels + rolling HMM features)
| Regime | Label | Recall | Notes |
|--------|-------|--------|-------|
| 0 | Low vol (bottom quartile) | 38% | Hardest — transitional signal is weak |
| 1 | Normal (25–50th pct) | 49% | Moderate |
| 2 | Elevated (50–75th pct) | 67% | Reasonable |
| 3 | Crisis (top quartile) | 97% | Excellent |

Top features by importance: `regime_lag1` (63%), `rv_21d` (9%), `p_hmm3_tmrw` (2.2%), `p_hmm0_tmrw` (2.0%), combined HMM importance ~7.2%.

### Rolling HMM Forward Filter
At each day T, the pipeline fits a 4-state Gaussian HMM (diagonal covariance, refit every 21 days on an expanding window) and computes P(state_T+1 | observations_0:T) using the incremental Bayesian filter. Label switching is resolved at every refit by reordering states to ascending `rv_21d` mean. These four probabilities are added as features `p_hmm0_tmrw` through `p_hmm3_tmrw`.

Key implementation choices:
- `covariance_type='diag'` — full covariance produced near-singular matrices with smaller windows; diagonal is stable and has fewer parameters to estimate
- Incremental forward algorithm between refits — avoids O(T²) cost of running `predict_proba` on the full window each day
- `predict_proba(X_fit)[-1]` at refit points gives filtered = smoothed probability (no future information at the terminal observation)

### What Changed vs. VIX-Threshold Version
| | VIX thresholds | rv_21d quartiles | rv_21d + rolling HMM |
|--|--|--|--|
| Overall accuracy | 72% | 69% | 69% |
| Crisis recall | 30% | 98% | 97% |
| VIX importance | 33% | 1.6% | 1.2% |
| HMM importance | — | — | ~7.2% combined |
| Fold consistency | moderate | weak (fold 4: 44%) | weak (fold 2: 39%) |

The HMM columns contribute at the 2–2.2% level each, totaling ~7.2% combined — above the 1–2% baseline of raw features and in line with expectations. They add non-trivial signal but don't close the fold-consistency gap.

---

## Known Limitations

### 1. ~~VIX Circularity~~ — Fixed
Labels are `rv_21d` expanding-window quartiles. VIX importance is now 1.2%.

### 2. ~~HMM Lookahead Bias~~ — Fixed
Rolling HMM uses only data available at time T. Label switching resolved at every refit.

### 3. No Economic Value Test
69% classification accuracy doesn't mean tradeable alpha. The honest test is a backtest: reduce equity exposure when the model predicts Regime 2/3, measure Sharpe improvement over buy-and-hold.

### 4. Fold Consistency is Poor
One fold consistently scores ~40% — near random. This suggests the model learns patterns from specific market regimes that don't transfer to others. The rolling HMM didn't fix this. Likely cause: the expanding-window label distribution shifts over time (early years have few Low vol observations), creating train/test distribution mismatch across folds.

### 5. Low Vol Recall is 38%
The bottom quartile is hard to detect in advance — calm periods look like elevated periods in near-term features. Less critical than missing crisis periods, but matters for re-entry timing.

### 6. Single Asset, Single Decade
Fitted and validated on SPY 2016–2026 only. Not tested on other indices (QQQ, EFA) or earlier periods (GFC, dot-com).

### 7. Feature Set is Narrow
All features are volatility-derived. Standard additions: high-yield credit spreads (HY OAS), vol term structure (VIX minus VIX3M), put-call ratio, market breadth.

---

## Highest-Impact Next Steps

1. **Backtest** — convert regime predictions to a positioning rule and compute Sharpe, max drawdown vs. buy-and-hold. This is the only real validity check.
2. **Investigate fold instability** — understand why one fold consistently fails; likely a distribution shift in the expanding-window labels. Consider using fixed-threshold percentiles rather than purely expanding.
3. **Expand the feature set** — high-yield credit spreads, VIX term structure (VIX-VIX3M), put-call ratio are standard additions.
4. **Out-of-sample on other assets** — run on QQQ or EFA to check generalization.
