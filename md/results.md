# Results — SPY Regime Prediction

**Run date:** 2026-05-12  
**Data:** SPY 2016-06-02 → 2026-05-01 (2,475 rows)  
**Model:** XGBoost classifier, walk-forward CV (5 splits)  
**Labels:** `rv_21d` expanding-window quartiles  
**Features:** 23 total — BASE + lags 1 & 5, `regime_lag1`, rolling HMM forward probabilities  

---

## Regime Definitions

| Regime | Label | Definition |
|--------|-------|------------|
| 0 | Low vol | `rv_21d` in bottom quartile of expanding history |
| 1 | Normal | 25th–50th percentile |
| 2 | Elevated | 50th–75th percentile |
| 3 | Crisis | Top quartile |

Distribution in sample: Low vol 258 / Normal 641 / Elevated 838 / Crisis 676

---

## Walk-Forward CV Results

**Feature matrix:** 2,160 rows × 23 features (after lag/HMM warmup)

| Fold | Test rows | Accuracy |
|------|-----------|----------|
| 1 | 360 | 75.3% |
| 2 | 360 | 39.4% |
| 3 | 360 | 80.0% |
| 4 | 360 | 71.9% |
| 5 | 360 | 78.9% |
| **Overall** | **1,800** | **69.2%** |

### Classification Report

| Regime | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| 0-Low vol | 0.94 | 0.38 | 0.54 | 118 |
| 1-Normal | 0.75 | 0.49 | 0.59 | 477 |
| 2-Elevated | 0.67 | 0.67 | 0.67 | 698 |
| 3-Crisis | 0.66 | 0.97 | 0.79 | 507 |
| **Weighted avg** | **0.71** | **0.69** | **0.68** | **1,800** |

### Confusion Matrix (rows = actual, cols = predicted)

|  | Pred 0 | Pred 1 | Pred 2 | Pred 3 |
|--|--------|--------|--------|--------|
| **Actual 0** | 45 | 64 | 3 | 6 |
| **Actual 1** | 2 | 234 | 211 | 30 |
| **Actual 2** | 1 | 12 | 471 | 214 |
| **Actual 3** | 0 | 0 | 13 | 494 |

---

## Feature Importances (Top 15)

| Feature | Importance |
|---------|------------|
| `regime_lag1` | 0.6273 |
| `rv_21d` | 0.0894 |
| `rv_21d_lag1` | 0.0249 |
| `p_hmm3_tmrw` | 0.0225 |
| `p_hmm0_tmrw` | 0.0195 |
| `garch_var` | 0.0162 |
| `p_hmm2_tmrw` | 0.0161 |
| `garch_var_lag1` | 0.0158 |
| `slope_2s10s` | 0.0143 |
| `log_return` | 0.0141 |
| `slope_2s10s_lag1` | 0.0136 |
| `p_hmm1_tmrw` | 0.0136 |
| `rv_21d_lag5` | 0.0129 |
| `vix` | 0.0122 |
| `vix_lag1` | 0.0116 |

Combined HMM importance (`p_hmm0–3_tmrw`): **~7.2%**

---

## GARCH(1,1)

| Metric | Value |
|--------|-------|
| AIC | 6,274.5 |
| BIC | 6,297.7 |

---

## Next-Day Prediction (as of 2026-04-30)

| | |
|--|--|
| Today's regime | 2 — Elevated |
| **Predicted tomorrow** | **1 — Normal** |
| R0 probability | 0.00 |
| R1 probability | **0.90** |
| R2 probability | 0.10 |
| R3 probability | 0.00 |
