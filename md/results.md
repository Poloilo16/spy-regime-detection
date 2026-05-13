# Results — SPY Vol-Shock Pipeline (modular)

**Last log sync:** 2026-05-13 (`logs/experiment_log.csv`)  
**Data path:** `src/Data/quant.db` (via `training.DEFAULT_DB_PATH`)  
**Model:** `XGBClassifier`, walk-forward CV (`TimeSeriesSplit`, rolling train `max_train_size=756` where configured)  
**Labels (`target`):** 5d realized-vol **shock ratio** `fwd_rv / bwd_rv - 1` vs. bands `RV_SHOCK_LOW` / `RV_SHOCK_HIGH` (±0.075) → **0 Quéda**, **1 Estável**, **2 Alta**  
**Features:** `BASE` + lags `[1,5]`, `regime_lag1` (expanding `rv_21d` quartiles), rolling HMM **`p_hmm*_tmrw`**

---

## Target definitions (3-class)

| Class | Name | Condition (vol shock ratio) |
|-------|------|-------------------------------|
| 0 | Queda | ratio < −0.075 |
| 1 | Estável | within band |
| 2 | Alta | ratio > +0.075 |

---

## Walk-forward reference — default `scripts/predict.py`

Hyperparameters (illustrative): `max_depth=4`, `n_estimators=300`, `HMM_REFIT=21`, `N_HMM_STATES=4`, etc. — see `scripts/predict.py`.

**Logged runs (2026-05-13):** two identical executions reported **global accuracy ≈ 52.16%**, **Recall_Estável ≈ 0.029**, **F1 weighted ≈ 0.50** (class imbalance + balanced weights compress headline accuracy vs. legacy quartile-only reports).

For historical comparison: earlier project snapshots using **different label definitions** (pure `rv_21d` quartile classification) reported **~69%** overall accuracy; **direct numeric comparison across label laws is invalid** — treat ~69% as a **different experiment**, not the same `target` row.

---

## Optuna — best logged trial (F1 **Estável** maximization)

**Study objective:** maximize `f1_stable` from `walk_forward_cv_metrics` (F1 of class 1).

**Best row (`OPTUNA_BEST_20260513_162138`):**

| Metric | Value |
|--------|--------|
| **Global accuracy** | **0.4777 (~47.77%)** |
| **Recall_Estável** | 0.1503 |
| **F1 weighted** | 0.4828 |
| `N_HMM_STATES` | 3 |
| `HMM_REFIT` | 63 |
| `XGB_MAX_DEPTH` | 3 |
| `XGB_LEARNING_RATE` | 0.01 |
| `XGB_N_ESTIMATORS` | 100 |

**Takeaway:** pushing the optimizer to lift **Estável** forces a **shallower** booster and a **slower** HMM refresh cadence; **global accuracy falls** into the high-40s (%). This matches the qualitative diagnosis: **VIX / VRP are shock-aligned features**, not mean-reversion anchors.

---

## Explicability (SHAP)

`scripts/predict.py` runs **`shap.TreeExplainer`** on the **last** feature row and prints the top-5 signed SHAP values for the **predicted** class — use this to validate that the model is not trivially collapsing to a single spurious driver on the live date.

---

## GARCH(1,1)

Rolling window fit (252 returns) through each **t**; conditional variance forecast at **t** feeds `garch_var` and lagged columns. Per-trial AIC/BIC are printed in verbose paths when enabled in scripts.

---

## Next steps for fresh numbers

```bash
python scripts/predict.py
python scripts/optimize.py   # optional; long-running
```

Then re-open **`logs/experiment_log.csv`** and **`logs/preds_*.csv`** for the authoritative metrics row.
