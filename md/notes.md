# Project Status & Robustness Notes

## What We Have

### Pipeline (modular)

1. **Data** — SPY daily panel in DuckDB (`prices_daily`); macro / credit / claims in `macro_daily`. Notebook + `scripts/update_db.py` maintain ingestion.
2. **Core library (`src/training.py`)** — `load_merged_df` (as-of merge), `build_xy` (quartile `regime`, GARCH(1,1), rolling HMM forward **P(state tomorrow | data ≤ t)**, 5d vol-shock `target`, feature matrix), `walk_forward_cv_metrics`.
3. **Global HMM (`src/main.py`)** — full-sample Gaussian HMM, BIC-driven state count, `regime_plot.png`; **not** the XGBoost label path.
4. **`scripts/predict.py`** — default hyperparameters, walk-forward CV, final fit, **SHAP (TreeExplainer)** on the latest observation, append row to **`logs/experiment_log.csv`**, write **`logs/preds_*.csv`**.
5. **`scripts/optimize.py`** — Optuna search maximizing **F1 of class 1 (Estável)** on the same walk-forward objective; logs best trial to `experiment_log.csv`.
6. **`scripts/backtest.py`** — applies the latest `preds_*.csv` with **T+1** alignment (`log_return` shifted so the signal at **T** earns **T+1**), turnover costs, Sharpe / drawdown printout.

### Alignment & leakage

- **Rolling HMM:** only data ≤ **t** inside each refit window; incremental forward filter between refits; state order by in-sample VIX mean at refit.
- **GARCH:** rolling 252d fit ending at **t**, 1-step variance forecast at **t**.
- **Macro join:** `pd.merge_asof(direction='backward')` after strict sorting replaces naive `JOIN ON date` for the predictive loader — see CLAUDE.md.

### Optuna vs. “accuracy-first” profile (stable-regime bottleneck)

Hyperparameter search that **maximizes F1(Estável)** does not lift global accuracy — it **compresses** it.

| | Configuração focada em acurácia global (referência interna / modelo mais expressivo) | Melhor trial Optuna (F1 Estável ↑) |
|--|--|--|
| Acurácia global (walk-forward) | ordem **~69%** em relatórios históricos com alvo/artefactos anteriores; runs recentes com alvo **choque 5d** no mesmo pipeline podem situar-se ~**52%** conforme pesos e janela — ver `logs/experiment_log.csv` | **~47.7%** (ex.: `OPTUNA_BEST_20260513_162138`) |
| XGBoost `max_depth` | tipicamente 4–6 sob expressividade | **3** (poda agressiva) |
| `HMM_REFIT` | 21 (default em `predict.py`) | **63** (HMM mais “lento”, menos overfit local) |
| `N_HMM_STATES` | 4 no default de predição | **3** no melhor trial logado |
| Interpretação | fronteira de precisão/recall mais equilibrada para o multiclasse | o classificador **paga** acurácia global para tentar recuperar a classe minoritária estável |

**Feature semantics:** **VIX**, **VRP**, e derivados de choque comportam-se como ***shock hunters*** — elevam sinal em compressões/expansões de vol — e **não** carregam assinatura limpa de **mean-reversion** na classe “Estável”. Por isso o F1 da classe 1 permanece o **gargalo estável**: o Optuna confirma que “abrir” o modelo para crise/queda de vol **não** resolve mean-reversion sem degradar métricas globais.

### SHAP & logging

- Cada corrida de `predict.py` gera **`Experiment_ID`** temporal e grava métricas + top features.
- SHAP no último dia destrincha a contribuição marginal das features para a classe prevista (auditável no terminal).

---

## Known Limitations

### 1. ~~VIX circularity (threshold labels)~~ — Addressed

Target is the **5d realized-vol shock ratio** banded into three classes, not VIX bins.

### 2. ~~HMM lookahead in rolling path~~ — Addressed

Forward filter + refit discipline; HMM probability columns are **tomorrow** conditioned on **≤ t**.

### 3. ~~Naive SQL date JOIN for predictive load~~ — Addressed in `training.load_merged_df`

`merge_asof(backward)` + sort eliminates calendar **lookahead** when stitching daily SPY to irregular / mixed-release macro.

### 4. **Optuna “stable F1” vs. global accuracy — structural trade-off**

Maximizing **F1(Estável)** pushes toward **shallow trees** and **infrequent HMM refits** (ex. `max_depth=3`, `HMM_REFIT=63`), **dropping** global accuracy to **~47.7%** vs. **~69%** in accuracy-oriented configurations on comparable splits. This is not a bug — it exposes that **stability** is almost orthogonal to the **shock-heavy** feature geometry (VIX, VRP, GARCH).

### 5. No guaranteed economic value

High accuracy or tuned recall on one class does not imply tradable alpha; `backtest.py` is the minimal **T+1** sanity check with costs.

### 6. Fold / regime instability

TimeSeriesSplit still exposes **non-stationarity**: some folds underperform when the label distribution shifts (expanding quartiles + shock ratios across cycles).

### 7. Single asset, single decade

SPY-centric; limited stress vs. pre-2016 crises unless data extended.

### 8. Narrow volatility–macro feature cone

Natural extensions: HY term structure, VIX vs VIX3M, breadth, cross-asset vol — especially if we ever need **mean-reversion** signal orthogonal to shocks.

---

## Highest-Impact Next Steps

1. **Backtest loop** — formalize `strategy_log.csv`, grid de custos, compare to buy-and-hold.
2. **Explicit “stability” features** — if the product goal is Estável, engineer signals that mean-revert (range proxies, realized vs implied convergence) rather than only shock ramps.
3. **Reconcile objectives** — multi-objective Optuna (accuracy **and** F1-stable Pareto front) instead of single scalar F1.
4. **OOS assets** — QQQ / EFA with identical pipeline to test transportability of `merge_asof` + HMM block.
