# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative research project for **market regime / volatility-shock detection** on SPY. The pipeline collects time series from Interactive Brokers and FRED, stores them in DuckDB, engineers features (GARCH, rolling HMM forward probabilities, macro), and predicts a **5-day realized-vol shock ratio** target with **XGBoost** under walk-forward validation. A separate **global HMM** branch (`src/main.py`) remains for exploratory visualization.

## Running the Code

```bash
python scripts/predict.py    # build_xy → walk-forward CV → final fit → SHAP → logs/
python scripts/optimize.py   # Optuna (requires pip install optuna); appends best trial to logs/
python scripts/backtest.py   # T+1 execution on latest preds_*.csv in logs/
python src/main.py           # global HMM → regime_plot.png (descriptive only)
jupyter notebook notebook.ipynb
```

No formal lint/test harness — research-grade codebase with CSV experiment logging.

## Architecture (post-refactor)

**Library module (`src/training.py`):** single source of truth for data fusion, labels, and models:

- **`load_merged_df`**: SPY from `prices_daily`, macro series pivoted from `macro_daily`, then **`pd.merge_asof(..., direction='backward')`** on sorted dates (see Data Shielding below).
- **`build_xy`**: orchestrates `add_regime` (expanding `rv_21d` quartiles), **`add_garch`**, **`add_hmm_forward_proba`** (rolling HMM + forward filter / refit cadence), **`add_target`** (5d vol-shock ratio → 3 classes), **`add_feature_matrix`** (BASE + lags + `regime_lag1` + `p_hmm*_tmrw`).
- **`walk_forward_cv_metrics`**: `TimeSeriesSplit` (optional `max_train_size` for rolling train), balanced sample weights, returns concatenated OOF preds + `classification_report` dict.

**Entry scripts (`scripts/`):** thin runners that import `training` and set script-level hyperparameters (XGBoost, `N_HMM_STATES`, `HMM_REFIT`, CV window).

### Import path (`sys.path`)

Python does not treat `src/` as a package root when you execute `python scripts/optimize.py` from the repo root. **`scripts/optimize.py`** therefore does:

```python
_ROOT = Path(__file__).resolve().parent.parent  # repository root
sys.path.append(str(_ROOT / 'src'))
import training
```

This appends the **`src`** directory to `sys.path` so `import training` resolves to **`src/training.py`** without an editable install. Other entrypoints should follow the same pattern (or set `PYTHONPATH=src`) so imports are deterministic from any working directory.

**Database:** `DEFAULT_DB_PATH` in `training.py` points to `src/Data/quant.db` (path anchored on `Path(__file__).resolve().parent` inside `src/`). Override via `build_xy(db_path=...)`.

**Features (illustrative):** `BASE` includes `log_return`, `rv_21d`, `vix`, `vrp`, `slope_2s10s`, `garch_var`, `amihud_liquidity`, `hy_oas`, `icsa` with `LAGS = [1, 5]`, plus `regime_lag1` and dynamic `p_hmm0_tmrw`…`p_hmm{N-1}_tmrw`.

**Key dependencies:** `duckdb`, `hmmlearn`, `arch`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`; **`shap`** for TreeExplainer in `predict.py`; **`optuna`** for `optimize.py`; notebook stack: `ib_insync`, `fredapi`, `nest_asyncio`.

## MLOps & Explicability

- **`logs/experiment_log.csv`**: append-only CSV with unique **`Experiment_ID`** (timestamp or `OPTUNA_BEST_*`), HMM meta, CV settings, **accuracy**, **Recall_Estável**, **F1_Weighted**, top features / best-params JSON.
- **`logs/preds_<Experiment_ID>.csv`**: walk-forward OOF predictions (dates, returns, actual vs predicted target) for diagnostics and **`scripts/backtest.py`**.
- **`scripts/predict.py`**: after the final `XGBClassifier` fit, runs **`shap.TreeExplainer`** on the latest row and prints the top contributions to the **predicted** shock class — local explanation of the boosted tree ensemble.

## Data Shielding (macro alignment)

Legacy SQL `LEFT JOIN macro_daily ON p.date = m.date` equates **calendar dates** across mixed-frequency series and can silently attach **future-revised** or **stale** macro prints relative to the equity bar. In **`load_merged_df`**, both legs are sorted; macro is forward-filled only **up to** each SPY row’s timestamp; then **`merge_asof(direction='backward')`** attaches the **last macro observation known on or before** each SPY date. That removes same-calendar **lookahead** in daily SPY vs FRED-style macro alignment.

## Important Notes

- **`src/main.py`** still uses a SQL join for its simpler load path — it is **descriptive** only. The **predictive** path uses **`training.load_merged_df`**.
- Target is **not** VIX-threshold labels: it is the **5d vol-shock ratio** banded into Queda / Estável / Alta — avoids using VIX as both label and dominant feature.
- XGBoost on macOS may need `brew install libomp`.
- IB + FRED data collection requires live gateway (e.g. port 4001) and API keys as in the notebook.
