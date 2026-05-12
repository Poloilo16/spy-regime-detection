# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative research project for **market regime detection** on SPY. The pipeline collects time series data from Interactive Brokers and FRED, stores it in DuckDB, classifies volatility regimes via HMM, and predicts tomorrow's regime using GARCH + XGBoost.

## Running the Code

```bash
python main.py      # HMM regime detection → regime_plot.png
python predict.py   # GARCH + XGBoost regime prediction
jupyter notebook notebook.ipynb  # data collection pipeline
```

No build, lint, or test infrastructure — this is a research project.

## Architecture

**Three-stage pipeline:**

1. **Data collection** (`notebook.ipynb`): Connects to Interactive Brokers (port 4001) to pull 10 years of SPY OHLCV and VIX data, fetches yield curve slope (10Y-2Y) from FRED, computes derived features (log returns, realized volatility windows, VRP, vol-of-vol), writes to DuckDB.

2. **HMM regime detection** (`main.py`): Reads from DuckDB, standardizes features, fits GaussianHMMs with 2–4 states, selects best model by BIC, generates Viterbi regime assignments + soft probabilities, reorders states by ascending `rv_21d` mean (state 0 = lowest vol), saves `regime_plot.png`.

3. **Regime prediction** (`predict.py`): Labels regimes using `rv_21d` expanding-window quartiles (`MIN_WINDOW=63` days). Fits GARCH(1,1) for conditional variance. Runs a rolling HMM forward filter (`MIN_HMM=252`, refit every `HMM_REFIT=21` days) that produces four `p_hmmN_tmrw` features — P(state_T+1 | observations_0:T) — using the incremental Bayesian filter (predict via `transmat_`, update via diagonal emission probability) to avoid reprocessing the full window each day. Label switching is resolved at every refit by reordering states to ascending `rv_21d` mean. Trains an XGBoost classifier with walk-forward CV (5 splits). Outputs CV metrics, feature importances, and a next-day prediction with probabilities. `predict.py` takes ~60 seconds to run due to the rolling HMM loop.

**Database** (`Data/quant.db` — DuckDB):
- `prices_daily`: SPY OHLCV + `log_return`, `rv_5d`/`rv_21d`/`rv_63d`, `vol_of_vol`, `vol_term_structure`, `vrp`, `return_autocorr`
- `macro_daily`: series_id keyed rows for `VIX` and `2S10S` (10Y-2Y spread)

**Features fed to XGBoost (23 total):** `log_return`, `rv_21d`, `vix`, `vrp`, `slope_2s10s`, `garch_var` — each with lags 1 and 5 (`LAGS = [1, 5]`) — plus `regime_lag1` and four `p_hmmN_tmrw` columns. The HMM columns are not lagged (they already represent tomorrow's probabilities). Top features: `regime_lag1` (63%), `rv_21d` (9%), `p_hmm3_tmrw`/`p_hmm0_tmrw` (~2% each, combined HMM importance ~7%).

**Key dependencies:** `duckdb`, `hmmlearn`, `arch` (GARCH), `xgboost`, `ib_insync`, `fredapi`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`

## Important Notes

- Both `main.py` and `predict.py` hardcode the path to `Data/quant.db` — update `DB_PATH` if the database moves.
- `predict.py` uses `rv_21d` expanding-window quartile labels — not VIX thresholds and not the HMM regimes from `main.py`. Using VIX both as the label definition and as a feature creates circularity (the model would just learn VIX autocorrelation); the expanding quartile approach forces genuine prediction of future realized volatility state.
- XGBoost requires `libomp` on macOS: `brew install libomp`.
- Data collection in the notebook requires an active Interactive Brokers TWS/Gateway session on port 4001 and a FRED API key.
- The notebook uses `nest_asyncio` to patch the event loop for `ib_insync` in Jupyter.
- Python environment is managed via conda (per `.vscode/settings.json`).
