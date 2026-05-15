# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative research pipeline for **SPY market regime / volatility-shock detection**. Collects time series from Interactive Brokers and FRED, stores them in DuckDB, engineers features (GARCH, rolling HMM forward probabilities, macro), and predicts a **5-day realized-vol shock ratio** target (3 classes: Queda / Estável / Alta) with **XGBoost** under walk-forward validation. No lint or test harness — research-grade codebase with CSV experiment logging.

## Running the Pipeline

All scripts are run from the repository root:

```bash
python scripts/predict.py          # build_xy → walk-forward CV → final fit → SHAP → logs/
python scripts/optimize.py         # Optuna search (pip install optuna); appends best trial to logs/
python scripts/backtest.py         # hard-class signal backtest on latest preds_*.csv
python scripts/backtest_v2.py      # probability-surface conviction backtest + equity_curve_v2.png
python scripts/run_mcp_macro.py    # bootstrap Sharpe p-value significance test on latest preds
python scripts/update_db.py        # refresh macro_daily (FRED: HY_OAS, ICSA) + Amihud recalc
python src/main.py                 # global HMM → regime_plot.png (descriptive/exploratory only)
```

## Architecture

### Core library: `src/training.py`

Single source of truth — import-safe, no side effects at import time.

| Function | Purpose |
|---|---|
| `load_merged_df` | Loads SPY from `prices_daily`, pivots `macro_daily`, joins with `merge_asof(direction='backward')` — see Data Shielding below |
| `add_regime` | Expanding-window `rv_21d` quartiles → 4 regime labels (0=Low vol … 3=Crisis) |
| `add_garch` | Rolling GARCH(1,1): full MLE every `garch_refit=21` days, recursive variance update `h_{t+1} = ω + α·r_t² + β·h_t` between refits |
| `add_hmm_forward_proba` | Rolling HMM refit every `HMM_REFIT` steps; incremental forward filter between refits; outputs `p_hmm{s}_tmrw` = P(state tomorrow | data ≤ t) |
| `add_target` | 5-day realized-vol shock ratio banded into 3 classes; not VIX threshold labels |
| `add_feature_matrix` | BASE features + lags + `regime_lag1` + HMM forward probabilities |
| `build_xy` | Orchestrates the full pipeline; returns `(df, features, X, y)` |
| `walk_forward_cv_metrics` | `TimeSeriesSplit` with balanced sample weights; returns OOF preds + classification report |
| `apply_trade_confidence_filter` | Forces class 1 (Estável) when directional confidence < `TRADE_CONFIDENCE_THRESHOLD` (0.65) |
| `apply_min_hold_rows` | Enforces minimum holding period: re-evaluates desired signal only every `hold` rows |

### Entry scripts: `scripts/`

Thin runners that set script-level hyperparameters and call `training` functions. Each appends a row to `logs/experiment_log.csv` and writes `logs/preds_<Experiment_ID>.csv`.

`optimize.py` caches `load_merged_df` + `add_regime` + `add_garch` once at module level, and caches `add_hmm_forward_proba` results keyed on `(n_hmm_states, hmm_refit)` — at most 12 unique HMM fits across 50 Optuna trials.

### Statistical testing: `mcp/tester.py`

`bootstrap_test(signals, market_returns, metric, n_bootstraps)` — vectorised Monte Carlo bootstrap that tests whether signal timing beats i.i.d. random permutations of the same signal distribution. Called by `scripts/run_mcp_macro.py`.

## Import Path Convention

Scripts do **not** use an editable install. Every entry script resolves `src/` and adds it to `sys.path` at the top:

```python
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))
import training
```

`DEFAULT_DB_PATH` in `training.py` is anchored on `Path(__file__).resolve().parent.parent / 'Data' / 'quant.db'` (relative to `src/training.py`, not the CWD). Override via `build_xy(db_path=...)`.

## Data Shielding (Macro Alignment)

`src/main.py` uses a raw SQL `LEFT JOIN ON date` — calendar-date match only, acceptable for its descriptive purpose.

`load_merged_df` (the **predictive** path) instead:
1. Sorts both SPY and macro legs by date.
2. Forward-fills macro within its own series.
3. `pd.merge_asof(df, macro_wide, on='date', direction='backward')` — attaches the **last macro observation known on or before** each SPY date.

## Known Data Gaps

**HY_OAS (`BAMLH0A0HYM2`):** ICE BofA changed their FRED licensing around 2022-2023. The series now only returns data from **2023-05-15** onward regardless of `observation_start`. SPY history starts 2016-05-04, leaving 7 years without real HY spread data. `load_merged_df` fills pre-2023 rows with `0.0` (neutral constant via `ffill().fillna(0.0)`) — **not** `bfill`, which would backfill a 2023 value into 2016-2022 training rows and introduce lookahead bias. If real HY spread history is needed, source it from Bloomberg (USHY OAS) or construct a proxy from Treasuries + corporates.

**ICSA:** Weekly frequency, 523 rows for 2016-2026. `merge_asof` forward-fills correctly — each daily SPY bar gets the last known weekly print. No coverage issue.

**`amihud_liquidity`:** Computed column — run `scripts/update_db.py` (or the inline SQL in that script) if the column is missing from `prices_daily`. Required by `load_merged_df`.

## MLOps / Logging

- **`logs/experiment_log.csv`** — append-only; columns: `Experiment_ID`, `Timestamp`, `HMM_FEATURES`, `Lags`, `Alvo_Banda`, `CV_Janela`, `CV_max_train_size`, `Acurácia`, `Recall_Estável`, `F1_Weighted`, `Top_5_Features`.
- **`logs/preds_<Experiment_ID>.csv`** — OOF walk-forward predictions: `date`, `close`, `p_queda`, `p_estavel`, `p_alta`, `log_p_*`, `actual_target`, `pred_raw`, `predicted_target`, `fwd_return_5d`. Consumed by all three backtest scripts.
- `predict.py` runs `shap.TreeExplainer` on the **latest row** and prints top 5 SHAP contributions for the predicted class.

## Database Schema

`Data/quant.db` (DuckDB — single-writer, no concurrent connections):
- **`prices_daily`**: `(date, ticker, open, high, low, close, volume, log_return, rv_21d, vrp, amihud_liquidity, ...)` — SPY daily panel.
- **`macro_daily`**: `(date, series_id, value)` — long format; series: `VIX`, `2S10S`, `HY_OAS` (2023+), `ICSA`.

## Key Design Decisions

- **Target**: 5-day realized-vol shock ratio (fwd_rv / bwd_rv − 1) banded at ±15%, not VIX bins — avoids using VIX as both label and dominant feature.
- **HMM state ordering**: sorted by mean VIX in-sample at each refit so state indices are comparable across refits.
- **GARCH refit cadence**: full MLE every 21 days, recursive update between refits — ~95% fewer MLE fits vs. daily-refit approach with negligible accuracy impact.
- **Optuna objective**: maximizes F1(Estável / class 1). Trades ~20pp global accuracy for better recall on the minority stable-regime class — known structural trade-off (see `md/notes.md`).
- **Backtest signal convention**: class 0 (Queda) → long +1, class 1 (Estável) → flat 0, class 2 (Alta vol) → short −1. Transaction cost = 0.02% per trade.
- **Bootstrap significance** (`run_mcp_macro.py`): p=0.0075 on the 2026-05-14 run — signal timing is statistically significant vs. random permutations, even when simple backtest P&L is weak due to costs and the discrete 5-day hold.

## System Dependencies

- `brew install libomp` required for XGBoost on macOS.
- `scripts/update_db.py` requires a FRED API key (hardcoded in the file); IB data collection requires a live IB Gateway on port 4001.
- `optuna` is optional — only needed for `optimize.py`.
- `shap` required for `predict.py` (`pip install shap`).
- DuckDB does not allow concurrent connections — close DBeaver / any other client before running pipeline scripts.
