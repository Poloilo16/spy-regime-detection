# 🏛️ QUANTITATIVE SYSTEM ARCHITECTURE (DO NOT IGNORE)

**⚠️ CRITICAL INSTRUCTION FOR AI AGENTS:** You are acting as a Senior Quantitative Developer. You MUST adhere strictly to the rules below. Violation of these rules, especially regarding latency, vectorization, or Tier separation, will result in immediate rejection of your code.

## 1. System Overview & Tech Stack
- **Domain:** High-Frequency/Mid-Frequency Trading Infrastructure and Regime Detection.
- **Primary Target:** SPY (S&P 500 ETF).
- **Core Languages:** Python (Pandas, NumPy, PySpark for research/Tier 1) and C++ (Tier 2 Execution Engine).
- **Database:** DuckDB (`Data/quant.db`).

## 2. Structural Tiers (Strict Isolation)
The system is divided into two distinct latency tiers. **NEVER** import or couple Tier 2 modules into Tier 1 modules.

### Tier 1: Macro & Regime Detection (Mid-Frequency)
- **Objective:** Predict 5-day realized volatility shock ratios.
- **Stack:** Hidden Markov Models (HMM), GARCH(1,1), XGBoost.
- **Rule:** Walk-forward validation is mandatory. No lookahead bias allowed (e.g., forbidden use of `.bfill()` on macro series like HY OAS).

### Tier 2: Microstructure & Execution (High-Frequency)
- **Objective:** Order book dynamics, directional liquidity, and execution routing.
- **Stack:** L1/L2 Tick Ingestion, C++ Gateway.
- **Key Metrics:** VPIN (Volume-Synchronized Probability of Informed Trading) and TIB (Trade Imbalance).
- **Rule:** This tier processes fractional volume bucketing. Latency is critical.

## 3. Immutable Coding Guidelines
1. **Mathematical Integrity > Clean Code:** Do not wrap high-frequency recursive loops (like GARCH variance updates) in standard Python `try/except` blocks. If performance is bottlenecked, use Numba (`@njit`) or vectorized NumPy/Pandas operations. 
2. **Database Pathing:** All database connections must use paths relative to the project root directory to ensure cross-compatibility between local Windows development environments and Linux/Oracle Cloud deployment servers. Example: `_ROOT.parent / 'Data' / 'quant.db'`.
3. **Logging:** Experiment predictions must be saved in dynamically timestamped CSVs within the `logs/` directory. Do not overwrite past experiment results.
4. **No Assumption of "Dead Code":** If a script exists in `scripts/` or `src/` but is not imported into `main.py`, assume it is an active quantitative research laboratory. **DO NOT DELETE IT.**