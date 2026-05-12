import warnings
warnings.filterwarnings('ignore')

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

DB_PATH = '/Users/lucaszelmanovits/Desktop/Quant/Research/Data/quant.db'

# ── 1. Load & clean ──────────────────────────────────────────────────────────
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

# ── 2. Standardize ───────────────────────────────────────────────────────────
FEATURES = ['log_return', 'rv_21d', 'vix', 'vrp', 'slope_2s10s']
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURES])

# ── 3. Fit HMM models & compute BIC ─────────────────────────────────────────
def count_params(n_states, n_features):
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    cov   = n_states * n_features * (n_features + 1) // 2
    return trans + means + cov

rows, fitted = [], {}
for n in [2, 3, 4]:
    m = GaussianHMM(n_components=n, covariance_type='full', n_iter=1000, random_state=42)
    m.fit(X)
    ll  = m.score(X) * len(X)
    k   = count_params(n, len(FEATURES))
    bic = k * np.log(len(X)) - 2 * ll
    rows.append({'n_states': n, 'log_likelihood': round(ll, 2), 'n_params': k, 'BIC': round(bic, 2)})
    fitted[n] = m

summary = pd.DataFrame(rows)
print("\nHMM model selection:")
print(summary.to_string(index=False))

# ── 4. Best model — Viterbi + smoothed probs ─────────────────────────────────
best_n    = int(summary.loc[summary['BIC'].idxmin(), 'n_states'])
model     = fitted[best_n]
print(f"\nBest model: {best_n} states")

hard = model.predict(X)
soft = model.predict_proba(X)

# ── 5. Reorder states: state 0 = lowest rv_21d ───────────────────────────────
rv_col      = FEATURES.index('rv_21d')
mean_rv     = [X[hard == s, rv_col].mean() for s in range(best_n)]
order       = np.argsort(mean_rv)
remap       = np.empty_like(order)
remap[order] = np.arange(best_n)

hard = remap[hard]
soft = soft[:, order]

df['regime'] = hard
for s in range(best_n):
    df[f'prob_{s}'] = soft[:, s]

# ── 6. Regime validation ─────────────────────────────────────────────────────
print("\nRegime statistics:")
print(df.groupby('regime')[['log_return', 'rv_21d', 'vix']].agg(['mean', 'std']).round(4))

# ── 7. Plot ───────────────────────────────────────────────────────────────────
COLORS = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'][:best_n]
dates  = pd.to_datetime(df.date).values

def shade_regimes(ax, dates, states, colors):
    changes = np.concatenate([[0], np.where(np.diff(states))[0] + 1, [len(states)]])
    for i in range(len(changes) - 1):
        s = states[changes[i]]
        ax.axvspan(dates[changes[i]], dates[changes[i + 1] - 1],
                   alpha=0.25, color=colors[s], linewidth=0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})

shade_regimes(ax1, dates, hard, COLORS)
ax1.plot(dates, df.close, color='black', linewidth=0.8, zorder=3)
ax1.set_ylabel('SPY Close')
ax1.set_title(f'SPY Regime Detection — {best_n}-State HMM')
patches = [mpatches.Patch(color=COLORS[s], alpha=0.6, label=f'Regime {s}') for s in range(best_n)]
ax1.legend(handles=patches, loc='upper left')

for s in range(best_n):
    ax2.plot(dates, df[f'prob_{s}'], color=COLORS[s], linewidth=0.8, label=f'Regime {s}')
ax2.set_ylabel('Probability')
ax2.set_ylim(0, 1)
ax2.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('regime_plot.png', dpi=150)
plt.show()
print("Saved regime_plot.png")
