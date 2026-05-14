"""
Backtest v2 — sinal a partir da superfície de probabilidade (p_queda, p_estavel, p_alta),
com holding alinhado ao horizonte à frente e custos por mudança de posição.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = _ROOT / 'logs'

# Deve coincidir com `training.FORWARD_HORIZON` (import de `training` puxa duckdb/arch).
FORWARD_HORIZON = 5

# Institucional: zona de convicção na diferença p_alta − p_queda
THRESH_ALTA_VOL = 0.4
THRESH_RISK_ON = -0.4
P_ESTAVEL_NEUTRAL = 0.5
TX_COST = 0.0002


def find_latest_preds_csv(logs_dir: Path) -> Path | None:
    files = sorted(f for f in os.listdir(logs_dir) if f.startswith('preds_') and f.endswith('.csv'))
    if not files:
        return None
    return logs_dir / files[-1]


def load_preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {'p_queda', 'p_estavel', 'p_alta', 'fwd_return_5d', 'date'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f'CSV incompleto; faltam colunas: {sorted(missing)}')
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date', kind='mergesort').reset_index(drop=True)


def conviction_score(df: pd.DataFrame) -> pd.Series:
    return df['p_alta'].astype(float) - df['p_queda'].astype(float)


def desired_signal_surface(df: pd.DataFrame) -> np.ndarray:
    """
    Vetorizado:
    - p_estavel > 0.5 → 0 (neutro absoluto)
    - conviction > 0.4 → -1 (alta convicção em explosão de vol / pressão de p_alta)
    - conviction < -0.4 → +1 (risk-on / pressão de p_queda)
    - caso contrário → 0
    """
    p_e = df['p_estavel'].to_numpy(dtype=float)
    conv = conviction_score(df).to_numpy(dtype=float)
    sig = np.zeros(len(df), dtype=np.int8)
    stable = p_e > P_ESTAVEL_NEUTRAL
    active = ~stable
    sig[active & (conv > THRESH_ALTA_VOL)] = -1
    sig[active & (conv < THRESH_RISK_ON)] = 1
    sig[stable] = 0
    return sig


def apply_min_hold_rows(desired: np.ndarray, hold: int) -> np.ndarray:
    """Reavalia o sinal desejado apenas a cada `hold` linhas (mandato de maturação)."""
    n = len(desired)
    if n == 0:
        return desired.astype(np.int8)
    eff = np.empty(n, dtype=np.int8)
    eff[0] = int(desired[0])
    last_eval = 0
    for i in range(1, n):
        if i - last_eval < hold:
            eff[i] = eff[i - 1]
        else:
            eff[i] = int(desired[i])
            last_eval = i
    return eff


def attach_strategy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['conviction_score'] = conviction_score(out)
    desired = desired_signal_surface(out)
    out['signal_desired'] = desired
    held = apply_min_hold_rows(desired, FORWARD_HORIZON)
    out['signal'] = held
    out['trades'] = pd.Series(out['signal']).diff().abs().fillna(0.0)
    fwd = out['fwd_return_5d'].astype(float)
    out['strat_ret'] = out['signal'].astype(float) * fwd - out['trades'] * TX_COST
    return out


def period_factor() -> float:
    """Períodos de ~5 dias úteis por ano (Sharpe / média anualizada)."""
    return 252.0 / float(FORWARD_HORIZON)


def compute_metrics(strat_ret: pd.Series) -> dict[str, float]:
    r = strat_ret.dropna().astype(float)
    if r.empty:
        return {
            'total_return': 0.0,
            'sharpe_ann': 0.0,
            'max_drawdown': 0.0,
            'n_periods': 0.0,
        }
    wealth = (1.0 + r).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    vol = float(r.std(ddof=0))
    mu = float(r.mean())
    ann = period_factor()
    sharpe_ann = (mu / vol) * np.sqrt(ann) if vol > 0 else 0.0
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    max_drawdown = float(dd.min())
    return {
        'total_return': total_return,
        'sharpe_ann': sharpe_ann,
        'max_drawdown': max_drawdown,
        'n_periods': float(len(r)),
    }


def save_equity_curve(dates: pd.Series, wealth: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, wealth.values, color='#1f77b4', linewidth=1.2, label='Equity (cumprod)')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_title('Backtest v2 — curva de equity')
    ax.set_xlabel('Data')
    ax.set_ylabel('Wealth index')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_backtest_v2(
    *,
    preds_path: Path | None = None,
    save_plot: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    logs = LOGS_DIR if preds_path is None else preds_path.parent
    path = preds_path or find_latest_preds_csv(logs)
    if path is None or not path.is_file():
        raise FileNotFoundError(f'Nenhum preds_*.csv em {logs}')

    base = load_preds(path)
    df = attach_strategy_columns(base)
    df = df.dropna(subset=['strat_ret', 'fwd_return_5d']).reset_index(drop=True)

    metrics = compute_metrics(df['strat_ret'])
    wealth = (1.0 + df['strat_ret']).cumprod()

    plot_path = LOGS_DIR / 'equity_curve_v2.png'
    if save_plot and not df.empty:
        save_equity_curve(df['date'], wealth, plot_path)

    return df, metrics | {'preds_file': str(path), 'equity_plot': str(plot_path) if save_plot else ''}


def main() -> None:
    try:
        df, m = run_backtest_v2()
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    print('--- BACKTEST V2 (superfície de probabilidade) ---')
    print(f"Arquivo       : {m['preds_file']}")
    print(f"Observações   : {int(m['n_periods'])}")
    print(f"Retorno total : {m['total_return']:.2%}")
    print(f"Sharpe (ann., ~{FORWARD_HORIZON}d freq): {m['sharpe_ann']:.2f}")
    print(f"Max drawdown  : {m['max_drawdown']:.2%}")
    print(f"Trades (|ds|) : {df['trades'].sum():.0f}")
    if m.get('equity_plot'):
        print(f"Gráfico       : {m['equity_plot']}")


if __name__ == '__main__':
    main()
