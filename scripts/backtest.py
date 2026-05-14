import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))

from training import FORWARD_HORIZON

LOGS_DIR = _ROOT / 'logs'


def _pred_class_to_signal(s: np.ndarray) -> np.ndarray:
    """0 → long +1, 1 → flat 0, 2 → short -1."""
    out = np.zeros(len(s), dtype=int)
    out[s == 0] = 1
    out[s == 2] = -1
    return out


def _apply_min_hold_rows(desired: np.ndarray, hold: int) -> np.ndarray:
    """Só reavalia a posição desejada a cada `hold` linhas (alinhado ao horizonte à frente)."""
    n = len(desired)
    if n == 0:
        return desired
    eff = np.empty(n, dtype=int)
    eff[0] = int(desired[0])
    last_eval = 0
    for i in range(1, n):
        if i - last_eval < hold:
            eff[i] = eff[i - 1]
        else:
            eff[i] = int(desired[i])
            last_eval = i
    return eff


def run_backtest_blindado():
    pred_files = sorted([f for f in os.listdir(LOGS_DIR) if f.startswith('preds_')])
    if not pred_files:
        return

    df = pd.read_csv(LOGS_DIR / pred_files[-1])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', kind='mergesort').reset_index(drop=True)

    custo_transacao = 0.0002
    pred_col = 'predicted_target'
    if pred_col not in df.columns:
        return

    desired = _pred_class_to_signal(df[pred_col].to_numpy(dtype=int))
    df['signal'] = _apply_min_hold_rows(desired, FORWARD_HORIZON)
    df['trades'] = pd.Series(df['signal']).diff().abs()

    if 'fwd_return_5d' in df.columns:
        df['strat_ret'] = (df['signal'] * df['fwd_return_5d']) - (df['trades'] * custo_transacao)
        df = df.dropna(subset=['strat_ret', 'fwd_return_5d'])
        total_ret = float((1.0 + df['strat_ret']).prod() - 1.0)
        vol_anual = df['strat_ret'].std() * np.sqrt(252)
        ret_anual = df['strat_ret'].mean() * 252
        sharpe = ret_anual / vol_anual if vol_anual != 0 else 0.0
        wealth = (1.0 + df['strat_ret']).cumprod()
        peak = wealth.expanding().max()
        mdd = float((wealth / peak - 1.0).min())
        label = (
            f'T→T+{FORWARD_HORIZON} (fwd_return_{FORWARD_HORIZON}d) + custos '
            f'+ reavaliação a cada {FORWARD_HORIZON} linhas'
        )
    else:
        if 'log_return' not in df.columns:
            print('CSV sem fwd_return_5d nem log_return; impossível backtest.')
            return
        df['next_day_ret'] = df['log_return'].shift(-1)
        df['strat_ret'] = (df['signal'] * df['next_day_ret']) - (df['trades'] * custo_transacao)
        df = df.dropna()
        cum_ret = df['strat_ret'].cumsum()
        vol_anual = df['strat_ret'].std() * np.sqrt(252)
        ret_anual = df['strat_ret'].mean() * 252
        sharpe = ret_anual / vol_anual if vol_anual != 0 else 0.0
        total_ret = float(np.exp(cum_ret.iloc[-1]) - 1.0)
        peak = cum_ret.expanding().max()
        mdd = float((np.exp(cum_ret - peak) - 1.0).min())
        label = f'T+1 (log_return) + custos + hold {FORWARD_HORIZON} linhas'

    print(f'--- BACKTEST ({label}) ---')
    print(f'Retorno Total: {total_ret:.2%}')
    print(f'Sharpe Ratio: {sharpe:.2f}')
    print(f'Max Drawdown: {mdd:.2%}')
    print(f'Número de Trades: {df["trades"].sum():.0f}')


if __name__ == '__main__':
    run_backtest_blindado()
