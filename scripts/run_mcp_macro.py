"""
Bootstrap de significância estatística (timing de sinais vs. mercado fixo).
Lê o preds_*.csv mais recente, reconstrói o sinal por convicção e roda `bootstrap_test`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp.tester import bootstrap_test

LOGS_DIR = _REPO_ROOT / 'logs'

THRESH_ALTA_VOL = 0.4
THRESH_RISK_ON = -0.4
P_ESTAVEL_NEUTRAL = 0.5


def _latest_preds_path() -> Path | None:
    names = sorted(
        f for f in os.listdir(LOGS_DIR)
        if f.startswith('preds_') and f.endswith('.csv')
    )
    if not names:
        return None
    return LOGS_DIR / names[-1]


def _conviction_signal(df: pd.DataFrame) -> pd.Series:
    conv = df['p_alta'].astype(float) - df['p_queda'].astype(float)
    sig = np.zeros(len(df), dtype=np.int8)
    stable = df['p_estavel'].to_numpy(dtype=float) > P_ESTAVEL_NEUTRAL
    active = ~stable
    sig[active & (conv.to_numpy(dtype=float) > THRESH_ALTA_VOL)] = -1
    sig[active & (conv.to_numpy(dtype=float) < THRESH_RISK_ON)] = 1
    sig[stable] = 0
    return pd.Series(sig, index=df.index, name='signal')


def _n_trades(signal: pd.Series) -> int:
    return int(signal.astype(float).diff().abs().fillna(0.0).sum())


def main() -> None:
    path = _latest_preds_path()
    if path is None:
        print(f'Nenhum preds_*.csv em {LOGS_DIR}')
        return

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', kind='mergesort').reset_index(drop=True)

    need = {'p_queda', 'p_estavel', 'p_alta', 'fwd_return_5d'}
    if not need.issubset(df.columns):
        print(f'Colunas obrigatórias em falta: {sorted(need - set(df.columns))}')
        return

    signals = _conviction_signal(df)
    # Entrada no fecho de T; retorno alocado ao período subsequente (sem lookahead no forward).
    market_returns = df['fwd_return_5d'].astype(float).shift(1)
    df_work = pd.DataFrame({'signal': signals, 'mkt': market_returns}, index=df.index)
    df_ok = df_work.dropna()
    if df_ok.empty:
        print('Série vazia após alinhar sinal com fwd_return_5d.shift(1).')
        return

    out = bootstrap_test(
        df_ok['signal'],
        df_ok['mkt'],
        metric='sharpe',
        n_bootstraps=2000,
    )

    n_trades = _n_trades(df_ok['signal'])
    sharpe_real = out['real']
    pval = out['pvalue']
    edge = pval < 0.05
    concl = 'Edge Significativo (p < 0.05)' if edge else 'Sem evidência de edge a 5% (p >= 0.05)'

    print('--- MCP MACRO — Bootstrap de Sharpe ---')
    print(f'Preds         : {path}')
    print(f'Observações   : {len(df_ok)}')
    print(f'N trades      : {n_trades}')
    print(f'Sharpe real   : {sharpe_real:.4f}')
    print(f'P-valor       : {pval:.4f}')
    print(f'Conclusão     : {concl}')


if __name__ == '__main__':
    main()
