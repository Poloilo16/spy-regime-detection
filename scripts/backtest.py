import pandas as pd
import numpy as np
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
LOGS_DIR = _ROOT / 'logs'

def run_backtest_blindado():
    pred_files = sorted([f for f in os.listdir(LOGS_DIR) if f.startswith('preds_')])
    if not pred_files: return
    
    df = pd.read_csv(LOGS_DIR / pred_files[-1])
    
    # --- A CORREÇÃO CRUCIAL ---
    # O sinal de hoje (T) só opera o retorno de AMANHÃ (T+1)
    df['next_day_ret'] = df['log_return'].shift(-1)
    
    # --- CUSTOS DE TRANSAÇÃO ---
    # 0.02% por trade (estimativa conservadora para SPY/Futuros)
    custo_transacao = 0.0002 
    
    df['signal'] = 0
    df.loc[df['predicted_target'] == 0, 'signal'] = 1   # Long (Queda de Vol)
    df.loc[df['predicted_target'] == 2, 'signal'] = -1  # Short (Alta de Vol)
    
    # Verifica se houve mudança de posição para aplicar custo
    df['trades'] = df['signal'].diff().abs()
    
    # Cálculo do retorno líquido
    df['strat_ret'] = (df['signal'] * df['next_day_ret']) - (df['trades'] * custo_transacao)
    
    # --- MÉTRICAS ---
    df = df.dropna()
    cum_ret = df['strat_ret'].cumsum()
    
    # Sharpe Ratio Realista
    vol_anual = df['strat_ret'].std() * np.sqrt(252)
    ret_anual = df['strat_ret'].mean() * 252
    sharpe = ret_anual / vol_anual if vol_anual != 0 else 0
    
    total_ret = np.exp(cum_ret.iloc[-1]) - 1
    
    # Max Drawdown
    peak = cum_ret.expanding().max()
    dd = np.exp(cum_ret - peak) - 1
    mdd = dd.min()

    print(f"--- BACKTEST REALISTA (T+1 + Custos) ---")
    print(f"Retorno Total: {total_ret:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Número de Trades: {df['trades'].sum():.0f}")

if __name__ == "__main__":
    run_backtest_blindado()