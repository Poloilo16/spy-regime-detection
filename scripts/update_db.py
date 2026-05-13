import duckdb
from fredapi import Fred
import pandas as pd
from pathlib import Path

# Ajuste automático do caminho para o seu ambiente Windows
DB_PATH = str(Path(__file__).resolve().parent / 'Data' / 'quant.db')

print("Conectando ao banco de dados...")
conn = duckdb.connect(DB_PATH)

# 1. Cálculo da Liquidez de Amihud
print("Calculando Liquidez de Amihud...")
conn.execute("ALTER TABLE prices_daily ADD COLUMN IF NOT EXISTS amihud_liquidity DOUBLE")
conn.execute("""
    UPDATE prices_daily
    SET amihud_liquidity = sub.amihud
    FROM (
        SELECT date, ticker, 
               (ABS(log_return) / NULLIF(volume, 0)) * 1000000 AS amihud
        FROM prices_daily
    ) sub
    WHERE prices_daily.date = sub.date AND prices_daily.ticker = sub.ticker
""")

# 2. Download dos Fatores Macro (FRED)
print("Baixando dados Macro (HY_OAS e ICSA) via FRED API...")
# Usando a sua chave da API que estava no notebook original
fred = Fred(api_key='6c8b2beafa3484a2becf6e8d428f7f76') 

series = {
    'HY_OAS': 'BAMLH0A0HYM2',
    'ICSA': 'ICSA'
}

records = []
for series_id, fred_code in series.items():
    print(f"  Puxando {series_id}...")
    s = fred.get_series(fred_code, observation_start='2016-05-03')
    for date, value in s.dropna().items():
        records.append({'date': date.date(), 'series_id': series_id, 'value': value})

if records:
    macro_df = pd.DataFrame(records)
    conn.execute("INSERT OR IGNORE INTO macro_daily SELECT date, series_id, value FROM macro_df")
    print(f"Inseridos {len(macro_df)} registros na macro_daily.")

print("\nSucesso! Banco de dados atualizado.")
conn.close()