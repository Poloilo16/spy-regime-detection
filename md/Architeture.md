# Arquitetura do Sistema Quant (SPY Regime Detection)

## 1. Ojetivo Principal
Sistema de negociação quantitativa dividido em dois níveis de latência.
- **Tier 1 (Macro - Python):** Deteção de Regimes de Mercado (HMM + XGBoost) usando dados diários (DuckDB).
- **Tier 2 (Micro - C++/Python):** Execução e microestrutura (VPIN, Trade Imbalance) em alta frequência no SPY (L1/L2).

## 2. Regras Intocáveis (Imutáveis)
- NUNCA misturar a lógica de predição do Tier 1 com a lógica de execução do Tier 2.
- O banco de dados central é o DuckDB (`Data/quant.db`). Os caminhos devem ser relativos à raiz do projeto.
- A matemática financeira (ex: cálculo do VPIN e TIB) tem prioridade sobre "otimizações de código genéricas". Não simplifique loops que afetem a precisão de ponto flutuante de dados financeiros.