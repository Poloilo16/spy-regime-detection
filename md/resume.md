# Resumo do projeto — lógica ficheiro a ficheiro e ligações

Este documento descreve o pipeline **coleta → DuckDB → `src/training.py` (build_xy, HMM, GARCH) → scripts (`predict`, `optimize`, `backtest`)** após a modularização MLOps.

---

## 1. Visão geral do fluxo

1. **`notebook.ipynb`** e **`scripts/update_db.py`** alimentam **`src/Data/quant.db`** (DuckDB): `prices_daily`, `macro_daily`.
2. **`src/training.py`** concentra toda a lógica preditiva partilhada: fusão temporal à prova de *lookahead*, etiquetas de regime (`rv_21d` expansivo), GARCH rolling, HMM com filtro *forward*, alvo de **choque de vol a 5 dias**, matriz de features e walk-forward CV.
3. **`scripts/predict.py`** — importa `training`, corre `build_xy`, validação temporal, modelo final, **SHAP (`TreeExplainer`)**, regista **`logs/experiment_log.csv`** e **`logs/preds_<id>.csv`**.
4. **`scripts/optimize.py`** — Optuna sobre hiperparâmetros (XGB + `N_HMM_STATES` + `HMM_REFIT`); acrescenta linha `OPTUNA_BEST_*` ao mesmo CSV de experiências.
5. **`scripts/backtest.py`** — lê o último `preds_*.csv`, aplica sinal em **T+1** sobre `log_return`, custos de transação, imprime Sharpe / drawdown.
6. **`src/main.py`** — ramo **descritivo**: HMM global em toda a série, `regime_plot.png`; **não** alimenta o XGBoost.

```
[notebook + update_db]  →  src/Data/quant.db
                              │
                    ┌─────────┴─────────┐
                    v                   v
            src/training.py      src/main.py
            (build_xy, WF)        (HMM global, PNG)
                    │
        ┌───────────┼───────────┐
        v           v           v
  predict.py   optimize.py  backtest.py
```

---

## 2. Resolução de imports: `sys.path.append`

Os *entry points* em **`scripts/`** não estão dentro do pacote `src`. Em **`scripts/optimize.py`** usa-se:

```python
_ROOT = Path(__file__).resolve().parent.parent  # raiz do repositório
sys.path.append(str(_ROOT / 'src'))
import training
```

Isto acrescenta **`src`** ao `sys.path`, permitindo `import training` → ficheiro **`src/training.py`** sem `pip install -e .`. **Recomendação:** replicar o mesmo padrão em qualquer script que importe `training` (ou exportar `PYTHONPATH=src`) para que `python scripts/...` funcione a partir de qualquer diretório de trabalho.

---

## 3. Blindagem de dados: `merge_asof` em vez de `JOIN` na carga preditiva

No ramo preditivo, **`load_merged_df`** em `training.py`:

1. Carrega SPY (`prices_daily`) e macro em formato largo, **ordenado por `date`** (`kind='mergesort'` onde aplicável).
2. Faz *forward-fill* só na série macro **antes** do *as-of join* (informação publicada até aquela data, não do futuro).
3. Une com **`pd.merge_asof(df_spy, macro_wide, on='date', direction='backward')`**.

Com **`direction='backward'`**, cada barra diária do SPY recebe **a última observação macro cuja data é ≤ à data do SPY**. Isto corrige o risco do **`LEFT JOIN ... ON p.date = m.date`**: em calendários mistos (FRED com revisões, *lags* de divulgação, buracos), o *join* por igualdade de data pode alinhar um valor que **ainda não existia** no instante de decisão ou ignorar a ordem temporal real. O *merge asof* **backward** implementa a mesma disciplina que um trader teria ao olhar só para dados **já disponíveis** até ao fecho do dia de referência — eliminando *lookahead bias* de alinhamento na fronteira SPY vs macro.

> **Nota:** `src/main.py` mantém um `SELECT` com `JOIN` clássico para visualização rápida; o pipeline **numérico** que alimenta XGBoost deve usar sempre **`training.load_merged_df`**.

---

## 4. Base de dados DuckDB (`src/Data/quant.db`)

### 4.1. `prices_daily`

Uma linha por **(date, ticker)**; SPY como ativo central. Colunas típicas: OHLCV, `log_return`, `rv_*`, `vrp`, `amihud_liquidity`, etc.

### 4.2. `macro_daily`

Chave lógica **`(date, series_id, value)`**. Exemplos: `VIX`, `2S10S`, `HY_OAS`, `ICSA`.

---

## 5. `src/training.py` — núcleo

- **`load_merged_df`**: fusão SPY + macro com `merge_asof`.
- **`add_regime`**: quartis expansivos de `rv_21d` (`MIN_WINDOW=63`) → classes 0–3 (nomes Low … Crisis); usado como **`regime_lag1`** nas features.
- **`add_garch`**: GARCH(1,1) em janela rolling de 252 dias até **t**, variância condicional a um passo em **t**.
- **`add_hmm_forward_proba`**: HMM Gaussiano com `covariance_type='full'` em observações `HMM_FEATURES` (p.ex. `log_return`, `vix`); refit a cada **`hmm_refit`** dias; entre refits, passo de predição + atualização Bayesiana; **P(estado amanhã | dados até t)** → colunas `p_hmm*_tmrw` (reordenadas por média de VIX in-sample no refit).
- **`add_target`**: razão de choque de RV 5d → classes 0/1/2.
- **`build_xy`**: encadeia tudo e devolve `df`, lista de features, `X`, `y`.
- **`walk_forward_cv_metrics`**: `TimeSeriesSplit`, pesos balanceados por *fold*, métricas e preds OOF.

---

## 6. `scripts/predict.py`

- Chama `build_xy(DB_PATH, N_HMM_STATES, HMM_REFIT, ...)`.
- Walk-forward + relatório + matriz de confusão.
- Treino final em todo o `X,y`.
- **SHAP:** `TreeExplainer` sobre a última linha; imprime top 5 impactos na classe prevista.
- **Logging:** `Experiment_ID` temporal; append a **`logs/experiment_log.csv`**; grava **`logs/preds_<Experiment_ID>.csv`** com datas e alvos para *backtest*.

---

## 7. `scripts/optimize.py`

- Optuna: objetivo = **maximizar F1 da classe Estável (1)** (`f1_stable` retornado por `walk_forward_cv_metrics`).
- O estudo mostrou trade-off forte: melhor trial logado (**`HMM_REFIT=63`**, **`max_depth=3`**, **`N_HMM_STATES=3`**) com **acurácia global ~47.7%** vs. configurações mais “ricas” que historicamente atingiam **~69%** de acurácia em configurações orientadas à precisão global (e vs. ~52% em corridas recentes do mesmo alvo de choque — ver CSV). Conclusão: **VIX/VRP são *shock hunters***; não trazem assinatura limpa de **mean-reversion** para a classe estável.

---

## 8. `scripts/backtest.py`

- Usa o ficheiro `preds_` mais recente em **`logs/`**.
- **`next_day_ret = log_return.shift(-1)`** — sinal em **T** só afeta retorno **T+1** (*lookahead* de execução evitado).
- Custos proporcionais a mudanças de posição.

---

## 9. `src/main.py`

Carga via SQL + HMM global + gráfico. Independente do XGBoost.

---

## 10. `notebook.ipynb` / `scripts/update_db.py`

Sem alteração de papel: ingestão IB/FRED, features em SQL/Python, escrita DuckDB.

---

## 11. Documentação

| Ficheiro | Função |
|----------|--------|
| **`md/CLAUDE.md`** | Guia para agentes / devs: comandos, arquitetura, `merge_asof`, SHAP, logs. |
| **`md/notes.md`** | Estado do projeto, limitações (incl. Optuna), próximos passos. |
| **`md/results.md`** | Métricas de referência e linhas Optuna (sincronizar com `logs/`). |

---

## 12. Ordem sugerida no dia a dia

1. Atualizar dados (notebook / `update_db.py`).
2. `python scripts/predict.py` — métricas + SHAP + logs.
3. Opcional: `python scripts/optimize.py` — hiperparâmetros.
4. Opcional: `python scripts/backtest.py` — leitura económica minimalista.
5. Opcional: `python src/main.py` — figura descritiva.

---

*Se alterar `RV_SHOCK_*`, `BASE`, `HMM_REFIT` default ou o objetivo do Optuna, atualize `md/results.md` e este resumo em conjunto.*
