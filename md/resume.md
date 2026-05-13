# Resumo do projeto — lógica ficheiro a ficheiro e ligações

Este documento descreve, ponto a ponto, o que cada componente faz e como o pipeline **coleta dados → DuckDB → análise (HMM) → predição (GARCH + HMM + XGBoost)** se encadeia.

---

## 1. Visão geral do fluxo

1. **`notebook.ipynb`** (e, em parte, **`update_db.py`**) alimentam o ficheiro **`Data/quant.db`** (DuckDB): tabelas `prices_daily` e `macro_daily`.
2. **`main.py`** lê a mesma base, ajusta um **HMM em toda a série** (exploração / visualização) e grava **`regime_plot.png`**. Não grava regimes na base; é um ramo **descritivo**.
3. **`predict.py`** lê a mesma base, constrói **rótulos e features** (incluindo GARCH rolling, HMM rolling e XGBoost), avalia com **walk-forward CV** e imprime a **predição** para o último dia disponível. É o ramo **predítivo**.

```
[IB + FRED + SQL no notebook]     [update_db.py: FRED + Amihud + macro]
              \                              /
               v                            v
                    Data/quant.db
                    /           \
                   v             v
              main.py       predict.py
           (HMM global)    (GARCH + HMM + XGB)
                |                 |
         regime_plot.png     stdout (métricas + próximo choque)
```

---

## 2. Base de dados DuckDB (`Data/quant.db`)

### 2.1. `prices_daily`

- Uma linha por **(date, ticker)**; o SPY é o ativo central.
- Colunas típicas (consoante o notebook e updates): OHLCV, **`log_return`**, volatilidades realizadas **`rv_*`**, **`vrp`**, derivadas de vol (**`vol_of_vol`**, etc., conforme células do notebook), **`amihud_liquidity`** (pode ser calculada no notebook e/ou em `update_db.py`).

### 2.2. `macro_daily`

- Chave lógica: **`(date, series_id, value)`**.
- **`series_id`** usados no projeto incluem, entre outros: **`VIX`**, **`2S10S`** (inclinação da curva 10Y−2Y), **`HY_OAS`**, **`ICSA`** (pedidos iniciais de desemprego), conforme o notebook e `update_db.py` populam.

Os scripts Python resolvem o caminho com **`Path(__file__).parent / 'Data' / 'quant.db'`**, para funcionar em qualquer máquina desde que a pasta `Data/` exista ao lado dos `.py`.

---

## 3. `notebook.ipynb` — coleta e engenharia no DuckDB

### 3.1. Arranque assíncrono

- **`nest_asyncio`**: necessário para correr **`ib_insync`** dentro do Jupyter sem conflitos de *event loop*.

### 3.2. Interactive Brokers (`ib_insync`)

- Ligação ao TWS/Gateway (porta habitual do projeto: **4001**).
- Pedido de histórico do **SPY**; construção de `log_return` e escrita em **`prices_daily`** (inserção de linhas novas relativamente ao que já existe na tabela).
- Semelhante para o **VIX** como índice (CBOE), gravado em **`macro_daily`** com `series_id = 'VIX'`.

### 3.3. SQL sobre `prices_daily`

- Criação/alteração da tabela: colunas de **RV** em janelas (ex.: `rv_5d`, `rv_21d`, `rv_63d`) via `UPDATE` com subqueries de somas de quadrados de retornos.
- Features derivadas (ex.: **`vol_of_vol`**, **`vol_term_structure`**, **`vrp`**, **`return_autocorr`**) através de `JOIN` com `macro_daily` (VIX) e lógica SQL no `UPDATE`.
- Pode existir célula para **Amihud** (`|retorno|/volume`, escalado) em `prices_daily` — hoje também há caminho em **`update_db.py`**.

### 3.4. FRED (`fredapi`)

- Séries macro (ex.: **10Y−2Y** como `2S10S`) descarregadas e inseridas em **`macro_daily`** com `INSERT OR IGNORE`.

### 3.5. Papel na cadeia

- O notebook é a **fonte primária** de dados de mercado (IB) e parte do macro; tudo converge para **`quant.db`**, que **`main.py`** e **`predict.py`** assumem já coerente e alinhado por `date`.

---

## 4. `update_db.py` — atualização auxiliar da base

### 4.1. Ligação

- Abre o mesmo **`Data/quant.db`** com caminho relativo ao script.

### 4.2. Amihud

- Garante coluna **`amihud_liquidity`** em `prices_daily`.
- `UPDATE` com subquery: **`|log_return| / NULLIF(volume,0) * 1e6`** por `(date, ticker)`.

### 4.3. Macro FRED

- Usa **`fredapi.Fred`** para séries **`HY_OAS`** (`BAMLH0A0HYM2`) e **`ICSA`** (`ICSA`), desde uma data inicial fixa.
- Converte para `DataFrame` de linhas `(date, series_id, value)` e faz **`INSERT OR IGNORE`** em **`macro_daily`**.

### 4.4. Papel na cadeia

- Complementa o notebook: **não substitui** o IB; **atualiza** features macro e liquidez que **`predict.py`** já tenta ler (`hy_oas`, `icsa`, `amihud_liquidity`).
- **Nota de segurança**: a chave FRED está no código; o ideal é migrar para variável de ambiente (como no notebook) para não expor credenciais.

---

## 5. `main.py` — HMM global e gráfico (análise)

### 5.1. Carga

- `SELECT` de SPY em **`prices_daily`** com `LEFT JOIN` a **`macro_daily`** para **VIX** e **2S10S** (`slope_2s10s`).
- Remove linhas com `NA` nas colunas carregadas.

### 5.2. Features do HMM

- Lista fixa: **`log_return`**, **`rv_21d`**, **`vix`**, **`vrp`**, **`slope_2s10s`**.
- **StandardScaler** em toda a série (fit = série completa).

### 5.3. Seleção de modelo

- Ajusta **GaussianHMM** com **2, 3 e 4 estados**, `covariance_type='full'`, muitas iterações.
- Calcula **BIC** manualmente (contagem de parâmetros + log-verosimilhança) e escolhe o **número de estados** com menor BIC.

### 5.4. Regimes e ordem dos estados

- **Viterbi** (`predict`) para sequência de estados; **`predict_proba`** para probabilidades suavizadas.
- **Reordenação**: estado **0** = menor média de **`rv_21d`** (dentro dos pontos classificados naquele estado), para interpretação “baixa → alta vol”.

### 5.5. Saída

- Estatísticas por `regime`.
- Figura com preço **SPY** e sombras por regime + painel de probabilidades → **`regime_plot.png`**.

### 5.6. Relação com `predict.py`

- **Independente**: o HMM aqui é **um único fit** em toda a história; em **`predict.py`** o HMM é **rolling + filtro forward** e usa **outro conjunto de observações** (só `log_return` e `vix`). Os “regimes” do `main.py` **não** são os rótulos do XGBoost.

---

## 6. `predict.py` — pipeline preditivo completo

### 6.1. Carga e limpeza

- `SELECT` mais largo: preços + **`amihud_liquidity`** + macro agregado por `CASE`: **VIX**, **2S10S**, **HY_OAS**, **ICSA**.
- Ordenação por `date`.
- **`hy_oas`** e **`icsa`**: `ffill` e `bfill` para buracos; **`icsa`** ainda pode ser preenchido com **0** se tudo for nulo (ex.: alinhamento de calendário).
- Comentário no código fala em colunas “core”; na prática o script faz **`dropna()`** sobre o `DataFrame` inteiro após estes passos — ou seja, **qualquer `NA` restante** nas colunas presentes elimina a linha (importante para séries que comecem mais tarde).

### 6.2. Feature categórica `regime` (não é o alvo do XGBoost)

- Percentil **expanding** de **`rv_21d`** com `MIN_WINDOW = 63` dias.
- `pd.cut` em quartis → classes **0–3** (`REGIME_NAMES`: Low vol … Crisis).
- Serve como **`regime_lag1`** (com `shift(1)`) nas features do classificador — **persistência do quartil de vol realizada**.

### 6.3. `garch_var` — GARCH(1,1) rolling

- Para cada índice `t` a partir de **`MIN_GARCH - 1`**: janela de **252** retornos até `t` (escalados ×100), ajusta **GARCH(1,1)**, prevê **variância a 1 passo** e grava em **`garch_var[t]`**.
- Objetivo: **sem *lookahead*** no parâmetro da volatilidade condicional (o fit não usa retornos futuros).

### 6.4. HMM rolling e colunas `p_hmm*_tmrw`

- Observações do HMM: apenas **`log_return`** e **`vix`**, padronizadas.
- A partir de **`MIN_HMM = 756`** dias: a cada **`HMM_REFIT = 21`** dias, **refita** o HMM (4 estados, **`covariance_type='full'`**) em **todos** os dados `0…t`.
- Entre refits: **passo de filtro** — predição com **`transmat_`**, atualização com densidade **Gaussiana multivariada** completa (`_emission_full`, com `eps` na diagonal para estabilidade).
- Ordenação dos estados por **média de VIX** in-sample (estado mais “calmo” em medo = índice menor após reorder).
- Em cada `t`: guarda **P(estado amanhã | dados até t)** → **`p_hmm0_tmrw` … `p_hmm3_tmrw`** (já reordenadas). Estas colunas **não** recebem lag extra no *feature matrix* (já são “amanhã” condicional a informação até `t`).

### 6.5. Alvo `target` — choque de vol em 5 dias

- **`FORWARD_HORIZON = 5`**.
- RV “para trás” 5d e RV “para a frente” 5d (somas de quadrados de `log_return`, anualizadas com fator `252/5`).
- **`vol_shock_ratio = fwd_rv / bwd_rv - 1`**.
- Limiares **`RV_SHOCK_LOW`** / **`RV_SHOCK_HIGH`** (atualmente **±0.075**): abaixo → classe **0 (Queda)**, acima → **2 (Alta)**, meio → **1 (Estável)**. Onde o rácio é inválido, `target` é `NaN` e a linha cai no `dropna` final.

### 6.6. Lista `BASE` e *lags*

- **`BASE`**: `log_return`, `rv_21d`, `vix`, `vrp`, `slope_2s10s`, `garch_var`, `amihud_liquidity`, `hy_oas`, `icsa`.
- Para cada coluna em `BASE`, cria **`_lag1`** e **`_lag5`**.
- Features finais: **todas as colunas base e lags** + **`regime_lag1`** + **`HMM_COLS`**.

### 6.7. XGBoost e validação

- **`XGBClassifier`** com hiperparâmetros fixos (`mlogloss`, profundidade limitada, *subsample*, etc.).
- **`TimeSeriesSplit(n_splits=5)`**: validação **walk-forward** no tempo.
- **`compute_sample_weight(..., class_weight='balanced')`** no treino (por *fold* e no modelo final) para mitigar desequilíbrio entre Queda / Estável / Alta.
- Relatório de classificação e matriz de confusão sobre predições concatenadas dos testes.
- Modelo **final** treinado em **todo** o `X,y` com os mesmos pesos.

### 6.8. Predição no último dia

- Usa a **última linha** de `df` após `dropna` para `predict_proba`.
- Imprime: data, **regime de ontem** (quartil `rv_21d` via `regime_lag1`), classe prevista do **choque de vol** e probabilidades das três classes.

### 6.9. Relação com `main.py` e com o notebook

- Usa **as mesmas tabelas** que o notebook alimenta; **não** consome o PNG nem os regimes do `main.py`.
- Depende de colunas opcionais (**`hy_oas`**, **`icsa`**, **`amihud_liquidity`**) estarem preenchidas ou tratadas por `ffill`/`bfill`; caso contrário, `dropna` reduz a amostra.

---

## 7. Ficheiros de documentação / notas

| Ficheiro    | Função |
|------------|--------|
| **`CLAUDE.md`** | Guia rápido para quem desenvolve: comandos, arquitetura, dependências (pode estar ligeiramente desatualizado face ao `predict.py` atual — convém alinhar após mudanças). |
| **`notes.md`** / **`results.md`** | Notas ou resultados em texto; não fazem parte do pipeline executável. |

---

## 8. Ordem sugerida de trabalho no dia a dia

1. Garantir **TWS/Gateway** (se for usar IB) e **FRED** com chave válida.
2. Correr células relevantes do **`notebook.ipynb`** para atualizar **SPY**, **VIX**, curva, RV, VRP, etc.
3. Opcional: **`python update_db.py`** para **HY_OAS**, **ICSA** e recalcular **Amihud** de forma consistente.
4. Opcional: **`python main.py`** para inspeção visual dos regimes HMM “globais”.
5. **`python predict.py`** para métricas out-of-sample (no esquema walk-forward) e predição do próximo choque de vol.

---

## 9. Resumo das ligações (uma frase por aresta)

| De | Para | Ligação |
|----|------|---------|
| Notebook / `update_db.py` | `quant.db` | Escrevem `prices_daily` e `macro_daily`. |
| `quant.db` | `main.py` | Leitura só de leitura; saída é imagem. |
| `quant.db` | `predict.py` | Leitura só de leitura; saída é texto no terminal. |
| `main.py` | `predict.py` | Nenhuma dependência direta; modelos e alvos diferentes. |
| `update_db.py` | `predict.py` | Atualiza colunas/series que `predict.py` espera no `SELECT`. |

---

*Documento gerado para acompanhar a lógica do repositório; se alterar constantes (`MIN_HMM`, limiares de choque, lista `BASE`), atualize este ficheiro em conjunto.*
