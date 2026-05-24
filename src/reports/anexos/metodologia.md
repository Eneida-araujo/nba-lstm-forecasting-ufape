# Metodologia

## 1. Objetivo do Projeto

O projeto teve como objetivo desenvolver um modelo baseado em redes neurais recorrentes do tipo LSTM (Long Short-Term Memory) para realizar forecasting de estatísticas de desempenho de equipes da NBA durante a temporada regular 2025/2026.

As previsões foram realizadas para as equipes Boston Celtics e Denver Nuggets, utilizando variáveis estatísticas relacionadas ao desempenho esportivo das equipes, incluindo:

- pontos (PTS);
- rebotes (REB);
- assistências (AST).

Além da modelagem preditiva utilizando LSTM, o projeto também teve como objetivo automatizar o processo de seleção de variáveis explicativas (features), utilizando obrigatoriamente:

- análise estatística baseada em p-value;
- análise de importância utilizando Random Forest.

Também foram avaliadas diferentes janelas temporais (Sliding Windows) de:

- 5 jogos;
- 10 jogos;
- 15 jogos;
- 20 jogos.

O objetivo principal da comparação entre as janelas temporais foi identificar quais configurações temporais apresentavam maior capacidade preditiva para cada variável alvo.

---

## 2. Bases de Dados

O projeto utilizou duas bases de dados disponibilizadas para a atividade:

- BaseA.csv
- BaseB.csv

Os arquivos encontram-se em:

- `data/raw/BaseA.csv`
- `data/raw/BaseB.csv`

O carregamento das bases foi realizado através do módulo:

- `src/data/loader.py`

Posteriormente, as bases foram integradas utilizando:

- `src/data/merger.py`

A BaseA foi utilizada como base principal do projeto, enquanto a BaseB foi utilizada como base complementar para adicionar atributos estatísticos adicionais.

O processo de integração utilizou as colunas:

- `GAME_ID`
- `TEAM_NAME`

Foi utilizada estratégia LEFT JOIN, preservando integralmente os registros da base principal.

---

## 3. Estrutura Arquitetural do Projeto

A arquitetura do projeto foi desenvolvida de forma modular, permitindo separação clara entre:

- carregamento dos dados;
- pré-processamento;
- engenharia de features;
- seleção estatística;
- benchmark experimental;
- treinamento da LSTM;
- avaliação experimental;
- geração de gráficos e outputs.

A execução principal da pipeline experimental foi centralizada em:

- `main.py`

A estrutura modular facilitou:

- manutenção do código;
- rastreabilidade experimental;
- organização científica;
- reprodutibilidade;
- expansão futura do projeto.

---

## 4. Engenharia de Dados

A etapa de engenharia de dados foi implementada nos arquivos:

- `src/data/preprocessing.py`
- `src/data/validation.py`

Nessa etapa foram realizados:

- padronização dos nomes das colunas;
- conversão da variável `GAME_DATE` para formato datetime;
- ordenação cronológica das partidas;
- remoção de registros duplicados;
- tratamento de valores ausentes;
- reorganização dos índices.

A ordenação temporal foi realizada utilizando:

- `TEAM_NAME`
- `GAME_DATE`

Esse processo foi fundamental para garantir consistência temporal nos experimentos de séries temporais utilizando LSTM.

Os valores ausentes numéricos foram tratados utilizando substituição pela mediana da própria variável.

Também foi realizada uma etapa inicial de validação estrutural dos dados utilizando:

- `src/data/validation.py`

Essa etapa verificou:

- dimensões do dataset;
- tipos de dados;
- valores ausentes;
- integridade estrutural das colunas.

---

## 5. Engenharia de Features

A engenharia de features foi implementada em:

- `src/features/engineering.py`

Nessa etapa foram criadas variáveis derivadas com objetivo de representar o comportamento histórico recente das equipes.

Foram construídas médias móveis temporais para diferentes janelas:

- 5 jogos;
- 10 jogos;
- 15 jogos;
- 20 jogos.

As médias móveis foram construídas utilizando `shift(1)`, garantindo que apenas informações passadas fossem utilizadas durante as previsões.

Essa estratégia evitou vazamento temporal de informações futuras para o passado.

As médias móveis foram utilizadas como variáveis explicativas auxiliares e não como substituição direta dos alvos principais.

---

## 6. Filtro de Correlação

Após a engenharia de features, foi aplicado filtro de correlação utilizando:

- `src/features/correlation_filter.py`

O objetivo dessa etapa foi remover variáveis altamente correlacionadas entre si, reduzindo redundância estatística e multicolinearidade.

O processo utilizou:

- matriz de correlação absoluta;
- análise do triângulo superior da matriz;
- threshold de correlação de 0.90.

Quando duas variáveis apresentavam correlação superior ao limite estabelecido, uma delas era removida.

As variáveis removidas foram registradas em:

- `outputs/tables/correlation_removed_features.csv`

---

## 7. Seleção por p-value

A seleção estatística baseada em significância foi implementada em:

- `src/features/pvalue_selection.py`

Foi utilizado modelo OLS (Ordinary Least Squares) da biblioteca StatsModels para calcular os p-values das variáveis explicativas.

O objetivo foi identificar quais features apresentavam relação estatisticamente significativa com os alvos preditivos.

Foram gerados rankings estatísticos para:

- PTS;
- REB;
- AST.

As features selecionadas foram utilizadas nas etapas posteriores de benchmark experimental.

---

## 8. Seleção por Random Forest

A seleção baseada em importância de variáveis foi implementada em:

- `src/features/random_forest_importance.py`

O algoritmo Random Forest foi utilizado para:

- ranquear variáveis;
- calcular importância relativa;
- identificar atributos mais relevantes;
- construir rankings preditivos.

Os rankings gerados foram salvos em:

- `outputs/tables/rf_feature_importance_pts.csv`
- `outputs/tables/rf_feature_importance_reb.csv`
- `outputs/tables/rf_feature_importance_ast.csv`

---

## 9. Construção dos Feature Sets

A construção dos conjuntos de features foi implementada em:

- `src/features/feature_sets.py`
- `src/features/feature_sets_summary.py`

Foram construídos diferentes conjuntos de variáveis para comparação experimental:

| Feature Set | Descrição |
|---|---|
| P-value | Features estatisticamente significativas |
| RF Top 10 | 10 features mais importantes |
| RF Top 20 | 20 features mais importantes |
| Intersection | Interseção entre p-value e Random Forest |
| Hybrid | Combinação estratégica |
| Full Features | Conjunto completo |

Essa estratégia permitiu comparar diferentes abordagens de seleção de atributos.

---

## 10. Benchmark Experimental dos Feature Sets

O benchmark experimental foi implementado em:

- `src/features/feature_experiments.py`

Foram realizados experimentos comparativos entre os diferentes conjuntos de features utilizando Random Forest.

Os experimentos avaliaram:

- RMSE;
- MAE;
- R²;
- estabilidade temporal;
- desempenho comparativo.

Os resultados foram utilizados para selecionar os melhores conjuntos para utilização na LSTM.

---

## 11. Sliding Windows

A geração das janelas temporais foi implementada em:

- `src/models/windowing.py`

As Sliding Windows foram utilizadas para transformar os dados tabulares em sequências temporais tridimensionais compatíveis com redes neurais LSTM.

Foram avaliadas janelas de:

- 5 jogos;
- 10 jogos;
- 15 jogos;
- 20 jogos.

As sequências foram organizadas no formato:

```python
(samples, timesteps, features)

O objetivo foi permitir que a LSTM aprendesse padrões temporais sequenciais das equipes ao longo da temporada.
```
---

## 12. Arquitetura LSTM

A arquitetura principal baseada em Deep Learning foi implementada em:

- `src/models/lstm_model.py`

A arquitetura foi composta por:

- duas camadas LSTM;
- camadas Dense;
- Dropout;
- recurrent_dropout;
- regularização L2;
- Adam Optimizer;
- gradient clipping.

A primeira camada LSTM foi construída com:

- 64 neurônios;
- `return_sequences=True`

A segunda camada LSTM foi construída com:

- 32 neurônios;
- `return_sequences=False`

Após as camadas LSTM, foi utilizada:

- uma camada Dense com 16 neurônios e ativação ReLU;
- uma camada de saída com 1 neurônio e ativação linear.

A função de perda utilizada foi:

- Mean Squared Error (MSE)

O otimizador utilizado foi:

- Adam

Com:

- `clipnorm=1.0`

Foram utilizadas técnicas de regularização para reduzir overfitting:

- `Dropout(0.20)`;
- `recurrent_dropout=0.10`;
- `kernel_regularizer=l2(0.001)`.

---

## 13. Treinamento dos Modelos

O treinamento dos modelos foi implementado em:

- `src/models/lstm_trainer.py`

Durante o treinamento foram utilizadas técnicas como:

- EarlyStopping;
- ReduceLROnPlateau;
- validação temporal;
- monitoramento da perda;
- salvamento dos melhores modelos.

Os dados foram divididos temporalmente em:

- treino;
- validação;
- teste.

Foi utilizada normalização utilizando:

- MinMaxScaler

Os modelos foram treinados separadamente para:

- PTS;
- REB;
- AST.

Também foram realizados experimentos comparando as diferentes Sliding Windows.

---

## 14. Predição Final

O processo de inferência foi implementado em:

- `src/models/predictor.py`

Nessa etapa o sistema realiza:

- carregamento dos modelos treinados;
- carregamento dos scalers;
- execução das previsões;
- cálculo de previsões finais;
- geração de interpretações quantitativas.

As previsões finais foram armazenadas em:

- `outputs/predictions/`

---

## 15. Avaliação Experimental

A avaliação quantitativa dos modelos foi implementada em:

- `src/models/evaluation.py`

Foram utilizadas métricas de regressão e classificação, incluindo:

- RMSE;
- MAE;
- MAPE;
- R²;
- erro médio;
- over prediction;
- under prediction;
- accuracy;
- F1-score;
- AUC;
- matriz de confusão.

Também foram calculados:

- intervalos de confiança;
- probabilidades interpretativas;
- análises residuais.

Essa etapa permitiu avaliar:

- precisão;
- estabilidade;
- robustez;
- capacidade de generalização.

---

## 16. Visualizações

A geração das visualizações foi implementada em:

- `src/visualization/plots.py`

Os gráficos gerados incluíram:

- Real vs Predito;
- histogramas de erro;
- matrizes de confusão;
- barras de confiança.

As figuras foram armazenadas em:

- `outputs/figures/`

---

## 17. RF7 — Relatório Técnico Executivo

O RF7 teve como objetivo consolidar os principais resultados experimentais do projeto em formato interpretativo e executivo.

Nessa etapa foram organizados:

- rankings de desempenho;
- interpretações quantitativas;
- comparações entre janelas;
- análise de desempenho dos targets;
- resultados dos modelos LSTM.

O objetivo principal foi facilitar interpretação técnica e análise final dos experimentos realizados.

---

## 18. Reprodutibilidade

O projeto foi desenvolvido visando total reprodutibilidade experimental.

Para isso foram utilizados:

- ambiente virtual Python (`venv`);
- modularização do código;
- versionamento Git;
- separação estruturada dos datasets;
- armazenamento organizado de outputs;
- controle de dependências.

Toda a execução experimental foi centralizada em:

- `main.py`

A arquitetura foi estruturada para permitir:

- replicação dos experimentos;
- rastreabilidade científica;
- auditoria metodológica;
- expansão futura da pesquisa.

---

## 19. Mapeamento entre Etapas e Arquivos

| Etapa | Arquivo |
|---|---|
| Execução principal | `main.py` |
| Configurações globais | `src/config/settings.py` |
| Carregamento dos dados | `src/data/loader.py` |
| Integração das bases | `src/data/merger.py` |
| Pré-processamento | `src/data/preprocessing.py` |
| Validação estrutural | `src/data/validation.py` |
| Engenharia de features | `src/features/engineering.py` |
| Filtro de correlação | `src/features/correlation_filter.py` |
| Seleção por p-value | `src/features/pvalue_selection.py` |
| Random Forest Importance | `src/features/random_forest_importance.py` |
| Construção dos feature sets | `src/features/feature_sets.py` |
| Resumo dos feature sets | `src/features/feature_sets_summary.py` |
| Benchmark experimental | `src/features/feature_experiments.py` |
| Sliding Windows | `src/models/windowing.py` |
| Arquitetura LSTM | `src/models/lstm_model.py` |
| Treinamento da LSTM | `src/models/lstm_trainer.py` |
| Predição final | `src/models/predictor.py` |
| Avaliação experimental | `src/models/evaluation.py` |
| Visualizações | `src/visualization/plots.py` |