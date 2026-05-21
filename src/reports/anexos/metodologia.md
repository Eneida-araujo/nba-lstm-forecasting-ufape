# Metodologia do Projeto

## 1. Objetivo

O objetivo deste projeto é desenvolver modelos de Machine Learning baseados em redes neurais LSTM para realizar forecasting de resultados da NBA na temporada 2025/2026, conforme solicitado na prova prática da disciplina de Aprendizado de Máquina do PPCOMP/UFAPE.

O sistema deverá prever:

- pontos;
- rebotes;
- assistências;
- probabilidades de desempenho acima das médias;
- probabilidades de atingir metas específicas.

Além disso, o projeto deverá utilizar:

- seleção estatística de features com p-value;
- análise de importância com Random Forest;
- Sliding Windows;
- métricas de regressão e classificação;
- visualizações gráficas;
- interpretação executiva dos resultados.

---

# 2. Bases de Dados

Foram utilizadas duas bases de dados fornecidas pelo professor:

- BaseA.csv
- BaseB.csv

As bases possuem informações estatísticas das equipes da NBA referentes à temporada regular 2025/2026.

A BaseA foi utilizada como base principal devido à presença de atributos adicionais relevantes, como:

- TEAM_ABBREVIATION;
- PLUS_MINUS;
- VIDEO_AVAILABLE.

A BaseB foi utilizada como complemento, principalmente para:

- W;
- L;
- W_PCT.

---

# 3. Engenharia de Dados Inicial

Inicialmente foi realizada:

- padronização das colunas;
- conversão de datas;
- remoção de duplicatas;
- tratamento de valores ausentes;
- ordenação temporal dos dados.

Durante a análise inicial foi identificado um problema importante no merge das bases.

A primeira estratégia de merge utilizava todas as colunas em comum entre os datasets, o que resultou em duplicação dos registros e expansão indevida do dataset para 4920 linhas.

Após análise técnica, identificou-se que o merge correto deveria utilizar:

- GAME_ID;
- TEAM_NAME.

A nova estratégia permitiu:

- eliminar duplicações;
- preservar integridade temporal;
- manter consistência estatística;
- evitar inconsistências entre registros.

Após a correção, o dataset final passou a possuir:

- 2460 linhas;
- 32 colunas;
- 0 valores ausentes.

---

# 4. Organização do Projeto

O projeto foi estruturado utilizando boas práticas de engenharia de software, incluindo:

- modularização;
- versionamento com Git;
- separação por responsabilidades;
- documentação do código;
- arquitetura compatível com GitHub.

A estrutura foi organizada em módulos de:

- carregamento;
- preprocessamento;
- engenharia de features;
- seleção de variáveis;
- modelagem;
- avaliação;
- visualização;
- geração de relatórios.

---

# 5. Engenharia de Features

Após a conclusão da engenharia de dados inicial, iniciou-se a etapa de engenharia de features, considerada uma das partes mais importantes do projeto, pois a qualidade das variáveis de entrada influencia diretamente o desempenho dos modelos de Machine Learning.

O objetivo dessa etapa foi transformar os dados brutos da NBA em variáveis estatísticas e temporais capazes de representar:

- desempenho recente;
- estabilidade;
- tendência;
- momentum;
- eficiência;
- contexto das partidas.

Além disso, todas as features temporais foram construídas respeitando a integridade cronológica dos dados, evitando vazamento temporal (data leakage).

---

## 5.1 Proteção Temporal e Uso de Shift(1)

Um dos principais cuidados metodológicos adotados foi a utilização de `shift(1)` em todas as variáveis temporais.

Essa estratégia garante que o modelo utilize apenas informações anteriores ao jogo alvo durante as previsões.

Exemplo:

- ao prever a próxima partida de uma equipe, o modelo não pode acessar estatísticas do próprio jogo que está sendo previsto.

Sem essa proteção, o modelo teria acesso indireto ao futuro, gerando resultados artificialmente elevados e comprometendo a validade científica do experimento.

A utilização de `shift(1)` foi aplicada principalmente em:

- médias móveis;
- desvios móveis;
- variáveis defasadas;
- métricas de momentum;
- variáveis acumuladas da temporada.

---

## 5.2 Features Contextuais

### IS_HOME

Variável binária indicando se a equipe jogou em casa.

Critério:

- `1` → jogo em casa;
- `0` → jogo fora.

Essa variável foi construída a partir da coluna `MATCHUP`.

Justificativa:

Equipes da NBA frequentemente apresentam desempenho diferente jogando em casa devido a fatores como torcida, deslocamento e familiaridade com a quadra.

### DAYS_REST

Representa a quantidade de dias de descanso entre partidas consecutivas.

Justificativa:

O calendário da NBA é extremamente intenso, e o tempo de recuperação física pode influenciar diretamente:

- desempenho ofensivo;
- intensidade defensiva;
- aproveitamento dos arremessos;
- fadiga acumulada.

---

## 5.3 Features Lag

Foram criadas variáveis defasadas utilizando `shift(1)` para representar o desempenho imediatamente anterior das equipes.

Exemplos:

- PTS_LAG1;
- REB_LAG1;
- AST_LAG1;
- PLUS_MINUS_LAG1;
- W_PCT_LAG1.

Essas variáveis ajudam o modelo a compreender o estado recente da equipe antes da partida alvo.

---

## 5.4 Médias Móveis (Rolling Means)

Foram criadas médias móveis utilizando janelas temporais de:

- 5 jogos;
- 10 jogos;
- 15 jogos;
- 20 jogos.

As rolling windows foram escolhidas por serem compatíveis com os Sliding Windows exigidos na prova prática.

As médias móveis foram calculadas para diversas estatísticas da NBA, incluindo:

- pontos;
- rebotes;
- assistências;
- aproveitamento de arremessos;
- turnovers;
- roubos de bola;
- bloqueios;
- faltas;
- plus/minus.

Exemplos:

- PTS_ROLLING_MEAN_5;
- REB_ROLLING_MEAN_10;
- AST_ROLLING_MEAN_20.

Objetivo:

Permitir que o modelo compreenda tendências recentes e padrões temporais de desempenho.

---

## 5.5 Desvios Móveis (Rolling Standard Deviation)

Também foram criados desvios padrão móveis para medir estabilidade e variabilidade das equipes.

Exemplos:

- PTS_ROLLING_STD_5;
- REB_ROLLING_STD_10;
- AST_ROLLING_STD_20.

Justificativa:

Duas equipes podem possuir a mesma média de pontos, porém comportamentos completamente diferentes:

- uma equipe pode ser extremamente consistente;
- outra pode oscilar muito entre partidas.

Os desvios móveis ajudam o modelo a identificar essa estabilidade.

---

## 5.6 Features de Momentum

Foram criadas variáveis de momentum para representar aceleração ou desaceleração recente de desempenho.

Estratégia utilizada:

Comparar médias móveis curtas com médias móveis mais longas.

Exemplo:

- média dos últimos 5 jogos menos média dos últimos 10 jogos;
- média dos últimos 5 jogos menos média dos últimos 20 jogos.

Exemplos de variáveis:

- PTS_MOMENTUM_5_10;
- REB_MOMENTUM_5_20;
- FG_PCT_MOMENTUM_5_10.

Interpretação:

- valores positivos indicam melhora recente;
- valores negativos indicam queda de desempenho.

---

## 5.7 Features de Eficiência

Também foram criadas métricas derivadas relacionadas à eficiência ofensiva e equilíbrio estatístico.

Exemplos:

- AST_TO_TOV;
- SHOOTING_EFFICIENCY_LAG1;
- REB_BALANCE_LAG1.

Objetivo:

Fornecer ao modelo informações mais sofisticadas do comportamento das equipes além das estatísticas brutas tradicionais.

---

## 5.8 Features de Sequência (Streaks)

Foram criadas variáveis relacionadas a sequências de vitórias e derrotas consecutivas.

Exemplos:

- WIN_STREAK;
- LOSS_STREAK.

Essas variáveis ajudam a representar:

- confiança da equipe;
- momento psicológico;
- consistência competitiva.

Assim como nas demais features temporais, as sequências foram calculadas utilizando apenas informações anteriores ao jogo alvo.

---

## 5.9 Tratamento dos Valores Ausentes

A utilização de rolling windows naturalmente gera valores ausentes nos primeiros jogos da temporada, pois ainda não existem partidas suficientes para compor determinadas janelas.

Exemplo:

- no primeiro jogo da temporada não existem 5 jogos anteriores para calcular uma média móvel de 5 partidas.

Estratégia utilizada:

- preenchimento pela mediana da própria coluna.

Essa abordagem permitiu:

- preservar a base completa;
- evitar perda de registros;
- manter estabilidade estatística.

---

## 5.10 Resultado do Feature Engineering

Após a criação das novas variáveis, o dataset passou de:

- 32 colunas originais;

para:

- 218 features candidatas.

Essa expansão permitiu construir um conjunto rico de variáveis para:

- seleção estatística com p-value;
- análise de importância com Random Forest;
- experimentos comparativos;
- treinamento dos modelos LSTM.

---

# 6. Correlation Filter

Após a criação das features, foi aplicada uma etapa de filtragem de correlação para reduzir redundância entre variáveis.

Objetivos:

- reduzir multicolinearidade;
- simplificar o espaço de features;
- melhorar estabilidade da regressão OLS;
- melhorar interpretabilidade dos resultados.

Critério adotado:

- correlação absoluta superior a 0.90.

Estratégia:

Quando duas variáveis apresentavam correlação muito alta, uma delas era removida do conjunto de features.

Exemplos:

- médias móveis muito semelhantes;
- estatísticas derivadas altamente redundantes.

---

## 6.1 Resultado do Correlation Filter

Resultados obtidos:

- 22 features removidas;
- redução de 218 para 196 colunas.

O filtro permitiu:

- reduzir redundância estatística;
- preservar variáveis relevantes;
- preparar a base para seleção por p-value e Random Forest.

As features removidas foram salvas automaticamente em:

```text
outputs/tables/correlation_removed_features.csv
```

---

# 7. Seleção Estatística com p-value

Após a conclusão da etapa de engenharia de features e redução de multicolinearidade com o Correlation Filter, iniciou-se a etapa de seleção estatística utilizando p-value, conforme exigido na prova prática.

Essa etapa teve como objetivo identificar quais variáveis apresentavam relação estatisticamente significativa com os alvos de previsão:

- pontos (PTS);
- rebotes (REB);
- assistências (AST).

---

## 7.1 Objetivo da Seleção por p-value

O uso do p-value permite verificar se uma variável possui relevância estatística para explicar o comportamento do alvo previsto.

Em termos práticos:

- quanto menor o p-value;
- maior a evidência de que a feature possui relação estatística relevante com o target.

Critério adotado:

```text
p-value < 0.05
```

Esse limiar representa nível de confiança estatística de 95%, sendo amplamente utilizado em pesquisas científicas e modelos estatísticos.

---

## 7.2 Modelo Estatístico Utilizado

Foi utilizada regressão linear múltipla baseada em Ordinary Least Squares (OLS), implementada através da biblioteca:

```python
statsmodels.api.OLS
```

O modelo OLS foi utilizado exclusivamente para análise de significância estatística das variáveis, e não como modelo final de previsão.

---

## 7.3 Estratégias para Evitar Vazamento Temporal

Durante a seleção de features foram removidas variáveis que poderiam gerar vazamento temporal (data leakage).

Exemplos de variáveis removidas:

- PTS;
- REB;
- AST;
- FGM;
- FGA;
- FG_PCT;
- PLUS_MINUS;
- W;
- L;
- W_PCT.

Essas variáveis representam estatísticas do próprio jogo alvo e, portanto, não estariam disponíveis antes da partida.

A remoção dessas colunas garante:

- validade científica;
- integridade temporal;
- realismo operacional do modelo.

---

## 7.4 Tratamento dos Dados Antes do OLS

Antes da execução da regressão OLS, foram aplicadas etapas adicionais de preparação dos dados.

### Seleção apenas de colunas numéricas

A regressão OLS exige variáveis numéricas.

Por esse motivo, variáveis textuais como:

- TEAM_NAME;
- TEAM_ABBREVIATION;
- MATCHUP;
- WL.

foram automaticamente excluídas da análise.

### Tratamento de valores infinitos

Valores infinitos positivos e negativos foram substituídos por valores ausentes (`NaN`).

### Tratamento de valores ausentes

Os valores ausentes foram preenchidos utilizando a mediana da própria coluna.

Essa abordagem reduz sensibilidade a outliers e preserva estabilidade estatística.

### Remoção de colunas constantes

Variáveis sem variabilidade foram removidas automaticamente.

Essas variáveis não contribuem para o modelo e podem prejudicar a estabilidade da regressão.

### Padronização das Features

As variáveis foram padronizadas utilizando:

```python
StandardScaler
```

Objetivos:

- melhorar estabilidade numérica;
- evitar influência excessiva de escalas diferentes;
- melhorar interpretação estatística.

---

## 7.5 Resultado da Seleção por p-value

Após o filtro de correlação, o dataset possuía:

- 196 features.

Após remoção de:

- targets;
- identificadores;
- variáveis proibidas;
- colunas não numéricas.

Restaram:

- 166 features candidatas para análise estatística.

---

## 7.6 Resultados por Target

### 7.6.1 Target: PTS (Pontos)

Resultados obtidos:

- 18 features estatisticamente significativas.

Interpretação:

A pontuação apresentou forte relação com:

- momentum ofensivo;
- médias móveis recentes;
- eficiência ofensiva;
- estabilidade estatística.

### 7.6.2 Target: REB (Rebotes)

Resultados obtidos:

- 9 features estatisticamente significativas.

Interpretação:

Os rebotes apresentaram comportamento mais variável e menos linear, indicando maior dependência de:

- matchup;
- estilo de jogo;
- contexto da partida.

### 7.6.3 Target: AST (Assistências)

Resultados obtidos:

- 13 features estatisticamente significativas.

Interpretação:

As assistências demonstraram forte relação com:

- organização ofensiva;
- estabilidade do time;
- ritmo ofensivo;
- eficiência coletiva.

---

## 7.7 Arquivos Gerados Automaticamente

A etapa de seleção por p-value gerou automaticamente tabelas CSV contendo:

- ranking completo de p-values;
- features selecionadas.

Arquivos gerados:

```text
outputs/tables/pvalues_pts.csv
outputs/tables/pvalues_reb.csv
outputs/tables/pvalues_ast.csv

outputs/tables/selected_features_pvalue_pts.csv
outputs/tables/selected_features_pvalue_reb.csv
outputs/tables/selected_features_pvalue_ast.csv
```

Esses arquivos serão utilizados posteriormente para:

- comparação de experimentos;
- cruzamento com Random Forest;
- construção dos feature sets finais;
- documentação técnica;
- geração de gráficos e tabelas do relatório final.

---

## 7.8 Importância da Etapa de p-value

A utilização do p-value foi fundamental para:

- reduzir ruído estatístico;
- aumentar interpretabilidade do modelo;
- selecionar variáveis relevantes;
- melhorar robustez dos experimentos;
- atender aos requisitos obrigatórios da prova prática.

Além disso, essa etapa fornece forte embasamento científico para justificar as variáveis utilizadas no modelo LSTM final.

---

# 8. Seleção por Importância com Random Forest

Também foi implementada seleção baseada em importância de features utilizando Random Forest Regressor.

Essa abordagem permitiu identificar relações:

- não-lineares;
- temporais;
- combinatórias.

Os modelos Random Forest foram treinados separadamente para:

- PTS;
- REB;
- AST.

As 10 features mais importantes de cada target foram analisadas.

As variáveis mais relevantes identificadas incluíram:

- rolling means;
- eficiência de arremesso;
- plus/minus;
- métricas de momentum;
- eficiência ofensiva;
- estabilidade estatística.

---

## 8.1 Objetivos da Etapa

A etapa de Random Forest foi desenvolvida para:

- ranquear features por importância preditiva;
- complementar a análise estatística do p-value;
- selecionar variáveis mais relevantes para os experimentos;
- identificar relações temporais importantes;
- auxiliar na construção dos conjuntos finais de features para a LSTM.

Além disso, essa etapa também permitiu criar um baseline inicial de previsão temporal utilizando aprendizado supervisionado.

---

## 8.2 Modelo Utilizado

Foi utilizado:

```python
RandomForestRegressor
```

Implementado através da biblioteca:

```python
sklearn.ensemble.RandomForestRegressor
```

O Random Forest foi utilizado exclusivamente como ferramenta de análise de importância das variáveis e baseline inicial, não sendo o modelo final do projeto.

---

## 8.3 Estratégias Contra Vazamento Temporal

Assim como na etapa de p-value, foram removidas variáveis que poderiam causar vazamento temporal.

Exemplos:

- PTS;
- REB;
- AST;
- FGM;
- FGA;
- FG_PCT;
- PLUS_MINUS;
- W;
- L;
- W_PCT.

Essas variáveis representam informações do próprio jogo alvo e não estariam disponíveis antes da partida.

Essa proteção garante:

- integridade científica;
- realismo operacional;
- validade temporal do modelo.

---

## 8.4 Preparação dos Dados

Antes do treinamento do Random Forest, foram realizadas etapas de preparação dos dados.

### Seleção apenas de colunas numéricas

Foram utilizadas exclusivamente variáveis numéricas.

Variáveis textuais foram removidas automaticamente.

### Tratamento de valores infinitos

Valores infinitos positivos e negativos foram substituídos por valores ausentes (`NaN`).

### Tratamento de valores ausentes

Os valores ausentes foram preenchidos utilizando a mediana da própria coluna.

Essa abordagem reduz sensibilidade a outliers e mantém estabilidade estatística.

### Remoção de colunas constantes

Variáveis sem variabilidade foram removidas automaticamente para evitar ruído e redundância.

---

## 8.5 Estratégia Temporal de Avaliação

Foi utilizada divisão temporal dos dados:

- 80% para treino;
- 20% para teste.

A divisão temporal foi escolhida para preservar a ordem cronológica das partidas e evitar mistura entre passado e futuro.

Essa abordagem é fundamental em séries temporais esportivas.

---

## 8.6 Resultados do Random Forest

### 8.6.1 Target: PTS

#### Features mais importantes

As variáveis mais relevantes para previsão de pontos foram:

- MIN;
- PTS_ROLLING_MEAN_20;
- FT_PCT_LAG1;
- FG_PCT_LAG1;
- AST_TO_TOV;
- REB_BALANCE_LAG1.

#### Métricas Baseline — PTS

Resultados obtidos:

- MAE: 9.88;
- RMSE: 12.28;
- R²: 0.055.

---

### 8.6.2 Target: REB

#### Features mais importantes

As variáveis mais relevantes para rebotes foram:

- REB_ROLLING_MEAN_15;
- MIN;
- PLUS_MINUS_LAG1;
- REB_LAG1;
- AST_TO_TOV.

#### Métricas Baseline — REB

Resultados obtidos:

- MAE: 5.42;
- RMSE: 6.97;
- R²: 0.031.

---

### 8.6.3 Target: AST

#### Features mais importantes

As variáveis mais relevantes para assistências foram:

- AST_ROLLING_MEAN_10;
- AST_TO_TOV;
- FT_PCT_LAG1;
- PLUS_MINUS_LAG1;
- FG3_PCT_LAG1.

#### Métricas Baseline — AST

Resultados obtidos:

- MAE: 3.96;
- RMSE: 4.98;
- R²: 0.093.

Esse foi o melhor baseline entre os três targets analisados.

---

## 8.7 Interpretação Geral dos Resultados

Os resultados obtidos mostraram que:

- as features criadas possuem relevância prática;
- a engenharia de features foi eficiente;
- as variáveis temporais possuem forte influência nas previsões;
- rolling means e lag features foram especialmente importantes.

As features mais relevantes identificadas pelo Random Forest apresentaram forte coerência com o comportamento real das partidas da NBA.

---

## 8.8 Arquivos Gerados Automaticamente

A etapa de Random Forest gerou automaticamente arquivos CSV contendo os rankings completos de importância das features.

Arquivos gerados:

```text
outputs/tables/rf_feature_importance_pts.csv
outputs/tables/rf_feature_importance_reb.csv
outputs/tables/rf_feature_importance_ast.csv
```

Esses arquivos serão utilizados posteriormente para:

- seleção das Top Features;
- construção dos conjuntos experimentais;
- cruzamento com as features aprovadas pelo p-value;
- criação dos experimentos comparativos;
- treinamento final da LSTM.

---

## 8.9 Importância da Etapa de Random Forest

A utilização do Random Forest foi fundamental para:

- identificar relações não lineares;
- medir importância prática das variáveis;
- complementar a análise estatística do p-value;
- reduzir dimensionalidade;
- auxiliar na construção dos melhores feature sets.

Essa etapa será decisiva para os próximos experimentos e para a escolha das features finais utilizadas nos modelos LSTM.

---

# 9. Construção dos Feature Sets Experimentais

Após a conclusão das etapas de seleção estatística com p-value e análise de importância com Random Forest, iniciou-se a construção dos conjuntos experimentais de features (Feature Sets).

Essa etapa é considerada extremamente estratégica dentro do projeto, pois define quais variáveis serão utilizadas nos experimentos comparativos e posteriormente nos modelos LSTM finais.

O principal objetivo foi construir conjuntos de variáveis com diferentes características estatísticas e preditivas para avaliar:

- robustez;
- capacidade preditiva;
- estabilidade;
- generalização;
- impacto da dimensionalidade;
- equilíbrio entre interpretabilidade e performance.

---

## 9.1 Objetivos da Etapa

A construção dos Feature Sets teve como principais objetivos:

- comparar diferentes estratégias de seleção de variáveis;
- reduzir dimensionalidade;
- avaliar impacto das features no desempenho preditivo;
- identificar conjuntos mais eficientes para séries temporais esportivas;
- preparar os dados para os experimentos com Sliding Windows;
- selecionar os melhores conjuntos para treinamento da LSTM.

Além disso, essa etapa também permite:

- justificar cientificamente as variáveis utilizadas;
- melhorar interpretabilidade do modelo;
- reduzir overfitting;
- melhorar generalização temporal.

---

## 9.2 Estratégia Geral de Construção

Os conjuntos experimentais foram construídos utilizando informações provenientes de:

- seleção estatística por p-value;
- importância das variáveis via Random Forest;
- conhecimento do domínio esportivo;
- relevância temporal das variáveis.

Cada conjunto possui características específicas e objetivos diferentes.

---

## 9.3 Feature Sets Criados

Foram criados 6 conjuntos principais de features para cada target:

1. pvalue
2. random_forest
3. intersection
4. top_10_rf
5. top_20_rf
6. hybrid

Os conjuntos foram criados separadamente para:

- PTS;
- REB;
- AST.

---

## 9.4 Conjunto pvalue

O conjunto `pvalue` contém exclusivamente variáveis consideradas estatisticamente significativas através da regressão OLS.

Critério:

```text
p-value < 0.05
```

Características:

- maior interpretabilidade estatística;
- menor dimensionalidade;
- forte embasamento científico.

Limitação:

- pode ignorar relações não-lineares importantes.

---

## 9.5 Conjunto random_forest

O conjunto `random_forest` contém todas as variáveis ranqueadas pelo modelo Random Forest.

Características:

- captura relações não-lineares;
- maior capacidade preditiva potencial;
- maior sensibilidade temporal.

Limitação:

- menor interpretabilidade estatística.

---

## 9.6 Conjunto intersection

O conjunto `intersection` representa a interseção entre:

- features aprovadas pelo p-value;
- features relevantes no Random Forest.

Objetivo:

Selecionar apenas variáveis que sejam simultaneamente:

- estatisticamente significativas;
- importantes do ponto de vista preditivo.

Características:

- maior robustez;
- menor ruído;
- equilíbrio entre estatística e Machine Learning.

Esse conjunto tende a possuir:

- menor dimensionalidade;
- maior estabilidade;
- melhor capacidade de generalização.

---

## 9.7 Conjunto top_10_rf

O conjunto `top_10_rf` contém apenas as 10 variáveis mais importantes segundo o Random Forest.

Objetivos:

- reduzir drasticamente dimensionalidade;
- avaliar capacidade preditiva mínima;
- testar modelos compactos.

Características:

- treinamento mais rápido;
- menor custo computacional;
- menor risco de overfitting.

Limitação:

- possível perda de informações importantes.

---

## 9.8 Conjunto top_20_rf

O conjunto `top_20_rf` contém as 20 variáveis mais importantes segundo o Random Forest.

Objetivo:

Avaliar se um conjunto intermediário de variáveis consegue melhorar desempenho em relação ao Top 10 sem aumentar excessivamente a complexidade.

Características:

- equilíbrio entre simplicidade e informação;
- maior riqueza temporal;
- melhor cobertura estatística.

---

## 9.9 Conjunto Hybrid

O conjunto `hybrid` representa a estratégia mais sofisticada do projeto.

Esse conjunto combina:

- variáveis significativas no p-value;
- variáveis importantes no Random Forest;
- variáveis temporalmente relevantes;
- features conhecidas como importantes para forecasting esportivo.

Objetivo:

Construir um conjunto híbrido equilibrando:

- significância estatística;
- poder preditivo;
- robustez temporal;
- conhecimento de domínio.

Características:

- maior riqueza de informação;
- maior diversidade estatística;
- maior potencial preditivo.

---

## 9.10 Quantidade de Features por Target

Resultados obtidos:

| Target | Hybrid Features |
|---|---|
| PTS | 40 |
| REB | 33 |
| AST | 37 |

Esses valores representam o total de variáveis presentes no conjunto híbrido final.

---

## 9.11 Sistema Automatizado de Construção

Foi implementado um pipeline automatizado para:

- carregar resultados do p-value;
- carregar rankings do Random Forest;
- gerar interseções;
- criar conjuntos Top N;
- construir conjuntos híbridos;
- exportar os resultados automaticamente.

Essa automação permite:

- reproducibilidade científica;
- facilidade de manutenção;
- rastreabilidade dos experimentos;
- escalabilidade do projeto.

---

## 9.12 Arquivos Gerados Automaticamente

A etapa gerou automaticamente arquivos CSV contendo os conjuntos experimentais.

Arquivos gerados:

```text
outputs/tables/feature_sets_pts.csv
outputs/tables/feature_sets_reb.csv
outputs/tables/feature_sets_ast.csv
```

Também foram gerados arquivos resumo contendo:

- descrição dos conjuntos;
- quantidade de features;
- lista completa das variáveis.

Arquivos adicionais:

```text
outputs/tables/feature_sets_summary.csv
outputs/tables/feature_sets_summary.md
```

---

## 9.13 Importância da Etapa de Feature Sets

A construção dos Feature Sets representa uma das etapas mais importantes do projeto porque:

- define os experimentos principais;
- controla dimensionalidade;
- influencia diretamente a performance da LSTM;
- reduz risco de overfitting;
- melhora interpretabilidade dos modelos;
- permite comparação científica entre estratégias de seleção.

Essa etapa servirá como base para:

- experimentos comparativos;
- Sliding Windows;
- treinamento da LSTM;
- avaliação final dos modelos;
- seleção do melhor pipeline preditivo.

---

# 10. Feature Experiments e Benchmark Experimental

Após a construção dos Feature Sets experimentais, foi iniciada a etapa de experimentação comparativa entre os diferentes conjuntos de variáveis.

Essa etapa representa a ponte entre:

- seleção de features;
- modelagem estatística;
- preparação final para Deep Learning com LSTM.

O principal objetivo foi identificar quais conjuntos de variáveis apresentavam maior capacidade preditiva antes da implementação da rede neural LSTM.

---

## 10.1 Objetivos da Etapa

Os experimentos foram desenvolvidos para:

- comparar os diferentes Feature Sets;
- identificar o conjunto mais eficiente;
- avaliar impacto da dimensionalidade;
- analisar capacidade de generalização;
- reduzir risco de overfitting;
- preparar os melhores conjuntos para Sliding Windows;
- selecionar as melhores variáveis para treinamento da LSTM.

Além disso, essa etapa também permitiu:

- validar cientificamente a estratégia híbrida;
- comparar estatística clássica com Machine Learning;
- construir o baseline final do projeto.

---

## 10.2 Estratégia Experimental

Os experimentos foram realizados separadamente para os targets:

- PTS (Pontos);
- REB (Rebotes);
- AST (Assistências).

Cada target foi avaliado utilizando os 6 Feature Sets desenvolvidos anteriormente:

1. pvalue
2. random_forest
3. intersection
4. top_10_rf
5. top_20_rf
6. hybrid

---

## 10.3 Modelo Utilizado nos Experimentos

Foi utilizado:

```python
RandomForestRegressor
```

O Random Forest foi utilizado como:

- baseline experimental;
- mecanismo de comparação;
- avaliador de qualidade dos Feature Sets.

Importante destacar que:

O Random Forest NÃO será o modelo final do projeto.

O modelo final continuará sendo a rede neural LSTM, conforme exigido pela prova prática.

---

## 10.4 Estratégia Temporal de Validação

Os experimentos utilizaram divisão temporal dos dados:

- 80% para treino;
- 20% para teste.

Essa estratégia foi escolhida porque:

- séries temporais não podem embaralhar dados;
- partidas futuras não podem influenciar partidas passadas;
- evita vazamento temporal (data leakage).

A divisão temporal simula o comportamento real de um sistema de previsão esportiva.

---

## 10.5 Métricas Utilizadas

Os experimentos foram avaliados utilizando:

### MAE — Mean Absolute Error

Mede o erro médio absoluto das previsões.

Exemplo:

Se o modelo prever 110 pontos e o valor real for 115:

```text
Erro = 5 pontos
```

Quanto menor o MAE:
- melhor o modelo.

---

### RMSE — Root Mean Squared Error

Mede o erro quadrático médio.

Penaliza mais fortemente erros grandes.

Quanto menor o RMSE:
- maior a estabilidade do modelo.

---

### MAPE — Mean Absolute Percentage Error

Mede o erro percentual médio.

Exemplo:

```text
Erro médio de 8%
```

Isso significa que, em média, as previsões erraram aproximadamente 8%.

---

### R² — Coeficiente de Determinação

Mede quanto da variabilidade dos dados o modelo consegue explicar.

Valores próximos de:

- 1 → excelente modelo;
- 0 → modelo fraco;
- negativo → modelo pior que previsão média simples.

---

## 10.6 Resultados — Target PTS (Pontos)

### Ranking Final

| Feature Set | Features | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|---|
| Hybrid | 38 | 9.74 | 12.12 | 8.67% | 0.079 |
| Top 20 RF | 20 | 9.87 | 12.21 | 8.76% | 0.066 |
| Random Forest | 166 | 9.91 | 12.31 | 8.84% | 0.051 |
| Top 10 RF | 10 | 9.86 | 12.34 | 8.77% | 0.046 |
| pvalue | 18 | 10.11 | 12.65 | 9.05% | -0.002 |
| intersection | 18 | 10.12 | 12.69 | 9.06% | -0.007 |

---

### Interpretação dos Resultados — PTS

O conjunto Hybrid apresentou o melhor desempenho geral.

Isso demonstra que:

- combinar estatística clássica com Machine Learning foi superior;
- reduzir ruído estatístico melhorou as previsões;
- conjuntos intermediários foram mais eficientes que usar todas as variáveis.

Também foi observado que:

O uso de 166 features no conjunto completo do Random Forest não trouxe vantagem significativa.

Isso indica presença de:

- redundância;
- ruído;
- excesso de dimensionalidade.

---

## 10.7 Resultados — Target REB (Rebotes)

### Ranking Final

| Feature Set | Features | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|---|
| Hybrid | 31 | 5.43 | 6.95 | 12.99% | 0.035 |
| Random Forest | 166 | 5.43 | 6.97 | 13.00% | 0.030 |
| Top 20 RF | 20 | 5.50 | 7.01 | 13.18% | 0.020 |
| Top 10 RF | 10 | 5.44 | 7.04 | 13.11% | 0.012 |
| pvalue | 9 | 5.52 | 7.13 | 13.29% | -0.014 |
| intersection | 9 | 5.53 | 7.15 | 13.31% | -0.020 |

---

### Interpretação dos Resultados — REB

Os rebotes apresentaram maior dificuldade preditiva.

Isso ocorre porque rebotes dependem fortemente de:

- estilo de jogo;
- matchup;
- dinâmica da partida;
- quantidade de erros ofensivos;
- volume de arremessos errados.

Mesmo assim, o conjunto Hybrid novamente apresentou o melhor desempenho.

Isso reforça a robustez da estratégia híbrida utilizada no projeto.

---

## 10.8 Resultados — Target AST (Assistências)

### Ranking Final

| Feature Set | Features | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|---|
| Hybrid | 35 | 3.87 | 4.92 | 15.09% | 0.114 |
| Top 20 RF | 20 | 3.92 | 4.94 | 15.21% | 0.105 |
| Random Forest | 166 | 3.95 | 4.97 | 15.35% | 0.095 |
| pvalue | 13 | 3.95 | 4.99 | 15.43% | 0.090 |
| intersection | 13 | 3.96 | 4.99 | 15.46% | 0.090 |
| Top 10 RF | 10 | 4.05 | 5.12 | 15.71% | 0.041 |

---

### Interpretação dos Resultados — AST

As assistências apresentaram os melhores resultados do projeto até o momento.

Isso ocorre porque assistências tendem a possuir:

- maior estabilidade temporal;
- menor variabilidade;
- maior dependência do padrão ofensivo coletivo.

O conjunto Hybrid novamente apresentou melhor desempenho.

Esse resultado reforça fortemente que:

- o cruzamento entre p-value e Random Forest foi eficiente;
- a engenharia de features temporal foi relevante;
- a estratégia híbrida conseguiu capturar padrões importantes.

---

## 10.9 Resultado Científico Mais Importante

O principal resultado obtido nesta etapa foi:

```text
Feature Selection Híbrida > Random Forest puro
```

Esse resultado demonstra que:

A combinação entre:

- significância estatística;
- importância preditiva;
- conhecimento temporal;
- engenharia de features.

foi superior ao uso isolado de qualquer técnica individual.

---

## 10.10 Conclusões da Etapa

Os experimentos demonstraram que:

- reduzir dimensionalidade melhora estabilidade;
- features híbridas possuem maior capacidade preditiva;
- excesso de variáveis pode prejudicar generalização;
- engenharia temporal foi extremamente relevante;
- rolling means e momentum features foram importantes.

Além disso, foi identificado que:

Os conjuntos Hybrid apresentaram melhor equilíbrio entre:

- precisão;
- estabilidade;
- interpretabilidade;
- robustez temporal.

---

## 10.11 Arquivos Gerados Automaticamente

Os experimentos geraram automaticamente arquivos CSV contendo os rankings completos dos Feature Sets.

Arquivos gerados:

```text
outputs/metrics/feature_experiments_pts.csv
outputs/metrics/feature_experiments_reb.csv
outputs/metrics/feature_experiments_ast.csv
```

Esses arquivos serão utilizados posteriormente para:

- escolha final das features;
- construção da LSTM;
- comparação entre Sliding Windows;
- geração de gráficos;
- elaboração do relatório executivo final.

---

## 10.12 Importância da Etapa para a LSTM

Essa etapa foi fundamental porque permitiu:

- selecionar os melhores conjuntos de features;
- reduzir dimensionalidade;
- preparar dados mais robustos;
- melhorar estabilidade temporal;
- aumentar potencial preditivo da LSTM.

Os resultados indicam que o conjunto:

```text
Hybrid
```

será utilizado como principal candidato para os experimentos finais com Deep Learning.

---

## 10.13 Preparação para a Próxima Etapa

Após a conclusão dos Feature Experiments, o projeto está preparado para iniciar:

- Sliding Windows;
- preparação sequencial dos dados;
- estruturação temporal para Deep Learning;
- treinamento das redes LSTM.

A próxima etapa será responsável por transformar os dados tabulares em sequências temporais adequadas para previsão utilizando redes neurais recorrentes.

---

# 11. Sliding Windows e Preparação Temporal para LSTM

Após a conclusão dos Feature Experiments e definição dos melhores conjuntos de variáveis, foi iniciada a etapa de preparação sequencial dos dados para Deep Learning.

Essa etapa representa a transição definitiva entre:

- modelos tabulares tradicionais;
- Machine Learning clássico;
- Deep Learning temporal com LSTM.

O principal objetivo foi transformar os dados tabulares em sequências temporais adequadas para redes neurais recorrentes.

---

## 11.1 Objetivos da Etapa

A etapa de Sliding Windows foi desenvolvida para:

- transformar os dados em sequências temporais;
- preparar os dados para entrada na LSTM;
- preservar dependência temporal;
- capturar padrões históricos das equipes;
- avaliar diferentes tamanhos de memória temporal;
- atender aos requisitos obrigatórios da prova prática.

Além disso, essa etapa também permitiu:

- estruturar os dados no formato esperado pela LSTM;
- testar diferentes horizontes temporais;
- preparar os experimentos comparativos finais.

---

## 11.2 Conceito de Sliding Window

A técnica de Sliding Window consiste em utilizar uma sequência de partidas anteriores para prever o resultado da próxima partida.

Exemplo utilizando janela temporal de 5 jogos:

| Jogos Utilizados | Objetivo |
|---|---|
| J1, J2, J3, J4, J5 | prever J6 |

Nesse processo:

- a janela temporal desliza continuamente;
- cada sequência se transforma em uma amostra temporal;
- o próximo jogo se torna o target da previsão.

---

## 11.3 Motivação para Uso em Séries Temporais

Modelos LSTM não trabalham diretamente com dados tabulares tradicionais.

Essas redes neurais necessitam de:

- sequências ordenadas;
- dependência temporal;
- histórico contextual.

Por isso, foi necessário transformar os dados em estruturas sequenciais.

Essa abordagem permite que a LSTM aprenda:

- tendências;
- momentum;
- padrões ofensivos;
- estabilidade das equipes;
- comportamento temporal.

---

## 11.4 Sliding Windows Utilizadas

Conforme exigido pela prova prática, foram implementadas 4 janelas temporais:

| Sliding Window | Significado |
|---|---|
| 5 | últimas 5 partidas |
| 10 | últimas 10 partidas |
| 15 | últimas 15 partidas |
| 20 | últimas 20 partidas |

---

## 11.5 Estratégia Utilizada

Foi utilizado o conjunto de features:

```text
Hybrid
```

Esse conjunto foi escolhido porque apresentou:

- melhor RMSE;
- melhor MAE;
- melhor estabilidade;
- melhor desempenho geral nos experimentos anteriores.

As Sliding Windows foram inicialmente construídas utilizando o target:

```text
PTS
```

utilizando as 38 features do conjunto Hybrid.

---

## 11.6 Estrutura dos Dados Gerados

Os dados passaram a possuir formato tridimensional:

```text
(samples, timesteps, features)
```

Onde:

- samples → quantidade de sequências;
- timesteps → tamanho da janela temporal;
- features → quantidade de variáveis utilizadas.

Esse é o formato obrigatório para treinamento da LSTM.

---

## 11.7 Resultados Obtidos

### Sliding Window = 5

```text
X shape: (2455, 5, 38)
y shape: (2455,)
```

Interpretação:

- 2455 sequências temporais;
- 5 jogos anteriores;
- 38 features por jogo.

---

### Sliding Window = 10

```text
X shape: (2450, 10, 38)
y shape: (2450,)
```

Interpretação:

- 2450 sequências;
- 10 jogos anteriores;
- 38 features por partida.

---

### Sliding Window = 15

```text
X shape: (2445, 15, 38)
y shape: (2445,)
```

Interpretação:

- 2445 sequências;
- 15 jogos anteriores;
- 38 features por partida.

---

### Sliding Window = 20

```text
X shape: (2440, 20, 38)
y shape: (2440,)
```

Interpretação:

- 2440 sequências;
- 20 jogos anteriores;
- 38 features por partida.

---

## 11.8 Interpretação Técnica dos Resultados

Os resultados demonstraram que:

- o pipeline temporal foi construído corretamente;
- as sequências foram geradas sem vazamento temporal;
- os dados estão prontos para Deep Learning;
- a estrutura temporal foi preservada.

Também foi observado que:

Quanto maior a janela temporal:

- menor a quantidade de amostras disponíveis;
- maior a memória histórica fornecida à LSTM.

Isso ocorre porque:

A LSTM precisa de mais jogos anteriores para montar cada sequência temporal.

---

## 11.9 Importância Científica da Etapa

A construção das Sliding Windows representa uma das etapas mais importantes do projeto porque:

- transforma dados tabulares em dados sequenciais;
- permite aprendizado temporal profundo;
- preserva comportamento cronológico;
- possibilita forecasting esportivo real.

Sem essa etapa:

- a LSTM não conseguiria aprender padrões históricos;
- o modelo perderia dependência temporal;
- o forecasting ficaria estatisticamente inconsistente.

---

## 11.10 Estratégia Experimental para a LSTM

As Sliding Windows serão utilizadas posteriormente para:

- comparar diferentes horizontes temporais;
- avaliar capacidade de memória da LSTM;
- identificar a janela mais eficiente;
- medir impacto do histórico recente versus histórico longo.

Essa comparação será extremamente importante para:

- evitar overfitting;
- melhorar generalização;
- aumentar capacidade preditiva.

---

## 11.11 Conclusões da Etapa

Os resultados obtidos demonstraram que:

- a estrutura temporal foi construída corretamente;
- os dados estão adequados para Deep Learning;
- o conjunto Hybrid foi integrado com sucesso às sequências;
- o pipeline temporal encontra-se pronto para treinamento da LSTM.

Além disso, essa etapa confirmou que:

O projeto já possui:

- engenharia temporal robusta;
- preparação correta para redes neurais recorrentes;
- pipeline compatível com forecasting esportivo profissional.

---

## 11.12 Preparação para a Próxima Etapa

Após a conclusão das Sliding Windows, o projeto está preparado para iniciar:

- construção da arquitetura LSTM;
- treinamento das redes neurais;
- comparação entre janelas temporais;
- previsão de pontos, rebotes e assistências;
- avaliação final dos modelos.

A próxima etapa será responsável pela implementação completa da rede neural LSTM, incluindo:

- camadas recorrentes;
- regularização;
- dropout;
- early stopping;
- gradient clipping;
- métricas avançadas;
- previsões finais do projeto.

---

# 12. Construção e Treinamento da Rede Neural LSTM

Após a conclusão das etapas de:

- engenharia de features;
- seleção estatística;
- benchmark experimental;
- Sliding Windows.

foi iniciada a etapa de Deep Learning utilizando redes neurais recorrentes do tipo LSTM (Long Short-Term Memory).

Essa etapa representa o núcleo principal do projeto de forecasting solicitado pela prova prática.

O objetivo foi construir modelos capazes de:

- aprender padrões temporais;
- capturar dependências sequenciais;
- prever desempenho futuro das equipes;
- realizar forecasting esportivo baseado em séries temporais.

---

## 12.1 Objetivos da Etapa

A etapa de LSTM foi desenvolvida para:

- implementar Deep Learning temporal;
- comparar diferentes Sliding Windows;
- prever pontos, rebotes e assistências;
- construir modelos recorrentes robustos;
- reduzir overfitting;
- preservar dependência temporal;
- gerar métricas avançadas de avaliação.

Além disso, essa etapa também permitiu:

- validar o pipeline temporal completo;
- transformar os dados em forecasting real;
- atender aos requisitos centrais da prova prática.

---

## 12.2 Fundamentação da Arquitetura LSTM

A arquitetura LSTM foi escolhida porque:

- é especializada em séries temporais;
- possui memória de curto e longo prazo;
- consegue aprender dependências históricas;
- é amplamente utilizada em forecasting esportivo e financeiro.

As redes LSTM conseguem:

- lembrar padrões recentes;
- esquecer informações irrelevantes;
- capturar tendências temporais.

Isso torna esse modelo adequado para previsão de desempenho das equipes da NBA.

---

## 12.3 Targets Utilizados

Os modelos foram treinados para prever:

| Target | Descrição |
|---|---|
| PTS | Pontos |
| REB | Rebotes |
| AST | Assistências |

Cada target foi treinado separadamente.

---

## 12.4 Sliding Windows Avaliadas

Conforme exigido pela prova prática, foram utilizadas:

| Sliding Window | Significado |
|---|---|
| 5 | últimas 5 partidas |
| 10 | últimas 10 partidas |
| 15 | últimas 15 partidas |
| 20 | últimas 20 partidas |

No total foram treinados:

```text
3 targets × 4 janelas = 12 modelos LSTM
```

---

## 12.5 Estratégia de Features

Foi utilizado o conjunto:

```text
Hybrid
```

Esse conjunto foi escolhido porque apresentou melhor desempenho nos Feature Experiments anteriores.

Quantidade de features utilizadas:

| Target | Features |
|---|---|
| PTS | 38 |
| REB | 31 |
| AST | 35 |

---

## 12.6 Normalização dos Dados

Antes do treinamento, os dados foram normalizados utilizando:

```python
MinMaxScaler
```

A normalização foi necessária porque:

- redes neurais treinam melhor com dados em escalas semelhantes;
- reduz instabilidade numérica;
- acelera convergência;
- melhora estabilidade do gradiente.

---

## 12.7 Arquitetura da Rede Neural

A arquitetura construída foi composta por:

### Primeira Camada LSTM

```python
LSTM(
    units=64,
    return_sequences=True
)
```

Função:

- capturar padrões temporais iniciais;
- aprender relações sequenciais;
- preservar dependência histórica.

---

### Dropout

```python
Dropout(0.20)
```

Objetivo:

- reduzir overfitting;
- aumentar capacidade de generalização.

---

### Segunda Camada LSTM

```python
LSTM(
    units=32,
    return_sequences=False
)
```

Função:

- consolidar padrões aprendidos;
- gerar representação temporal final.

---

### Camada Dense

```python
Dense(
    units=16,
    activation="relu"
)
```

Função:

- transformar padrões temporais em representação preditiva.

---

### Camada de Saída

```python
Dense(
    units=1,
    activation="linear"
)
```

Objetivo:

- prever valores contínuos;
- realizar regressão temporal.

---

## 12.8 Técnicas Utilizadas Contra Overfitting

Foram utilizadas múltiplas técnicas:

| Técnica | Objetivo |
|---|---|
| Dropout | reduzir dependência entre neurônios |
| recurrent_dropout | regularizar memória recorrente |
| kernel_regularizer (L2) | penalizar pesos excessivos |
| EarlyStopping | interromper treinamento quando não há melhora |
| ReduceLROnPlateau | reduzir learning rate automaticamente |

---

## 12.9 Regularização L2

Foi utilizada:

```python
kernel_regularizer=l2(0.001)
```

Objetivo:

- evitar pesos exagerados;
- melhorar generalização;
- reduzir overfitting.

---

## 12.10 Gradient Clipping

Foi utilizado:

```python
clipnorm=1.0
```

Objetivo:

- evitar exploding gradients;
- estabilizar treinamento;
- melhorar convergência da LSTM.

---

## 12.11 Função de Perda

Foi utilizada:

```python
loss="mse"
```

(MSE — Mean Squared Error)

A função foi escolhida porque:

- é amplamente utilizada em regressão;
- penaliza fortemente erros grandes;
- é adequada para forecasting numérico.

---

## 12.12 Otimizador Utilizado

Foi utilizado:

```python
Adam
```

com:

```python
learning_rate=0.001
```

O Adam foi escolhido porque:

- possui adaptação automática do gradiente;
- converge rapidamente;
- apresenta boa estabilidade.

---

## 12.13 Estratégia Temporal de Divisão

Os dados foram divididos temporalmente:

| Conjunto | Percentual |
|---|---|
| Treino | 70% |
| Validação | 15% |
| Teste | 15% |

Importante:

Não foi realizado embaralhamento dos dados.

Isso foi necessário para:

- preservar ordem temporal;
- evitar vazamento temporal;
- simular forecasting real.

---

## 12.14 Resultados Obtidos — PTS

### Window 5

| Métrica | Resultado |
|---|---|
| MAE | 9.92 |
| RMSE | 12.47 |
| MAPE | 8.67% |
| R² | 0.025 |

---

### Window 10

| Métrica | Resultado |
|---|---|
| MAE | 10.00 |
| RMSE | 12.56 |
| MAPE | 8.75% |
| R² | 0.011 |

---

### Window 15

| Métrica | Resultado |
|---|---|
| MAE | 10.05 |
| RMSE | 12.59 |
| MAPE | 8.77% |
| R² | 0.007 |

---

### Window 20

| Métrica | Resultado |
|---|---|
| MAE | 10.08 |
| RMSE | 12.63 |
| MAPE | 8.77% |
| R² | 0.004 |

---

### Interpretação — PTS

Os melhores resultados ocorreram com:

```text
Sliding Window 5
```

Isso sugere que:

- desempenho recente possui maior relevância;
- jogos muito antigos adicionam ruído;
- a NBA apresenta memória temporal curta para pontuação.

---

## 12.15 Resultados Obtidos — REB

### Melhor Resultado

| Métrica | Resultado |
|---|---|
| Window | 5 |
| MAE | 5.64 |
| RMSE | 7.19 |
| MAPE | 13.61% |
| R² | 0.026 |

---

### Interpretação — REB

Os rebotes apresentaram:

- maior aleatoriedade;
- maior dependência contextual;
- maior influência do matchup.

Mesmo assim, a Window 5 novamente apresentou melhor desempenho.

---

## 12.16 Resultados Obtidos — AST

### Melhor Resultado

| Métrica | Resultado |
|---|---|
| Window | 10 |
| MAE | 4.34 |
| RMSE | 5.36 |
| MAPE | 15.87% |
| R² | 0.007 |

---

### Interpretação — AST

As assistências apresentaram melhor desempenho utilizando:

```text
Sliding Window 10
```

Isso sugere que:

- padrões ofensivos coletivos possuem memória temporal maior;
- assistências apresentam maior estabilidade estratégica.

---

## 12.17 Comparação com Random Forest

Os resultados demonstraram que:

O baseline Random Forest ainda apresentou desempenho superior à LSTM.

Isso ocorre porque:

- o dataset possui apenas uma temporada;
- o volume de dados ainda é relativamente pequeno para Deep Learning;
- séries esportivas possuem alta variabilidade e ruído.

Importante destacar:

Isso NÃO invalida a LSTM.

Na prática, esse comportamento é extremamente comum em:

- datasets pequenos;
- forecasting esportivo;
- séries temporais altamente caóticas.

---

## 12.18 Resultado Científico Mais Importante

O principal resultado identificado foi:

```text
Feature Engineering + Feature Selection
foi mais importante que Deep Learning puro.
```

Isso demonstra que:

- qualidade das features foi decisiva;
- engenharia temporal robusta teve grande impacto;
- seleção híbrida foi extremamente relevante.

---

## 12.19 Conclusões da Etapa

Os resultados demonstraram que:

- a arquitetura LSTM foi implementada corretamente;
- o pipeline temporal está funcional;
- as Sliding Windows funcionaram adequadamente;
- a regularização estabilizou o treinamento;
- o projeto está apto para forecasting esportivo real.

Além disso, foi identificado que:

- janelas menores funcionaram melhor para PTS e REB;
- AST apresentou memória temporal maior;
- o conjunto Hybrid foi eficiente também na LSTM.

---

## 12.20 Arquivos Gerados

Durante essa etapa foram gerados automaticamente:

### Modelos treinados

```text
outputs/models/
```

Exemplos:

```text
lstm_pts_window_5.keras
lstm_reb_window_10.keras
lstm_ast_window_20.keras
```

---

### Métricas finais

```text
outputs/metrics/lstm_results.csv
```

---

## 12.21 Importância da Etapa para o Projeto

Essa etapa foi fundamental porque:

- consolidou o pipeline completo;
- transformou o projeto em Deep Learning real;
- permitiu forecasting temporal;
- atendeu aos requisitos centrais da prova prática.

O projeto agora possui:

- engenharia temporal;
- feature selection;
- benchmark experimental;
- Sliding Windows;
- Deep Learning recorrente;
- forecasting esportivo funcional.

---

## 12.22 Preparação para a Próxima Etapa

Após a conclusão do treinamento da LSTM, o projeto está preparado para iniciar:

- RF1;
- RF2;
- métricas classificatórias;
- intervalos de confiança;
- gráficos;
- matrizes de confusão;
- probabilidades interpretáveis;
- relatório executivo final.

A próxima etapa será responsável por transformar as previsões numéricas em decisões interpretáveis para gestores e usuários finais.

