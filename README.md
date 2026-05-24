# NBA LSTM Forecasting — UFAPE

Sistema de Forecasting da NBA utilizando séries temporais, engenharia de features e redes neurais LSTM.

Projeto desenvolvido para a disciplina de Aprendizado de Máquina do Programa de Pós-Graduação em Ciência da Computação (PPCOMP/UFAPE).

---

# Objetivo do Projeto

O objetivo deste projeto foi desenvolver modelos baseados em redes neurais recorrentes do tipo LSTM (Long Short-Term Memory) para realizar forecasting de estatísticas de equipes da NBA durante a temporada regular 2025/2026.

As previsões foram realizadas para as equipes:

- Boston Celtics
- Denver Nuggets

O sistema prevê:

- pontos (PTS);
- rebotes (REB);
- assistências (AST).

Além da modelagem preditiva com LSTM, o projeto também implementa:

- engenharia de features;
- seleção estatística com p-value;
- seleção de variáveis com Random Forest;
- benchmark entre feature sets;
- Sliding Windows temporais;
- métricas de regressão e classificação;
- análise probabilística;
- interpretação quantitativa dos resultados.

---

# Requisitos Funcionais Implementados

## RF1 — Previsão Acima da Média da Equipe

O sistema calcula a probabilidade das equipes superarem suas próprias médias históricas de:

- pontos;
- rebotes;
- assistências.

---

## RF2 — Previsão Acima de Thresholds

O sistema calcula a probabilidade das equipes realizarem:

| Estatística | Threshold |
|---|---|
| PTS | 100 |
| REB | 30 |
| AST | 20 |

---

## RF3 — Forecasting Numérico

O sistema prevê numericamente:

- pontos;
- rebotes;
- assistências.

Exemplo:

```text
126.2 pontos previstos
```

---

## RF4 — Métricas Estatísticas

O sistema calcula:

### Métricas de Regressão

- MAE;
- RMSE;
- MAPE;
- R².

### Métricas de Classificação

- Accuracy;
- F1-score;
- AUC;
- matriz de confusão.

Também são calculados:

- erro médio;
- over prediction;
- under prediction;
- intervalos de confiança;
- análises residuais.

---

## RF5 — Interpretação Probabilística

O sistema converte previsões numéricas em interpretações quantitativas e probabilísticas.

Exemplos:

```text
85% de chance (alta) de superar 100 pontos.
```

```text
57% de chance (moderada) de realizar 20 ou mais assistências.
```

```text
A previsão é que a equipe faça
115 pontos, 46 rebotes e 31 assistências.
```

---

## RF6 — Visualizações

O sistema gera automaticamente:

- gráficos Real vs Predito;
- histogramas de erro;
- barras de confiança;
- matrizes de confusão derivadas das classificações probabilísticas.

Os gráficos são armazenados em:

```text
outputs/figures/
```

---

## RF7 — Relatório Técnico Executivo

O projeto também consolida os principais resultados experimentais em formato interpretativo, permitindo análise quantitativa e suporte à tomada de decisão.

As interpretações incluem:

- probabilidades;
- intervalos de confiança;
- desempenho dos modelos;
- análise dos erros;
- comparação entre janelas temporais;
- interpretação dos targets previstos.

---

# Tecnologias Utilizadas

## Linguagem

- Python 3.10+

---

## Bibliotecas Principais

- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- statsmodels

---

# Estrutura do Projeto

```text
nba-lstm-forecasting-ufape/
│
├── data/
│   └── raw/
│       ├── BaseA.csv
│       └── BaseB.csv
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   └── tables/
│
├── reports/
│
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── reports/
│   └── visualization/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Bases de Dados

Foram utilizadas duas bases fornecidas para a atividade:

| Arquivo | Função |
|---|---|
| BaseA.csv | estatísticas principais |
| BaseB.csv | estatísticas complementares |

O merge entre as bases foi realizado utilizando:

```python
GAME_ID + TEAM_NAME
```

Essa estratégia evitou:

- duplicações;
- inconsistências;
- vazamento de informações;
- expansão indevida do dataset.

---

# Pipeline Experimental

A pipeline experimental foi organizada nas seguintes etapas:

| Etapa | Descrição |
|---|---|
| Engenharia de Dados | carregamento, merge e preprocessamento |
| Feature Engineering | criação de features temporais |
| Correlation Filter | remoção de multicolinearidade |
| p-value Selection | seleção estatística |
| Random Forest | importância das variáveis |
| Feature Sets | construção de conjuntos híbridos |
| Feature Experiments | benchmark entre conjuntos |
| Sliding Windows | geração de sequências temporais |
| LSTM | Deep Learning temporal |
| Predictor | previsões finais |
| Evaluation | métricas estatísticas |
| RF6 | gráficos e visualizações |
| RF7 | interpretação técnica dos resultados |

---

# Engenharia de Features

Foram criadas múltiplas categorias de features:

## Features Temporais

- rolling mean;
- rolling std;
- rolling max;
- rolling min.

---

## Features de Tendência

- momentum;
- variação percentual;
- tendência recente.

---

## Features de Estabilidade

- volatilidade;
- consistência temporal.

---

## Features Contextuais

- home/away;
- win streak;
- defeat streak.

---

# Seleção de Features

A seleção ocorreu em múltiplas etapas.

## 1. Correlation Filter

Remoção de features altamente correlacionadas.

Critério utilizado:

```text
Correlação > 0.90
```

---

## 2. p-value Selection

Seleção estatística utilizando regressão OLS.

Critério:

```text
p-value < 0.05
```

---

## 3. Random Forest Importance

Rankeamento das variáveis mais importantes utilizando:

- RandomForestRegressor;
- importance score.

---

# Feature Sets Avaliados

Foram construídos seis conjuntos de features para comparação experimental:

| Feature Set | Descrição |
|---|---|
| pvalue | features selecionadas por significância estatística com p-value |
| random_forest | features com importância positiva no Random Forest |
| intersection | interseção entre as features selecionadas por p-value e Random Forest |
| top_10_rf | 10 features mais importantes segundo o Random Forest |
| top_20_rf | 20 features mais importantes segundo o Random Forest |
| hybrid | combinação entre features do p-value, Top 20 do Random Forest e features técnicas relevantes |

O conjunto `hybrid` foi construído de forma estratégica, combinando:

- features selecionadas por p-value;
- as 20 features mais importantes segundo Random Forest;
- variáveis técnicas relevantes para o contexto esportivo, como mando de quadra, descanso, sequência de vitórias/derrotas, eficiência e estatísticas defasadas.

Esses conjuntos foram avaliados experimentalmente por meio de um benchmark com Random Forest Regressor, utilizando validação temporal 80/20. O objetivo foi comparar o desempenho dos diferentes grupos de variáveis antes do treinamento final com LSTM.

---

# Melhor Feature Set

O conjunto `Hybrid` apresentou desempenho competitivo e foi selecionado como principal conjunto utilizado nos experimentos finais com LSTM.

---

# Sliding Windows

Foram avaliadas janelas temporais de:

| Window | Descrição |
|---|---|
| 5 | últimos 5 jogos |
| 10 | últimos 10 jogos |
| 15 | últimos 15 jogos |
| 20 | últimos 20 jogos |

As Sliding Windows foram utilizadas para transformar os dados tabulares em sequências temporais compatíveis com LSTM.

---

# Arquitetura LSTM

A arquitetura principal foi composta por:

```python
LSTM(64)
Dropout(0.20)

LSTM(32)
Dropout(0.20)

Dense(16, activation="relu")
Dense(1, activation="linear")
```

---

# Técnicas Contra Overfitting

Foram utilizadas:

- Dropout;
- recurrent_dropout;
- EarlyStopping;
- ReduceLROnPlateau;
- Regularização L2;
- Gradient Clipping.

---

# Métricas Utilizadas

## Regressão

- MAE;
- RMSE;
- MAPE;
- R².

---

## Classificação

- Accuracy;
- F1-score;
- AUC;
- matriz de confusão.

---

# Resultados Experimentais

As melhores combinações observadas durante os experimentos comparativos foram:

| Target | Melhor Window |
|---|---|
| PTS | 5 |
| REB | 20 |
| AST | 10 |

As previsões finais implementadas na pipeline utilizaram:

| Target | Window Utilizada |
|---|---|
| PTS | 5 |
| REB | 5 |
| AST | 10 |

---

# Como Executar

## 1. Clonar o repositório

```bash
git clone <repository-url>
```

---

## 2. Criar ambiente virtual

```bash
python -m venv venv
```

---

## 3. Ativar ambiente virtual

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## 4. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 5. Executar o projeto

```bash
python main.py
```

A execução gera automaticamente:
- métricas;
- gráficos;
- tabelas;
- previsões;
- resultados experimentais.

Os modelos treinados podem ser regenerados automaticamente durante a execução da pipeline.
---

# Outputs Gerados

## Métricas

```text
outputs/metrics/
```

---

## Gráficos

```text
outputs/figures/
```

---

## Previsões

```text
outputs/predictions/
```

---

## Tabelas

```text
outputs/tables/
```

---

# Reprodutibilidade

O projeto foi desenvolvido com foco em rastreabilidade experimental e reprodutibilidade científica, utilizando:

- modularização;
- versionamento Git;
- separação entre dados, modelos e métricas;
- organização estruturada de outputs;
- controle de dependências via `requirements.txt`.

---

# Organização do Código

O projeto foi desenvolvido utilizando:

- modularização;
- orientação a objetos;
- separação por responsabilidades;
- documentação técnica;
- organização por pipeline experimental.

---

# Trabalhos Futuros

Possíveis melhorias futuras:

- Bidirectional LSTM;
- GRU;
- Attention Mechanism;
- Ensemble Learning;
- utilização de múltiplas temporadas;
- estatísticas de jogadores;
- análise do adversário direto;
- fatigue score;
- back-to-back games.

---

# Autor

Eneida Glauce de Araújo Medeiros

Programa de Pós-Graduação em Ciência da Computação — PPCOMP/UFAPE

---

# Licença

Projeto acadêmico desenvolvido para fins educacionais.