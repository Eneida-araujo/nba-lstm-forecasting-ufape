# hybrid-lstm-nba-forecasting

Arquitetura híbrida baseada em LSTM para previsão de estatísticas da NBA utilizando sequências temporais, engenharia de atributos, seleção híbrida de features e análise preditiva executiva.

---

# Visão Geral

Este projeto propõe um framework híbrido de previsão baseado em redes neurais Long Short-Term Memory (LSTM) para previsão de métricas de desempenho de equipes da NBA.

O sistema combina:

- modelagem temporal com Sliding Windows;
- engenharia de atributos híbrida;
- seleção estatística de features utilizando p-value;
- ranqueamento de importância com Random Forest;
- estratégia híbrida de interseção de features;
- previsão por regressão;
- interpretação executiva orientada à classificação;
- pipeline automatizado de visualizações.

O framework prevê três indicadores de desempenho no basquete:

- Pontos (PTS)
- Rebotes (REB)
- Assistências (AST)

O projeto foi desenvolvido como pipeline experimental de forecasting esportivo utilizando Deep Learning temporal.

---

# Objetivos da Pesquisa

O projeto está organizado em seis requisitos funcionais (RF):

| RF | Descrição |
|---|---|
| RF1 | Prever se a equipe irá superar sua própria média da temporada |
| RF2 | Prever se a equipe irá superar thresholds fixos |
| RF3 | Prever os valores numéricos de PTS, REB e AST |
| RF4 | Comparar diferentes tamanhos de Sliding Window |
| RF5 | Comparar diferentes estratégias de seleção de features |
| RF6 | Gerar relatórios visuais automatizados |

---

# Thresholds Fixos (RF2)

Os thresholds utilizados seguem exatamente o especificado na atividade acadêmica:

| Métrica | Threshold |
|---|---:|
| PTS | 100 |
| REB | 30 |
| AST | 20 |

Esses thresholds foram mantidos conforme solicitado no enunciado da atividade.

Como equipes modernas da NBA frequentemente ultrapassam esses valores, algumas matrizes de confusão podem apresentar desbalanceamento de classes concentrado em casos positivos ("Sim").

Esse comportamento representa a distribuição real dos dados e não um erro de implementação.

---

# Arquitetura do Projeto

```text
Dataset
   ↓
Pré-processamento
   ↓
Engenharia de Features
   ↓
Seleção de Features
   ├── p-value
   ├── Random Forest
   └── Interseção Híbrida
   ↓
Geração de Sliding Windows
   ↓
Treinamento LSTM
   ↓
Avaliação
   ├── RMSE
   ├── MAE
   ├── R²
   └── Interpretação Classificatória
   ↓
Sistema de Previsão
   ↓
Pipeline de Visualização
```

---

# Tecnologias Utilizadas

- Python 3.11+
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

# Dataset

O dataset contém estatísticas históricas de partidas da NBA.

Principais targets:

- PTS
- REB
- AST

Exemplos de features utilizadas:

- médias móveis;
- eficiência ofensiva;
- eficiência de arremessos;
- tendências temporais;
- estatísticas acumuladas;
- features derivadas e interações.

---

# Estratégia de Seleção de Features

O projeto avalia três abordagens de seleção de atributos:

## 1. Seleção Estatística (p-value)

As features são selecionadas utilizando análise de significância estatística.

## 2. Importância por Random Forest

As features são ranqueadas com base nos scores de importância obtidos pelo modelo Random Forest.

## 3. Feature Set Híbrido

O conjunto híbrido final é construído pela interseção entre:

- features estatisticamente significativas;
- features relevantes no Random Forest.

Essa estratégia busca reduzir ruído e aumentar robustez preditiva.

---

# Sliding Windows

As sequências temporais são geradas utilizando múltiplos tamanhos de janela:

| Window Size |
|---:|
| 5 |
| 10 |
| 15 |
| 20 |

O melhor modelo é selecionado automaticamente com base no menor RMSE.

---

# Previsão com LSTM

O pipeline utiliza redes neurais LSTM para aprendizado temporal das sequências.

Os modelos são treinados separadamente para:

- PTS
- REB
- AST

Cada target carrega automaticamente:

- melhor Sliding Window;
- melhor modelo treinado;
- melhor configuração baseada em RMSE.

---

# Sistema Executivo de Previsão

O módulo `NBAPredictor` gera previsões executivas para partidas futuras.

O sistema fornece:

- previsões numéricas;
- interpretação probabilística;
- comparação com média da temporada;
- comparação com thresholds fixos;
- interpretação baseada em RMSE.

Exemplo de saída:

```text
PTS previsto: 114.1
REB previsto: 41.0
AST previsto: 25.4
```

---

# Métricas de Avaliação

Métricas de regressão:

- RMSE
- MAE
- R²

Análises classificatórias:

- matriz de confusão;
- interpretação baseada em threshold;
- classificação executiva probabilística.

---

# Pipeline de Visualização

O projeto gera automaticamente gráficos analíticos.

Visualizações implementadas:

## Real vs Previsto

Comparação temporal entre valores reais e previstos.

## Histograma dos Erros

Análise da distribuição dos resíduos.

## Barra de Confiança

Intervalo de confiança calculado por:

```math
margem = 1.96 \times RMSE
```

## Matriz de Confusão

Avaliação classificatória baseada nos thresholds definidos pela atividade.

---

# Exemplos de Visualizações

## Real vs Previsto

- análise de estabilidade temporal;
- suavização das previsões;
- comparação entre volatilidade real e prevista.

## Histogramas dos Erros

- análise da distribuição residual;
- identificação de outliers;
- inspeção de variância.

## Barras de Confiança

- visualização da incerteza preditiva;
- interpretação baseada em RMSE.

## Matrizes de Confusão

- desempenho classificatório;
- análise executiva baseada em thresholds.

---

# Estrutura do Projeto

```text
hybrid-lstm-nba-forecasting/
│
├── data/
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   └── tables/
│
├── src/
│   ├── features/
│   ├── models/
│   ├── predictors/
│   ├── visualization/
│   └── utils/
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── main.py
```

---

# Seleção Automática do Melhor Modelo

O sistema seleciona automaticamente a melhor configuração utilizando o menor RMSE registrado em:

```text
outputs/metrics/lstm_results.csv
```

Esse comportamento é utilizado tanto no pipeline de previsão quanto no pipeline de visualização.

---

# Execução Rápida

git clone https://github.com/Eneida-araujo/nba-lstm-forecasting-ufape.git

cd nba-lstm-forecasting-ufape

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python main.py

# Ambiente utilizado

Python 3.10.11
TensorFlow 2.21
scikit-learn
pandas
numpy
matplotlib
statsmodels

# Reprodutibilidade

## 1. Clonar repositório

```bash
git clone https://github.com/seu-usuario/hybrid-lstm-nba-forecasting.git
```

## 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## 3. Executar projeto

```bash
python main.py
```

---

# Contribuições Científicas

Este projeto contribui ao combinar:

- Deep Learning temporal;
- seleção estatística de atributos;
- importância de features com Machine Learning;
- engenharia híbrida de features;
- previsão esportiva executiva;
- análise preditiva interpretável.

---

# Possíveis Extensões Futuras

Possíveis melhorias futuras incluem:

- previsão por jogador;
- arquiteturas Transformer;
- mecanismos de atenção;
- integração com mercado de apostas;
- previsão probabilística avançada;
- métodos de Explainable AI (SHAP/LIME);
- APIs de previsão em tempo real.

---

# Autores

Eneida Glauce de Araújo Medeiros

Kenneth Yan Santana Oliveira

Mestranda em Ciência da Computação  
Universidade Federal do Agreste de Pernambuco (UFAPE)

---

# Licença

Projeto desenvolvido para fins acadêmicos e de pesquisa.
