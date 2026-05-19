# Metodologia do Projeto

## Objetivo

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

# Bases de Dados

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

# Engenharia de Dados Inicial

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

# Organização do Projeto

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