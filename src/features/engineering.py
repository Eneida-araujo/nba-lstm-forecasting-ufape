"""
=========================================================
engineering.py
=========================================================

Responsável pela criação das features do projeto.

Nesta etapa criamos variáveis que ajudam o modelo LSTM
a compreender o comportamento recente de cada equipe.

Princípio fundamental:
NENHUMA feature temporal pode usar informação do jogo atual
para prever o próprio jogo.

Por isso, usamos sempre:
    shift(1)

Isso significa:
    "use apenas informações anteriores ao jogo alvo".

Essa proteção evita data leakage, ou seja, vazamento de dados.
=========================================================
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Classe responsável pela engenharia de features.

    As features criadas aqui serão usadas posteriormente em:
    - seleção por p-value;
    - seleção por Random Forest;
    - experimentos de features;
    - modelos LSTM.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def create_home_feature(self):
        """
        Cria a variável IS_HOME.

        A coluna MATCHUP informa se o jogo foi em casa ou fora.

        Na NBA:
        - "vs." indica jogo em casa;
        - "@" indica jogo fora.

        Essa feature é importante porque equipes costumam ter
        desempenho diferente jogando em casa.
        """

        self.df["IS_HOME"] = self.df["MATCHUP"].apply(
            lambda x: 1 if "vs." in str(x) else 0
        )

        return self

    def create_days_rest(self):
        """
        Cria a variável DAYS_REST.

        DAYS_REST representa quantos dias de descanso a equipe teve
        desde a partida anterior.

        Essa variável pode influenciar o desempenho físico da equipe.

        Exemplo:
        - 1 dia de descanso pode indicar calendário apertado;
        - 3 ou mais dias podem indicar recuperação maior.
        """

        self.df["DAYS_REST"] = (
            self.df.groupby("TEAM_NAME")["GAME_DATE"]
            .diff()
            .dt.days
        )

        # Primeiro jogo da equipe não possui jogo anterior.
        # Preenchemos com a mediana para não criar valor ausente.
        self.df["DAYS_REST"] = self.df["DAYS_REST"].fillna(
            self.df["DAYS_REST"].median()
        )

        return self

    def create_lag_features(self):
        """
        Cria variáveis defasadas.

        As variáveis W, L e W_PCT representam o histórico acumulado
        da equipe até determinado momento.

        Para evitar vazamento temporal, usamos shift(1), pois o modelo
        só pode conhecer o desempenho acumulado ANTES do jogo alvo.
        """

        lag_columns = [
            "W",
            "L",
            "W_PCT",
            "PTS",
            "REB",
            "AST",
            "PLUS_MINUS",
            "FG_PCT",
            "FG3_PCT",
            "FT_PCT",
            "OREB",
            "DREB",
            "TOV"
        ]

        for column in lag_columns:
            if column in self.df.columns:
                self.df[f"{column}_LAG1"] = (
                    self.df.groupby("TEAM_NAME")[column]
                    .shift(1)
                )

        return self

    def create_rolling_features(self):
        """
        Cria médias móveis e desvios móveis.

        Rolling windows usados:
        - 5 jogos;
        - 10 jogos;
        - 15 jogos;
        - 20 jogos.

        Esses tamanhos são compatíveis com os Sliding Windows exigidos
        na prova.

        Todas as estatísticas usam shift(1), garantindo que o jogo atual
        não entre no cálculo da própria previsão.
        """

        rolling_columns = [
            "PTS",
            "REB",
            "AST",
            "FGM",
            "FGA",
            "FG_PCT",
            "FG3M",
            "FG3A",
            "FG3_PCT",
            "FTM",
            "FTA",
            "FT_PCT",
            "OREB",
            "DREB",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PLUS_MINUS"
        ]

        windows = [5, 10, 15, 20]

        for column in rolling_columns:
            if column not in self.df.columns:
                continue

            for window in windows:
                self.df[f"{column}_ROLLING_MEAN_{window}"] = (
                    self.df.groupby("TEAM_NAME")[column]
                    .transform(
                        lambda x: x.shift(1).rolling(window).mean()
                    )
                )

                self.df[f"{column}_ROLLING_STD_{window}"] = (
                    self.df.groupby("TEAM_NAME")[column]
                    .transform(
                        lambda x: x.shift(1).rolling(window).std()
                    )
                )

        return self

    def create_momentum_features(self):
        """
        Cria variáveis de momentum.

        Momentum representa mudança recente de desempenho.

        Exemplo:
        PTS_MOMENTUM_5_10 =
            média dos últimos 5 jogos - média dos últimos 10 jogos

        Se o valor for positivo:
            a equipe está pontuando mais recentemente.

        Se for negativo:
            a equipe pode estar em queda de desempenho.
        """

        base_columns = [
            "PTS",
            "REB",
            "AST",
            "PLUS_MINUS",
            "FG_PCT",
            "FG3_PCT",
            "FT_PCT"
        ]

        for column in base_columns:
            mean_5 = f"{column}_ROLLING_MEAN_5"
            mean_10 = f"{column}_ROLLING_MEAN_10"
            mean_20 = f"{column}_ROLLING_MEAN_20"

            if mean_5 in self.df.columns and mean_10 in self.df.columns:
                self.df[f"{column}_MOMENTUM_5_10"] = (
                    self.df[mean_5] - self.df[mean_10]
                )

            if mean_5 in self.df.columns and mean_20 in self.df.columns:
                self.df[f"{column}_MOMENTUM_5_20"] = (
                    self.df[mean_5] - self.df[mean_20]
                )

        return self

    def create_efficiency_features(self):
        """
        Cria variáveis de eficiência.

        Essas variáveis ajudam a descrever a produtividade da equipe.

        Exemplos:
        - AST_TO_TOV: relação entre assistências e turnovers;
        - SHOOTING_EFFICIENCY: eficiência combinada dos arremessos;
        - REB_BALANCE: relação entre rebotes ofensivos e defensivos.
        """

        self.df["AST_TO_TOV"] = self.df["AST_LAG1"] / (
            self.df["TOV"].shift(1) + 1
        )

        self.df["SHOOTING_EFFICIENCY_LAG1"] = (
            self.df["FG_PCT_LAG1"] if "FG_PCT_LAG1" in self.df.columns else np.nan
        )

        if "OREB_LAG1" in self.df.columns and "DREB_LAG1" in self.df.columns:
            self.df["REB_BALANCE_LAG1"] = (
                self.df["OREB_LAG1"] / (self.df["DREB_LAG1"] + 1)
            )

        return self

    def create_streak_features(self):
        """
        Cria variáveis de sequência de vitórias e derrotas.

        WIN_STREAK:
            quantidade de vitórias consecutivas antes do jogo.

        LOSS_STREAK:
            quantidade de derrotas consecutivas antes do jogo.

        Para evitar vazamento temporal, usamos WL_SHIFTED.
        """

        def calculate_streaks(team_df: pd.DataFrame) -> pd.DataFrame:
            team_df = team_df.copy()

            team_df["WL_SHIFTED"] = team_df["WL"].shift(1)

            win_streak = []
            loss_streak = []

            current_wins = 0
            current_losses = 0

            for result in team_df["WL_SHIFTED"]:
                if result == "W":
                    current_wins += 1
                    current_losses = 0
                elif result == "L":
                    current_losses += 1
                    current_wins = 0
                else:
                    current_wins = 0
                    current_losses = 0

                win_streak.append(current_wins)
                loss_streak.append(current_losses)

            team_df["WIN_STREAK"] = win_streak
            team_df["LOSS_STREAK"] = loss_streak

            return team_df

        self.df = (
            self.df.groupby("TEAM_NAME", group_keys=False)
            .apply(calculate_streaks)
        )

        self.df.drop(columns=["WL_SHIFTED"], inplace=True)

        return self

    def handle_engineered_missing_values(self):
        """
        Trata valores ausentes gerados pelas janelas móveis.

        Como usamos rolling windows, os primeiros jogos de cada equipe
        naturalmente ficam sem médias anteriores suficientes.

        Estratégia:
        - preencher valores numéricos pela mediana da própria coluna;
        - isso mantém a base completa para os próximos passos.
        """

        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:
            self.df[column] = self.df[column].fillna(
                self.df[column].median()
            )

        return self

    def run(self) -> pd.DataFrame:
        """
        Executa todo o pipeline de engenharia de features.

        Ordem:
        1. mando de quadra;
        2. dias de descanso;
        3. variáveis defasadas;
        4. médias móveis;
        5. momentum;
        6. eficiência;
        7. sequências;
        8. tratamento de valores ausentes.
        """

        print("\n================================================")
        print("FEATURE ENGINEERING")
        print("================================================")

        self.df = self.df.sort_values(
            by=["TEAM_NAME", "GAME_DATE"]
        ).reset_index(drop=True)

        return (
            self.create_home_feature()
            .create_days_rest()
            .create_lag_features()
            .create_rolling_features()
            .create_momentum_features()
            .create_efficiency_features()
            .create_streak_features()
            .handle_engineered_missing_values()
            .df
        )