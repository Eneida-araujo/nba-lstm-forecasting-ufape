"""
=========================================================
pvalue_selection.py
=========================================================

Responsável pela seleção estatística de features usando p-value.

O p-value mede se uma variável possui relação estatisticamente
significativa com o alvo.

Critério adotado:
    p-value < 0.05

Importante:
Este módulo evita vazamento de dados removendo variáveis do jogo atual
que não poderiam ser conhecidas antes da partida.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler


class PValueSelector:
    """
    Seleciona features estatisticamente significativas usando OLS.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        significance_level: float = 0.05,
        output_dir: str | Path = "outputs/tables"
    ):
        self.df = dataframe.copy()
        self.target_column = target_column
        self.significance_level = significance_level
        self.output_dir = Path(output_dir)

        self.pvalues_df = None
        self.selected_features = []

    def get_forbidden_columns(self) -> list:
        """
        Define colunas que NÃO devem entrar como features.

        Motivo:
        algumas colunas representam informações do próprio jogo,
        ou seja, não estariam disponíveis antes da partida.
        """

        return [
            "SEASON_ID",
            "TEAM_ID",
            "GAME_ID",
            "VIDEO_AVAILABLE",

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
            "PLUS_MINUS",

            "W",
            "L",
            "W_PCT"
        ]

    def get_candidate_features(self) -> list:
        """
        Seleciona features numéricas candidatas.

        Mantém apenas variáveis numéricas e remove:
        - identificadores;
        - alvos;
        - estatísticas do jogo atual;
        - variáveis textuais.
        """

        numeric_columns = self.df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        forbidden_columns = self.get_forbidden_columns()

        candidate_features = [
            column
            for column in numeric_columns
            if column not in forbidden_columns
            and column != self.target_column
        ]

        return candidate_features

    def prepare_data(self):
        """
        Prepara X e y para regressão OLS.

        Etapas:
        - seleção das features candidatas;
        - remoção de valores infinitos;
        - preenchimento de ausentes;
        - remoção de colunas com variância zero;
        - padronização das features.
        """

        features = self.get_candidate_features()

        X = self.df[features].copy()
        y = self.df[self.target_column].copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))

        y = y.replace([np.inf, -np.inf], np.nan)
        y = y.fillna(y.median())

        # Remove colunas constantes, pois elas não ajudam o modelo
        # e podem prejudicar a regressão OLS.
        non_constant_columns = [
            column
            for column in X.columns
            if X[column].nunique() > 1
        ]

        X = X[non_constant_columns]

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        X_scaled = sm.add_constant(X_scaled)

        return X_scaled, y

    def calculate_pvalues(self) -> pd.DataFrame:
        """
        Calcula os p-values das features usando regressão OLS.

        Returns
        -------
        pd.DataFrame
            Tabela com features e seus respectivos p-values.
        """

        print("\n================================================")
        print(f"P-VALUE SELECTION — TARGET: {self.target_column}")
        print("================================================")

        X, y = self.prepare_data()

        model = sm.OLS(y, X).fit()

        pvalues = model.pvalues.drop("const", errors="ignore")

        self.pvalues_df = (
            pd.DataFrame({
                "feature": pvalues.index,
                "p_value": pvalues.values
            })
            .sort_values(by="p_value", ascending=True)
            .reset_index(drop=True)
        )

        self.selected_features = self.pvalues_df[
            self.pvalues_df["p_value"] < self.significance_level
        ]["feature"].tolist()

        print(f"[INFO] Features candidatas: {len(self.pvalues_df)}")
        print(
            f"[INFO] Features significativas "
            f"(p < {self.significance_level}): "
            f"{len(self.selected_features)}"
        )

        return self.pvalues_df

    def save_results(self):
        """
        Salva os resultados em CSV.

        Arquivos gerados:
        - tabela completa de p-values;
        - tabela apenas com features significativas.
        """

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        all_pvalues_path = (
            self.output_dir / f"pvalues_{self.target_column.lower()}.csv"
        )

        selected_path = (
            self.output_dir /
            f"selected_features_pvalue_{self.target_column.lower()}.csv"
        )

        self.pvalues_df.to_csv(
            all_pvalues_path,
            index=False
        )

        selected_df = pd.DataFrame({
            "feature": self.selected_features
        })

        selected_df.to_csv(
            selected_path,
            index=False
        )

        print(f"[INFO] P-values salvos em: {all_pvalues_path}")
        print(f"[INFO] Features selecionadas salvas em: {selected_path}")

    def run(self) -> list:
        """
        Executa o pipeline completo de seleção por p-value.
        """

        self.calculate_pvalues()
        self.save_results()

        return self.selected_features