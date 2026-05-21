"""
=========================================================
correlation_filter.py
=========================================================

Responsável por remover features altamente correlacionadas.

Objetivo:
- reduzir redundância;
- diminuir multicolinearidade;
- melhorar estabilidade do p-value;
- simplificar o conjunto de variáveis;
- melhorar interpretabilidade do modelo.

Critério utilizado:
    correlação absoluta > 0.90

Quando duas variáveis possuem correlação muito alta,
elas carregam informação muito parecida. Manter ambas pode
prejudicar modelos estatísticos como OLS, usado no p-value.
=========================================================
"""

from pathlib import Path

import pandas as pd
import numpy as np


class CorrelationFilter:
    """
    Classe responsável pela filtragem de features correlacionadas.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        threshold: float = 0.90,
        output_path: str | Path | None = None
    ):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset com as features candidatas.

        threshold : float
            Limite de correlação para remoção.
            Exemplo: 0.90 remove variáveis com correlação absoluta acima de 90%.

        output_path : str | Path | None
            Caminho opcional para salvar a lista de features removidas.
        """

        self.df = dataframe.copy()
        self.threshold = threshold
        self.output_path = Path(output_path) if output_path else None
        self.removed_features = []

    def get_numeric_features(self) -> pd.DataFrame:
        """
        Seleciona apenas colunas numéricas.

        A correlação só pode ser calculada entre variáveis numéricas.
        Colunas textuais como TEAM_NAME, MATCHUP e WL são ignoradas.
        """

        numeric_df = self.df.select_dtypes(
            include=[np.number]
        )

        return numeric_df

    def identify_correlated_features(self) -> list:
        """
        Identifica features altamente correlacionadas.

        Estratégia:
        1. Calcula matriz de correlação absoluta.
        2. Analisa apenas o triângulo superior da matriz.
        3. Remove uma das variáveis de cada par altamente correlacionado.

        Returns
        -------
        list
            Lista de features recomendadas para remoção.
        """

        numeric_df = self.get_numeric_features()

        correlation_matrix = numeric_df.corr().abs()

        upper_triangle = correlation_matrix.where(
            np.triu(
                np.ones(correlation_matrix.shape),
                k=1
            ).astype(bool)
        )

        features_to_remove = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > self.threshold)
        ]

        self.removed_features = features_to_remove

        return features_to_remove

    def apply_filter(self) -> pd.DataFrame:
        """
        Aplica o filtro de correlação ao dataframe.

        Remove apenas as colunas identificadas como altamente correlacionadas.
        """

        print("\n================================================")
        print("CORRELATION FILTER")
        print("================================================")

        features_to_remove = self.identify_correlated_features()

        print(f"[INFO] Threshold de correlação: {self.threshold}")
        print(f"[INFO] Features removidas: {len(features_to_remove)}")

        filtered_df = self.df.drop(
            columns=features_to_remove,
            errors="ignore"
        )

        print(f"[INFO] Shape antes do filtro: {self.df.shape}")
        print(f"[INFO] Shape depois do filtro: {filtered_df.shape}")

        if self.output_path:
            self.save_removed_features()

        return filtered_df

    def save_removed_features(self):
        """
        Salva a lista de features removidas em CSV.

        Isso ajuda na transparência metodológica do projeto
        e poderá ser usado no relatório final.
        """

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        removed_df = pd.DataFrame({
            "removed_feature": self.removed_features
        })

        removed_df.to_csv(
            self.output_path,
            index=False
        )

        print(
            f"[INFO] Lista de features removidas salva em: "
            f"{self.output_path}"
        )