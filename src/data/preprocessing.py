"""
=========================================================
preprocessing.py
=========================================================

Responsável pela limpeza e preparação inicial dos dados.

Etapas:
- padronização;
- conversão de datas;
- remoção de duplicatas;
- ordenação temporal;
- tratamento de valores ausentes.

=========================================================
"""

import pandas as pd
import numpy as np


class DataPreprocessor:

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe.copy()

    def standardize_columns(self):
        """
        Padroniza nomes das colunas.
        """

        self.df.columns = [
            col.upper().strip()
            for col in self.df.columns
        ]

        return self

    def convert_dates(self):
        """
        Converte GAME_DATE para datetime.

        Algumas bases possuem formatos mistos,
        então usamos format='mixed'.
        """

        if "GAME_DATE" in self.df.columns:

            self.df["GAME_DATE"] = pd.to_datetime(
                self.df["GAME_DATE"],
                errors="coerce",
                format="mixed"
            )

        return self

    def remove_invalid_dates(self):
        """
        Remove datas inválidas.
        """

        if "GAME_DATE" in self.df.columns:

            invalid_dates = (
                self.df["GAME_DATE"]
                .isnull()
                .sum()
            )

            if invalid_dates > 0:

                print(
                    f"[WARNING] Datas inválidas removidas: "
                    f"{invalid_dates}"
                )

            self.df = self.df.dropna(
                subset=["GAME_DATE"]
            )

        return self

    def sort_data(self):
        """
        Ordena os dados temporalmente.

        IMPORTANTE:
        Modelos temporais exigem ordem cronológica.
        """

        if (
            "TEAM_NAME" in self.df.columns
            and "GAME_DATE" in self.df.columns
        ):

            self.df.sort_values(
                by=["TEAM_NAME", "GAME_DATE"],
                inplace=True
            )

        return self

    def remove_duplicates(self):
        """
        Remove linhas duplicadas.
        """

        self.df.drop_duplicates(inplace=True)

        return self

    def handle_missing_values(self):
        """
        Trata valores ausentes numéricos.

        Estratégia:
        substituir pela mediana.
        """

        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:

            self.df[column] = self.df[column].fillna(
                self.df[column].median()
            )

        return self

    def reset_index(self):
        """
        Reorganiza índices.
        """

        self.df.reset_index(
            drop=True,
            inplace=True
        )

        return self

    def preprocess(self):
        """
        Executa pipeline completo de preprocessamento.
        """

        print("\n================================================")
        print("PREPROCESSAMENTO")
        print("================================================")

        return (
            self.standardize_columns()
            .convert_dates()
            .remove_invalid_dates()
            .sort_data()
            .remove_duplicates()
            .handle_missing_values()
            .reset_index()
            .df
        )