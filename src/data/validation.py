"""
=========================================================
validation.py
=========================================================

Responsável pela validação inicial do dataset.

Objetivo:
- verificar inconsistências;
- verificar valores ausentes;
- verificar tipos;
- entender estrutura do dataset.

=========================================================
"""

import pandas as pd


class DataValidator:

    @staticmethod
    def validate_dataframe(df: pd.DataFrame):
        """
        Exibe informações importantes do dataset.
        """

        print("\n================================================")
        print("VALIDAÇÃO DOS DADOS")
        print("================================================")

        # =================================================
        # SHAPE
        # =================================================

        print("\n[INFO] Shape do dataset:")

        print(df.shape)

        # =================================================
        # VALORES AUSENTES
        # =================================================

        print("\n[INFO] Valores ausentes:")

        print(df.isnull().sum())

        # =================================================
        # TIPOS
        # =================================================

        print("\n[INFO] Tipos de dados:")

        print(df.dtypes)

        # =================================================
        # PRIMEIRAS LINHAS
        # =================================================

        print("\n[INFO] Head do dataset:")

        print(df.head())