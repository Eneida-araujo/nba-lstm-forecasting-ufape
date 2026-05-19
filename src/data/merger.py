"""
=========================================================
merger.py
=========================================================

Arquivo responsável pela união das bases BaseA e BaseB.

Estratégia:
- BaseA será a base principal;
- BaseB fornecerá atributos complementares.

Merge realizado usando:
- GAME_ID
- TEAM_NAME

Objetivo:
- evitar duplicação;
- evitar inconsistências;
- preservar integridade temporal.

=========================================================
"""

import pandas as pd


class DataMerger:

    @staticmethod
    def merge_datasets(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Realiza merge entre BaseA e BaseB.

        Parameters
        ----------
        df_a : pd.DataFrame
            Base principal.

        df_b : pd.DataFrame
            Base complementar.

        Returns
        -------
        pd.DataFrame
            Base híbrida final.
        """

        print("\n================================================")
        print("MERGE DAS BASES")
        print("================================================")

        # =================================================
        # PADRONIZAÇÃO DOS NOMES DAS COLUNAS
        # =================================================

        df_a.columns = [
            col.upper().strip()
            for col in df_a.columns
        ]

        df_b.columns = [
            col.upper().strip()
            for col in df_b.columns
        ]

        # =================================================
        # COLUNAS ÚTEIS DA BASEB
        # =================================================
        #
        # Selecionamos apenas colunas relevantes
        # para evitar duplicações e ruído.
        #

        useful_columns_b = [
            "GAME_ID",
            "TEAM_NAME",
            "W",
            "L",
            "W_PCT"
        ]

        df_b_selected = df_b[useful_columns_b].copy()

        # =================================================
        # REMOÇÃO DE DUPLICATAS
        # =================================================

        df_b_selected.drop_duplicates(
            subset=["GAME_ID", "TEAM_NAME"],
            inplace=True
        )

        # =================================================
        # MERGE
        # =================================================
        #
        # LEFT JOIN:
        # preserva todas as linhas da BaseA.
        #

        merged_df = pd.merge(
            df_a,
            df_b_selected,
            on=["GAME_ID", "TEAM_NAME"],
            how="left"
        )

        print("[INFO] Merge concluído.")

        print(f"[INFO] Shape final: {merged_df.shape}")

        return merged_df