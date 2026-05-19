"""
=========================================================
loader.py
=========================================================

Responsável pelo carregamento das bases CSV.

Objetivos:
- padronizar carregamento;
- evitar código repetido;
- centralizar logs de leitura.

=========================================================
"""

import pandas as pd


class DataLoader:
    """
    Classe responsável pelo carregamento de datasets.
    """

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        """
        Carrega um arquivo CSV.

        Parameters
        ----------
        path : str
            Caminho do arquivo CSV.

        Returns
        -------
        pd.DataFrame
            Dataset carregado.
        """

        print("\n================================================")
        print("CARREGAMENTO DO DATASET")
        print("================================================")

        df = pd.read_csv(path)

        print(f"[INFO] Arquivo carregado: {path}")

        print(f"[INFO] Shape do dataset: {df.shape}")

        return df