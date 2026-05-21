"""
=========================================================
windowing.py
=========================================================

Responsável por criar Sliding Windows temporais
para treinamento da LSTM.

Janelas utilizadas:
- 5
- 10
- 15
- 20

O objetivo é transformar dados tabulares em
sequências temporais.
=========================================================
"""

import numpy as np
import pandas as pd


class SlidingWindowGenerator:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list,
        target_column: str,
        window_size: int
    ):
        self.df = dataframe.copy()

        self.feature_columns = feature_columns
        self.target_column = target_column
        self.window_size = window_size

    def create_windows(self):
        """
        Cria as sequências temporais para LSTM.

        Retorna:
        - X -> sequências temporais
        - y -> targets
        """

        X = []
        y = []

        data = self.df[self.feature_columns].values
        target = self.df[self.target_column].values

        total_samples = len(self.df)

        for i in range(self.window_size, total_samples):

            X_window = data[
                i - self.window_size:i
            ]

            y_target = target[i]

            X.append(X_window)
            y.append(y_target)

        X = np.array(X)
        y = np.array(y)

        return X, y

    def run(self):
        """
        Executa criação das Sliding Windows.
        """

        print("\n================================================")
        print("SLIDING WINDOWS")
        print("================================================")

        print(f"[INFO] Window size: {self.window_size}")

        X, y = self.create_windows()

        print(f"[INFO] X shape: {X.shape}")
        print(f"[INFO] y shape: {y.shape}")

        return X, y