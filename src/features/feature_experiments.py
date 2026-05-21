"""
=========================================================
feature_experiments.py
=========================================================

Executa experimentos comparativos entre os 6 feature sets:

1. pvalue
2. random_forest
3. intersection
4. top_10_rf
5. top_20_rf
6. hybrid

Objetivo:
- comparar desempenho dos conjuntos de features;
- gerar ranking dos melhores conjuntos;
- usar validação temporal;
- preparar a escolha final das features para LSTM.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class FeatureExperimentRunner:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        tables_dir: str | Path = "outputs/tables",
        output_dir: str | Path = "outputs/metrics",
        random_state: int = 42
    ):
        self.df = dataframe.copy()
        self.target_column = target_column
        self.target_name = target_column.lower()
        self.tables_dir = Path(tables_dir)
        self.output_dir = Path(output_dir)
        self.random_state = random_state

        self.results = []

    def load_feature_sets(self) -> pd.DataFrame:
        """
        Carrega os feature sets criados anteriormente.
        """

        file_path = (
            self.tables_dir /
            f"feature_sets_{self.target_name}.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo de feature sets não encontrado: {file_path}"
            )

        return pd.read_csv(file_path)

    def get_features_by_set(
        self,
        feature_sets_df: pd.DataFrame,
        feature_set_name: str
    ) -> list:
        """
        Retorna as features de um conjunto específico.
        """

        features = (
            feature_sets_df[
                feature_sets_df["feature_set"] == feature_set_name
            ]["feature"]
            .dropna()
            .tolist()
        )

        available_features = [
            feature
            for feature in features
            if feature in self.df.columns
        ]

        return available_features

    def prepare_data(self, features: list):
        """
        Prepara X e y para o experimento.

        Mantém a ordem temporal dos dados.
        """

        X = self.df[features].copy()
        y = self.df[self.target_column].copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))

        y = y.replace([np.inf, -np.inf], np.nan)
        y = y.fillna(y.median())

        non_constant_columns = [
            column
            for column in X.columns
            if X[column].nunique() > 1
        ]

        X = X[non_constant_columns]

        return X, y

    def temporal_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.20
    ):
        """
        Divide os dados respeitando a ordem temporal.

        80% inicial: treino
        20% final: teste
        """

        split_index = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        return X_train, X_test, y_train, y_test

    def calculate_mape(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> float:
        """
        Calcula MAPE evitando divisão por zero.
        """

        y_true_safe = np.where(y_true == 0, 1, y_true)

        return np.mean(
            np.abs((y_true - y_pred) / y_true_safe)
        ) * 100

    def evaluate_feature_set(
        self,
        feature_set_name: str,
        features: list
    ):
        """
        Treina e avalia um Random Forest para um feature set.

        Este Random Forest é usado como baseline experimental,
        não como modelo final do projeto.
        """

        if len(features) == 0:
            print(
                f"[WARNING] Feature set vazio ignorado: "
                f"{feature_set_name}"
            )
            return

        X, y = self.prepare_data(features)

        if X.shape[1] == 0:
            print(
                f"[WARNING] Feature set sem colunas válidas: "
                f"{feature_set_name}"
            )
            return

        X_train, X_test, y_train, y_test = self.temporal_train_test_split(
            X,
            y
        )

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mape = self.calculate_mape(y_test, predictions)

        errors = predictions - y_test.values

        mean_error = np.mean(errors)
        mean_over_prediction = np.mean(errors[errors > 0]) if np.any(errors > 0) else 0
        mean_under_prediction = np.mean(errors[errors < 0]) if np.any(errors < 0) else 0

        self.results.append({
            "target": self.target_column,
            "feature_set": feature_set_name,
            "n_features": X.shape[1],
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "mean_error": mean_error,
            "mean_over_prediction": mean_over_prediction,
            "mean_under_prediction": mean_under_prediction
        })

        print(
            f"[INFO] {self.target_column} | {feature_set_name} | "
            f"features={X.shape[1]} | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"MAPE={mape:.2f}% | "
            f"R²={r2:.4f}"
        )

    def run(self) -> pd.DataFrame:
        """
        Executa todos os experimentos para o target.
        """

        print("\n================================================")
        print(f"FEATURE EXPERIMENTS — TARGET: {self.target_column}")
        print("================================================")

        feature_sets_df = self.load_feature_sets()

        feature_set_names = [
            "pvalue",
            "random_forest",
            "intersection",
            "top_10_rf",
            "top_20_rf",
            "hybrid"
        ]

        for feature_set_name in feature_set_names:

            features = self.get_features_by_set(
                feature_sets_df,
                feature_set_name
            )

            self.evaluate_feature_set(
                feature_set_name,
                features
            )

        results_df = pd.DataFrame(self.results)

        results_df = results_df.sort_values(
            by=["rmse", "mae"],
            ascending=True
        ).reset_index(drop=True)

        self.save_results(results_df)

        return results_df

    def save_results(self, results_df: pd.DataFrame):
        """
        Salva os resultados dos experimentos em CSV.
        """

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            self.output_dir /
            f"feature_experiments_{self.target_name}.csv"
        )

        results_df.to_csv(
            output_path,
            index=False
        )

        print(f"[INFO] Resultados salvos em: {output_path}")