"""
=========================================================
predictor.py
=========================================================

Responsável por gerar previsões finais e interpretações
executivas para RF1, RF2 e RF3.

RF1:
Prever se a equipe fará PTS, REB e AST acima da própria
média da temporada.

RF2:
Prever se a equipe fará acima de:
- 100 pontos
- 30 rebotes
- 20 assistências

RF3:
Prever quantos pontos, rebotes e assistências a equipe fará.

Este módulo usa:
- modelos LSTM treinados;
- feature set Hybrid;
- Sliding Window vencedora pelo menor RMSE;
- ModelEvaluator.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from src.models.evaluation import ModelEvaluator


class NBAPredictor:
    """
    Classe responsável por gerar previsões finais para uma equipe.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        team_name: str,
        feature_set_name: str = "hybrid",
        tables_dir: str | Path = "outputs/tables",
        models_dir: str | Path = "outputs/models",
        metrics_dir: str | Path = "outputs/metrics"
    ):
        self.df = dataframe.copy()
        self.team_name = team_name
        self.feature_set_name = feature_set_name

        self.tables_dir = Path(tables_dir)
        self.models_dir = Path(models_dir)
        self.metrics_dir = Path(metrics_dir)

        self.evaluator = ModelEvaluator()

        self.best_model_info = self.load_best_model_info()

        self.fixed_thresholds = {
            "PTS": 100,
            "REB": 30,
            "AST": 20
        }

    def load_best_model_info(self) -> dict:
        """
        Carrega automaticamente a melhor janela temporal e o RMSE
        de cada target, com base no menor RMSE registrado em:
        outputs/metrics/lstm_results.csv.
        """

        results_path = self.metrics_dir / "lstm_results.csv"

        if not results_path.exists():
            raise FileNotFoundError(
                f"Arquivo de resultados LSTM não encontrado: {results_path}"
            )

        results_df = pd.read_csv(results_path)

        required_columns = {"target", "window_size", "rmse"}

        if not required_columns.issubset(results_df.columns):
            raise ValueError(
                "O arquivo lstm_results.csv precisa conter as colunas: "
                "target, window_size e rmse."
            )

        best_model_info = {}

        for target in ["PTS", "REB", "AST"]:

            target_df = results_df[
                results_df["target"] == target
            ].copy()

            if target_df.empty:
                raise ValueError(
                    f"Nenhum resultado encontrado para o target {target}."
                )

            best_row = target_df.loc[
                target_df["rmse"].idxmin()
            ]

            best_model_info[target] = {
                "window_size": int(best_row["window_size"]),
                "rmse": float(best_row["rmse"])
            }

        print("[INFO] Melhores modelos carregados automaticamente:")
        print(best_model_info)

        return best_model_info

    def get_team_data(self) -> pd.DataFrame:
        """
        Filtra os dados da equipe escolhida e mantém ordem temporal.
        """

        team_df = self.df[
            self.df["TEAM_NAME"] == self.team_name
        ].copy()

        if team_df.empty:
            raise ValueError(
                f"Equipe não encontrada no dataset: {self.team_name}"
            )

        team_df = team_df.sort_values(
            by="GAME_DATE"
        ).reset_index(drop=True)

        return team_df

    def load_features_for_target(
        self,
        target: str
    ) -> list:
        """
        Carrega as features do feature set escolhido.
        """

        file_path = (
            self.tables_dir /
            f"feature_sets_{target.lower()}.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {file_path}"
            )

        feature_sets_df = pd.read_csv(file_path)

        features = (
            feature_sets_df[
                feature_sets_df["feature_set"] == self.feature_set_name
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

    def load_model(
        self,
        target: str,
        window_size: int
    ):
        """
        Carrega o modelo LSTM treinado.
        """

        model_path = (
            self.models_dir /
            f"lstm_{target.lower()}_window_{window_size}.keras"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}"
            )

        model = tf.keras.models.load_model(model_path)

        return model

    def prepare_prediction_window(
        self,
        team_df: pd.DataFrame,
        features: list,
        target: str,
        window_size: int
    ):
        """
        Prepara a última janela temporal real da equipe.

        Essa janela será usada para prever a próxima partida ainda
        não presente na base.
        """

        if len(team_df) < window_size:
            raise ValueError(
                f"A equipe {self.team_name} possui menos jogos "
                f"do que a janela exigida ({window_size})."
            )

        data = team_df[features + [target]].copy()

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.median(numeric_only=True))

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        scaled_features = feature_scaler.fit_transform(
            data[features]
        )

        target_scaler.fit(
            data[[target]]
        )

        last_window = scaled_features[-window_size:]

        X_next = np.array([last_window])

        return X_next, target_scaler

    def predict_target(
        self,
        target: str
    ) -> dict:
        """
        Gera previsão para um único target.
        """

        team_df = self.get_team_data()

        window_size = self.best_model_info[target]["window_size"]
        rmse_model = self.best_model_info[target]["rmse"]

        features = self.load_features_for_target(target)

        model = self.load_model(
            target=target,
            window_size=window_size
        )

        X_next, target_scaler = self.prepare_prediction_window(
            team_df=team_df,
            features=features,
            target=target,
            window_size=window_size
        )

        scaled_prediction = model.predict(
            X_next,
            verbose=0
        )

        prediction = target_scaler.inverse_transform(
            scaled_prediction.reshape(-1, 1)
        ).flatten()[0]

        team_average = team_df[target].mean()

        fixed_threshold = self.fixed_thresholds[target]

        above_average_result = (
            self.evaluator.create_executive_interpretation(
                target_name=target,
                prediction=prediction,
                threshold=team_average,
                rmse_model=rmse_model
            )
        )

        fixed_threshold_result = (
            self.evaluator.create_executive_interpretation(
                target_name=target,
                prediction=prediction,
                threshold=fixed_threshold,
                rmse_model=rmse_model
            )
        )

        return {
            "target": target,
            "window_size": window_size,
            "rmse_model": rmse_model,
            "prediction": prediction,
            "team_average": team_average,
            "fixed_threshold": fixed_threshold,
            "above_average": above_average_result,
            "fixed_threshold_result": fixed_threshold_result
        }

    def predict_all_targets(self) -> dict:
        """
        Gera previsões para PTS, REB e AST.
        """

        results = {}

        for target in ["PTS", "REB", "AST"]:

            results[target] = self.predict_target(
                target=target
            )

        return results

    def create_report_rows(
        self,
        results: dict
    ) -> pd.DataFrame:
        """
        Organiza os resultados em formato tabular.
        """

        rows = []

        for target, result in results.items():

            rows.append({
                "team_name": self.team_name,
                "target": target,
                "window_size": result["window_size"],
                "rmse_model": result["rmse_model"],
                "prediction": result["prediction"],
                "team_average": result["team_average"],
                "fixed_threshold": result["fixed_threshold"],
                "probability_above_average_percent": result[
                    "above_average"
                ]["probability_percent"],
                "label_above_average": result[
                    "above_average"
                ]["label"],
                "probability_fixed_threshold_percent": result[
                    "fixed_threshold_result"
                ]["probability_percent"],
                "label_fixed_threshold": result[
                    "fixed_threshold_result"
                ]["label"]
            })

        return pd.DataFrame(rows)

    def print_executive_summary(
        self,
        results: dict
    ):
        """
        Exibe interpretação simples para gestores.
        """

        print("\n================================================")
        print(f"PREVISÃO EXECUTIVA — {self.team_name}")
        print("================================================")

        print("\nRF3 — Previsão numérica da próxima partida:")

        print(
            f"- Pontos previstos: "
            f"{results['PTS']['prediction']:.1f}"
        )

        print(
            f"- Rebotes previstos: "
            f"{results['REB']['prediction']:.1f}"
        )

        print(
            f"- Assistências previstas: "
            f"{results['AST']['prediction']:.1f}"
        )

        print("\nRF1 — Chance de superar a própria média:")

        for target, result in results.items():
            print(
                "- "
                + result["above_average"]["text"]
            )

        print("\nRF2 — Chance de superar metas fixas:")

        for target, result in results.items():
            print(
                "- "
                + result["fixed_threshold_result"]["text"]
            )

    def run(self) -> pd.DataFrame:
        """
        Executa previsão completa da equipe.
        """

        results = self.predict_all_targets()

        self.print_executive_summary(
            results=results
        )

        report_df = self.create_report_rows(
            results=results
        )

        return report_df