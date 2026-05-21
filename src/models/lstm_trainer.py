"""
=========================================================
lstm_trainer.py
=========================================================

Responsável por treinar e avaliar modelos LSTM.

Executa experimentos com:
- targets: PTS, REB, AST
- sliding windows: 5, 10, 15, 20
- feature set: Hybrid

Gera métricas finais para comparação:
- MAE
- RMSE
- MAPE
- R²
- erro médio
- erro para cima
- erro para baixo
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.models.windowing import SlidingWindowGenerator
from src.models.lstm_model import LSTMModelBuilder


class LSTMTrainer:
    """
    Classe responsável por treinar modelos LSTM com diferentes janelas temporais.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        targets: list[str],
        windows: list[int],
        feature_set_name: str = "hybrid",
        tables_dir: str | Path = "outputs/tables",
        output_dir: str | Path = "outputs/metrics",
        models_dir: str | Path = "outputs/models"
    ):
        self.df = dataframe.copy()
        self.targets = targets
        self.windows = windows
        self.feature_set_name = feature_set_name

        self.tables_dir = Path(tables_dir)
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)

        self.results = []

    def load_features_for_target(self, target: str) -> list:
        """
        Carrega as features do conjunto escolhido para um target.

        Exemplo:
        outputs/tables/feature_sets_pts.csv
        """

        file_path = self.tables_dir / f"feature_sets_{target.lower()}.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo de feature set não encontrado: {file_path}"
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

    def scale_features_and_target(
        self,
        features: list,
        target: str
    ):
        """
        Normaliza features e target usando MinMaxScaler.

        A LSTM geralmente treina melhor quando os valores estão
        em escala semelhante.
        """

        selected_columns = features + [target]

        data = self.df[selected_columns].copy()

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.median(numeric_only=True))

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        scaled_features = feature_scaler.fit_transform(data[features])
        scaled_target = target_scaler.fit_transform(data[[target]])

        scaled_df = pd.DataFrame(
            scaled_features,
            columns=features,
            index=self.df.index
        )

        scaled_df[target] = scaled_target.flatten()

        return scaled_df, feature_scaler, target_scaler

    def temporal_train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.70,
        val_size: float = 0.15
    ):
        """
        Divide os dados temporalmente.

        70% treino
        15% validação
        15% teste

        Não há embaralhamento, pois se trata de série temporal.
        """

        total_samples = len(X)

        train_end = int(total_samples * train_size)
        val_end = int(total_samples * (train_size + val_size))

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]

        X_test = X[val_end:]
        y_test = y[val_end:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def calculate_mape(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calcula MAPE evitando divisão por zero.
        """

        y_true_safe = np.where(y_true == 0, 1, y_true)

        return np.mean(
            np.abs((y_true - y_pred) / y_true_safe)
        ) * 100

    def calculate_error_direction(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """
        Calcula erro médio, erro para cima e erro para baixo.

        Erro positivo:
        modelo previu acima do real.

        Erro negativo:
        modelo previu abaixo do real.
        """

        errors = y_pred - y_true

        mean_error = np.mean(errors)

        mean_over_prediction = (
            np.mean(errors[errors > 0])
            if np.any(errors > 0)
            else 0
        )

        mean_under_prediction = (
            np.mean(errors[errors < 0])
            if np.any(errors < 0)
            else 0
        )

        return mean_error, mean_over_prediction, mean_under_prediction

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> dict:
        """
        Calcula métricas de regressão.
        """

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = self.calculate_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        (
            mean_error,
            mean_over_prediction,
            mean_under_prediction
        ) = self.calculate_error_direction(y_true, y_pred)

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "mean_error": mean_error,
            "mean_over_prediction": mean_over_prediction,
            "mean_under_prediction": mean_under_prediction
        }

    def train_single_experiment(
        self,
        target: str,
        window_size: int
    ):
        """
        Treina uma LSTM para um target e uma janela temporal.
        """

        print("\n================================================")
        print(f"LSTM TRAINING — TARGET: {target} | WINDOW: {window_size}")
        print("================================================")

        features = self.load_features_for_target(target)

        print(f"[INFO] Feature set: {self.feature_set_name}")
        print(f"[INFO] Features utilizadas: {len(features)}")

        scaled_df, feature_scaler, target_scaler = (
            self.scale_features_and_target(
                features=features,
                target=target
            )
        )

        window_generator = SlidingWindowGenerator(
            dataframe=scaled_df,
            feature_columns=features,
            target_column=target,
            window_size=window_size
        )

        X, y = window_generator.run()

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        ) = self.temporal_train_val_test_split(X, y)

        input_shape = (
            X_train.shape[1],
            X_train.shape[2]
        )

        lstm_builder = LSTMModelBuilder(
            input_shape=input_shape,
            learning_rate=0.001,
            clipnorm=1.0,
            l2_value=0.001
        )

        model = lstm_builder.build_model()

        history = lstm_builder.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=80,
            batch_size=32,
            verbose=0
        )

        scaled_predictions = lstm_builder.predict(X_test)

        predictions = target_scaler.inverse_transform(
            scaled_predictions.reshape(-1, 1)
        ).flatten()

        y_test_original = target_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        metrics = self.evaluate_predictions(
            y_true=y_test_original,
            y_pred=predictions
        )

        self.save_predictions(
            target=target,
            window_size=window_size,
            y_true=y_test_original,
            y_pred=predictions
        )
        result = {
            "target": target,
            "window_size": window_size,
            "feature_set": self.feature_set_name,
            "n_features": len(features),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "epochs_trained": len(history.history["loss"]),
            **metrics
        }

        self.results.append(result)

        print(
            f"[INFO] {target} | window={window_size} | "
            f"MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"MAPE={metrics['mape']:.2f}% | "
            f"R²={metrics['r2']:.4f}"
        )

        self.save_model(
            model=model,
            target=target,
            window_size=window_size
        )

    def save_model(
        self,
        model,
        target: str,
        window_size: int
    ):
        """
        Salva o modelo treinado.
        """

        self.models_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            self.models_dir /
            f"lstm_{target.lower()}_window_{window_size}.keras"
        )

        model.save(model_path)

        print(f"[INFO] Modelo salvo em: {model_path}")

    def save_results(self):
        """
        Salva os resultados finais da LSTM.
        """

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        results_df = pd.DataFrame(self.results)

        output_path = self.output_dir / "lstm_results.csv"

        results_df.to_csv(
            output_path,
            index=False
        )

        print(f"[INFO] Resultados LSTM salvos em: {output_path}")

        return results_df

    def run(self):
        """
        Executa todos os experimentos LSTM.
        """

        for target in self.targets:
            for window_size in self.windows:
                self.train_single_experiment(
                    target=target,
                    window_size=window_size
                )

        results_df = self.save_results()

        return results_df
    
    def save_predictions(
        self,
        target: str,
        window_size: int,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """
        Salva valores reais e previstos para geração de gráficos.
        """
        
        predictions_dir = Path("outputs/predictions")
        predictions_dir.mkdir(parents=True, exist_ok=True)

        predictions_df = pd.DataFrame({
            "target": target,
            "window_size": window_size,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": y_pred - y_true
        })

        output_path = (
            predictions_dir /
            f"lstm_predictions_{target.lower()}_window_{window_size}.csv"
        )

        predictions_df.to_csv(output_path, index=False)

        print(f"[INFO] Previsões salvas em: {output_path}")