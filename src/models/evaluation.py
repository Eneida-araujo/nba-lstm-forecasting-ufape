"""
=========================================================
evaluation.py
=========================================================

Responsável pela avaliação dos modelos.

Inclui:
- métricas de regressão;
- métricas de classificação;
- intervalos de confiança;
- probabilidades;
- interpretação executiva.
=========================================================
"""

import numpy as np
import scipy.stats as stats

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class ModelEvaluator:

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def calculate_mape(self, y_true, y_pred):
        y_true_safe = np.where(y_true == 0, 1, y_true)

        return np.mean(
            np.abs((y_true - y_pred) / y_true_safe)
        ) * 100

    def regression_metrics(self, y_true, y_pred):
        """
        Calcula métricas de regressão.
        A margem de erro foi calculada como z * RMSE,
        representando incerteza aproximada da predição.
        """

        mae = mean_absolute_error(y_true, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        mape = self.calculate_mape(y_true, y_pred)

        r2 = r2_score(y_true, y_pred)

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

        z_score = stats.norm.ppf(
            1 - (1 - self.confidence_level) / 2
        )

        confidence_interval = z_score * rmse

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "mean_error": mean_error,
            "mean_over_prediction": mean_over_prediction,
            "mean_under_prediction": mean_under_prediction,
            "confidence_interval": confidence_interval
        }

    def create_binary_labels(self, values, threshold):
        """
        Converte valores contínuos em classes binárias.
        """

        return np.where(values >= threshold, 1, 0)

    def classification_metrics(self, y_true, y_pred, threshold):
        """
        Calcula métricas classificatórias.
        O AUC utiliza y_pred contínuo, não apenas rótulos binários.
        """

        y_true_binary = self.create_binary_labels(
            y_true,
            threshold
        )

        y_pred_binary = self.create_binary_labels(
            y_pred,
            threshold
        )

        accuracy = accuracy_score(
            y_true_binary,
            y_pred_binary
        )

        f1 = f1_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0
        )

        try:
            auc = roc_auc_score(
                y_true_binary,
                y_pred
            )
        except Exception:
            auc = np.nan

        cm = confusion_matrix(
            y_true_binary,
            y_pred_binary
        )

        return {
            "accuracy": accuracy,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": cm
        }

    def probability_from_prediction(
        self,
        prediction,
        threshold,
        rmse_model
    ):
        """
        Calcula probabilidade baseada na distância entre previsão e threshold,
        usando o RMSE como medida de incerteza do modelo.
        """

        if rmse_model is None or rmse_model <= 0:
            raise ValueError(
                "rmse_model deve ser informado e maior que zero."
            )

        z_score = (prediction - threshold) / rmse_model

        probability = stats.norm.cdf(z_score)

        return probability

    def probability_label(self, probability):
        """
        Converte probabilidade em interpretação textual.
        """

        if probability >= 0.75:
            return "alta"

        elif probability >= 0.50:
            return "moderada"

        else:
            return "baixa"

    def create_executive_interpretation(
        self,
        target_name,
        prediction,
        threshold,
        rmse_model
    ):
        """
        Gera interpretação textual para gestores.
        """

        probability = self.probability_from_prediction(
            prediction=prediction,
            threshold=threshold,
            rmse_model=rmse_model
        )

        label = self.probability_label(probability)

        probability_percent = probability * 100

        text = (
            f"A previsão é que a equipe faça "
            f"{prediction:.1f} {target_name}, "
            f"com {probability_percent:.1f}% "
            f"de chance ({label}) de superar "
            f"{threshold:.2f}."
        )

        return {
            "prediction": prediction,
            "threshold": threshold,
            "probability": probability,
            "probability_percent": probability_percent,
            "label": label,
            "text": text
        }

    def full_evaluation(self, y_true, y_pred, threshold):
        """
        Executa avaliação completa.
        """

        regression = self.regression_metrics(
            y_true,
            y_pred
        )

        classification = self.classification_metrics(
            y_true,
            y_pred,
            threshold
        )

        return {
            **regression,
            **classification
        }