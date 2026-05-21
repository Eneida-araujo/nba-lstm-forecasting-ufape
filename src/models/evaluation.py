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
import pandas as pd

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

    def __init__(
        self,
        confidence_level: float = 0.95
    ):
        self.confidence_level = confidence_level

    # =====================================================
    # REGRESSION METRICS
    # =====================================================

    def calculate_mape(
        self,
        y_true,
        y_pred
    ):

        y_true_safe = np.where(y_true == 0, 1, y_true)

        return np.mean(
            np.abs((y_true - y_pred) / y_true_safe)
        ) * 100

    def regression_metrics(
        self,
        y_true,
        y_pred
    ):
        """
        Calcula métricas de regressão.
        """

        mae = mean_absolute_error(y_true, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        mape = self.calculate_mape(
            y_true,
            y_pred
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

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

        confidence_interval = 1.96 * np.std(errors)

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

    # =====================================================
    # CLASSIFICATION METRICS
    # =====================================================

    def create_binary_labels(
        self,
        values,
        threshold
    ):
        """
        Converte valores contínuos em classes binárias.

        Exemplo:
        pontos > 100 = 1
        pontos <= 100 = 0
        """

        return np.where(values >= threshold, 1, 0)

    def classification_metrics(
        self,
        y_true,
        y_pred,
        threshold
    ):
        """
        Calcula métricas classificatórias.
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
            y_pred_binary
        )

        try:

            auc = roc_auc_score(
                y_true_binary,
                y_pred_binary
            )

        except:

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

    # =====================================================
    # PROBABILITY INTERPRETATION
    # =====================================================

    def probability_from_prediction(
        self,
        prediction,
        threshold
    ):
        """
        Calcula probabilidade simples baseada
        na distância para o threshold.
        """

        difference = prediction - threshold

        probability = 1 / (
            1 + np.exp(-difference / 10)
        )

        return probability

    def probability_label(
        self,
        probability
    ):
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
        threshold
    ):
        """
        Gera interpretação textual para gestores.
        """

        probability = self.probability_from_prediction(
            prediction,
            threshold
        )

        label = self.probability_label(
            probability
        )

        probability_percent = probability * 100

        text = (
            f"A previsão é que a equipe faça "
            f"{prediction:.1f} {target_name}, "
            f"com {probability_percent:.1f}% "
            f"de chance ({label}) de superar "
            f"{threshold}."
        )

        return {
            "prediction": prediction,
            "threshold": threshold,
            "probability": probability,
            "probability_percent": probability_percent,
            "label": label,
            "text": text
        }

    # =====================================================
    # FULL EVALUATION
    # =====================================================

    def full_evaluation(
        self,
        y_true,
        y_pred,
        threshold
    ):
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