"""
=========================================================
plots.py
=========================================================

Geração dos gráficos solicitados no RF6.

Inclui:
- valores reais vs previstos;
- histograma dos erros;
- barra de confiança da previsão;
- matriz de confusão colorida.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


class PlotGenerator:

    def __init__(
        self,
        predictions_dir: str | Path = "outputs/predictions",
        metrics_dir: str | Path = "outputs/metrics",
        output_dir: str | Path = "outputs/figures"
    ):
        self.predictions_dir = Path(predictions_dir)
        self.metrics_dir = Path(metrics_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_predictions(
        self,
        target: str,
        window_size: int
    ) -> pd.DataFrame:

        file_path = (
            self.predictions_dir /
            f"lstm_predictions_{target.lower()}_window_{window_size}.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo de previsões não encontrado: {file_path}"
            )

        return pd.read_csv(file_path)

    def plot_real_vs_predicted(
        self,
        target: str,
        window_size: int
    ):
        """
        Gráfico de valores reais vs previstos.
        """

        df = self.load_predictions(target, window_size)

        plt.figure(figsize=(12, 6))

        plt.plot(df["y_true"].values, label="Real")
        plt.plot(df["y_pred"].values, label="Previsto")

        plt.title(f"{target} — Real vs Previsto | Window {window_size}")
        plt.xlabel("Amostras de teste")
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()

        output_path = (
            self.output_dir /
            f"real_vs_predicted_{target.lower()}_window_{window_size}.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"[INFO] Gráfico salvo em: {output_path}")

    def plot_error_histogram(
        self,
        target: str,
        window_size: int
    ):
        """
        Histograma dos erros.
        """

        df = self.load_predictions(target, window_size)

        plt.figure(figsize=(10, 6))

        plt.hist(df["error"], bins=30)

        plt.title(f"{target} — Histograma dos Erros | Window {window_size}")
        plt.xlabel("Erro")
        plt.ylabel("Frequência")
        plt.tight_layout()

        output_path = (
            self.output_dir /
            f"error_histogram_{target.lower()}_window_{window_size}.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"[INFO] Gráfico salvo em: {output_path}")

    def plot_confidence_bar(
        self,
        target: str,
        window_size: int
    ):
        """
        Barra de confiança da previsão.

        Usa a última previsão do conjunto de teste.
        """

        df = self.load_predictions(target, window_size)

        last_prediction = df["y_pred"].iloc[-1]
        confidence_margin = 1.96 * df["error"].std()

        plt.figure(figsize=(8, 6))

        plt.bar(
            [f"Previsão {target}"],
            [last_prediction],
            yerr=[confidence_margin],
            capsize=10
        )

        plt.title(
            f"{target} — Previsão: "
            f"{last_prediction:.1f} ± {confidence_margin:.1f}"
        )

        plt.ylabel(target)
        plt.tight_layout()

        output_path = (
            self.output_dir /
            f"confidence_bar_{target.lower()}_window_{window_size}.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"[INFO] Gráfico salvo em: {output_path}")

    def plot_confusion_matrix(
        self,
        target: str,
        window_size: int,
        threshold: float
    ):
        """
        Matriz de confusão colorida para classificação.

        Exemplo:
        PTS >= 100
        REB >= 30
        AST >= 20
        """

        df = self.load_predictions(target, window_size)

        y_true_binary = np.where(
            df["y_true"] >= threshold,
            1,
            0
        )

        y_pred_binary = np.where(
            df["y_pred"] >= threshold,
            1,
            0
        )

        cm = confusion_matrix(
            y_true_binary,
            y_pred_binary
        )

        plt.figure(figsize=(6, 5))

        plt.imshow(cm)
        plt.title(
            f"{target} — Matriz de Confusão | Threshold {threshold}"
        )

        plt.xlabel("Previsto")
        plt.ylabel("Real")

        plt.xticks([0, 1], ["Não", "Sim"])
        plt.yticks([0, 1], ["Não", "Sim"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        plt.colorbar()
        plt.tight_layout()

        output_path = (
            self.output_dir /
            f"confusion_matrix_{target.lower()}_window_{window_size}.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"[INFO] Gráfico salvo em: {output_path}")

    def run_best_models(self):
        """
        Gera gráficos dos melhores modelos encontrados.
        """

        best_models = {
            "PTS": {
                "window": 5,
                "threshold": 100
            },
            "REB": {
                "window": 20,
                "threshold": 30
            },
            "AST": {
                "window": 10,
                "threshold": 20
            }
        }

        for target, config in best_models.items():

            window_size = config["window"]
            threshold = config["threshold"]

            self.plot_real_vs_predicted(
                target=target,
                window_size=window_size
            )

            self.plot_error_histogram(
                target=target,
                window_size=window_size
            )

            self.plot_confidence_bar(
                target=target,
                window_size=window_size
            )

            self.plot_confusion_matrix(
                target=target,
                window_size=window_size,
                threshold=threshold
            )