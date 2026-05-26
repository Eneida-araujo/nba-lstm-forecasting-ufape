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

Os melhores modelos são selecionados automaticamente
com base no menor RMSE registrado em:
outputs/metrics/lstm_results.csv
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

        # Remove gráficos antigos para evitar duplicidade
        for old_file in self.output_dir.glob("*.png"):
            old_file.unlink()

    def load_predictions(
        self,
        target: str,
        window_size: int
    ) -> pd.DataFrame:
        """
        Carrega as previsões salvas pelo LSTMTrainer.
        """

        file_path = (
            self.predictions_dir /
            f"lstm_predictions_{target.lower()}_window_{window_size}.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo de previsões não encontrado: {file_path}"
            )

        return pd.read_csv(file_path)

    def load_best_models(self) -> dict:
        """
        Carrega automaticamente os melhores modelos com base no menor RMSE
        registrado em outputs/metrics/lstm_results.csv.
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

        thresholds = {
            "PTS": 100,
            "REB": 30,
            "AST": 20
        }

        best_models = {}

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

            best_models[target] = {
                "window": int(best_row["window_size"]),
                "rmse": float(best_row["rmse"]),
                "threshold": thresholds[target]
            }

        print("[INFO] Melhores modelos carregados automaticamente:")
        print(best_models)

        return best_models

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

        plt.plot(
            df["y_true"].values,
            label="Real",
            linewidth=2
        )

        plt.plot(
            df["y_pred"].values,
            label="Previsto",
            linewidth=2
        )

        plt.title(
            f"{target} — Real vs Previsto | Window {window_size}",
            fontsize=18,
            pad=15
        )

        plt.xlabel("Amostras de teste", fontsize=13)
        plt.ylabel(target, fontsize=13)
        plt.legend()
        plt.tight_layout()

        output_path = (
            self.output_dir /
            f"real_vs_predicted_{target.lower()}_window_{window_size}.png"
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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

        plt.hist(
            df["error"],
            bins=30,
            edgecolor="black",
            alpha=0.80
        )

        plt.title(
            f"{target} — Histograma dos Erros | Window {window_size}",
            fontsize=18,
            pad=15
        )

        plt.xlabel("Erro", fontsize=13)
        plt.ylabel("Frequência", fontsize=13)
        plt.tight_layout()

        output_path = (
            self.output_dir /
            f"error_histogram_{target.lower()}_window_{window_size}.png"
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Gráfico salvo em: {output_path}")

    def plot_confidence_bar(
        self,
        target: str,
        window_size: int,
        rmse: float
    ):
        """
        Barra de confiança baseada em:
        margem = 1.96 × RMSE.
        """

        df = self.load_predictions(target, window_size)

        last_prediction = df["y_pred"].iloc[-1]

        margin = 1.96 * rmse

        plt.figure(figsize=(8, 6))

        plt.bar(
            [0],
            [last_prediction],
            yerr=[margin],
            capsize=12,
            alpha=0.85
        )

        plt.xticks(
            [0],
            [f"Previsão {target}"],
            fontsize=13
        )

        plt.ylabel(target, fontsize=13)

        plt.title(
            f"{target} — Previsão: "
            f"{last_prediction:.1f} ± {margin:.1f}",
            fontsize=18,
            pad=15
        )

        plt.tight_layout()

        output_path = (
            self.output_dir /
            f"confidence_bar_{target.lower()}_window_{window_size}.png"
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Gráfico salvo em: {output_path}")

    def plot_confusion_matrix(
        self,
        target: str,
        window_size: int,
        threshold: float
    ):
        """
        Matriz de confusão colorida em tons de vermelho.
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

        plt.figure(figsize=(7, 6))

        plt.imshow(
            cm,
            cmap="Reds",
            interpolation="nearest"
        )

        plt.title(
            f"{target} — Matriz de Confusão | Threshold {threshold}",
            fontsize=17,
            pad=15
        )

        plt.xlabel("Previsto", fontsize=13)
        plt.ylabel("Real", fontsize=13)

        plt.xticks([0, 1], ["Não", "Sim"], fontsize=12)
        plt.yticks([0, 1], ["Não", "Sim"], fontsize=12)

        threshold_color = cm.max() / 2 if cm.max() > 0 else 0

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):

                color = (
                    "white"
                    if cm[i, j] > threshold_color
                    else "black"
                )

                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=15,
                    fontweight="bold"
                )

        plt.colorbar()
        plt.tight_layout()

        output_path = (
            self.output_dir /
            f"confusion_matrix_{target.lower()}_window_{window_size}.png"
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Gráfico salvo em: {output_path}")

    def run_best_models(self):
        """
        Gera gráficos dos melhores modelos encontrados automaticamente,
        com base no menor RMSE.
        """

        best_models = self.load_best_models()

        for target, config in best_models.items():

            window_size = config["window"]
            threshold = config["threshold"]
            rmse = config["rmse"]

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
                window_size=window_size,
                rmse=rmse
            )

            self.plot_confusion_matrix(
                target=target,
                window_size=window_size,
                threshold=threshold
            )