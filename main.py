"""
=========================================================
main.py
=========================================================

Arquivo principal do projeto.

Fluxo:
1. Carregar bases.
2. Merge dos datasets.
3. Pré-processamento.
4. Engenharia de features.
5. Correlation Filter.
6. Seleção por p-value.
7. Random Forest Feature Importance.
8. Construção dos Feature Sets.
9. Benchmark experimental.
10. Sliding Windows.
11. Treinamento LSTM.
12. Predições finais.
13. Geração de gráficos e métricas.
=========================================================
"""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config.settings import (
    BASE_A_FILE,
    BASE_B_FILE
)

from src.data.loader import DataLoader
from src.data.merger import DataMerger
from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator

from src.features.engineering import FeatureEngineer
from src.features.correlation_filter import CorrelationFilter
from src.features.pvalue_selection import PValueSelector
from src.features.random_forest_importance import RandomForestFeatureImportance
from src.features.feature_sets import FeatureSetBuilder
from src.features.feature_sets_summary import FeatureSetsSummary
from src.features.feature_experiments import FeatureExperimentRunner

from src.models.windowing import SlidingWindowGenerator
from src.models.lstm_trainer import LSTMTrainer
from src.models.predictor import NBAPredictor

from src.visualization.plots import PlotGenerator


# =====================================================
# REPRODUTIBILIDADE
# =====================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def main():

    print("\n================================================")
    print("NBA LSTM FORECASTING")
    print("================================================")

    df_a = DataLoader.load_csv(BASE_A_FILE)
    df_b = DataLoader.load_csv(BASE_B_FILE)

    merged_df = DataMerger.merge_datasets(
        df_a,
        df_b
    )

    preprocessor = DataPreprocessor(
        merged_df
    )

    processed_df = preprocessor.preprocess()

    engineer = FeatureEngineer(
        processed_df
    )

    featured_df = engineer.run()

    correlation_filter = CorrelationFilter(
        dataframe=featured_df,
        threshold=0.90,
        output_path="outputs/tables/correlation_removed_features.csv"
    )

    filtered_df = correlation_filter.apply_filter()

    DataValidator.validate_dataframe(filtered_df)

    for target in ["PTS", "REB", "AST"]:

        pvalue_selector = PValueSelector(
            dataframe=filtered_df,
            target_column=target,
            significance_level=0.05,
            output_dir="outputs/tables"
        )

        selected_features = pvalue_selector.run()

        print(
            f"[INFO] Target {target}: "
            f"{len(selected_features)} "
            f"features selecionadas por p-value."
        )

    for target in ["PTS", "REB", "AST"]:

        rf_importance = RandomForestFeatureImportance(
            dataframe=filtered_df,
            target_column=target,
            n_estimators=300,
            random_state=SEED,
            output_dir="outputs/tables"
        )

        importance_df = rf_importance.run()

        print(
            f"[INFO] Target {target}: "
            f"{len(importance_df)} features ranqueadas pelo Random Forest."
        )

    for target in ["PTS", "REB", "AST"]:

        feature_set_builder = FeatureSetBuilder(
            target_column=target,
            tables_dir="outputs/tables",
            output_dir="outputs/tables"
        )

        feature_sets = feature_set_builder.run()

        print(
            f"[INFO] Target {target}: "
            f"{len(feature_sets)} conjuntos de features criados."
        )

    summary_builder = FeatureSetsSummary(
        tables_dir="outputs/tables",
        output_dir="outputs/tables"
    )

    feature_sets_summary = summary_builder.run(
        targets=["PTS", "REB", "AST"]
    )

    print("\n[INFO] Quadro-resumo dos feature sets:")
    print(feature_sets_summary)

    for target in ["PTS", "REB", "AST"]:

        experiment_runner = FeatureExperimentRunner(
            dataframe=filtered_df,
            target_column=target,
            tables_dir="outputs/tables",
            output_dir="outputs/metrics",
            random_state=SEED
        )

        experiment_results = experiment_runner.run()

        print(f"\n[INFO] Ranking dos feature sets para {target}:")
        print(
            experiment_results[
                [
                    "feature_set",
                    "n_features",
                    "mae",
                    "rmse",
                    "mape",
                    "r2"
                ]
            ]
        )

    print("\n================================================")
    print("SLIDING WINDOW TEST")
    print("================================================")

    target = "PTS"

    feature_sets_df = pd.read_csv(
        "outputs/tables/feature_sets_pts.csv"
    )

    hybrid_features = (
        feature_sets_df[
            feature_sets_df["feature_set"] == "hybrid"
        ]["feature"]
        .dropna()
        .tolist()
    )

    hybrid_features = [
        feature
        for feature in hybrid_features
        if feature in filtered_df.columns
    ]

    print(
        f"[INFO] Quantidade de features Hybrid: "
        f"{len(hybrid_features)}"
    )

    for window_size in [5, 10, 15, 20]:

        window_generator = SlidingWindowGenerator(
            dataframe=filtered_df,
            feature_columns=hybrid_features,
            target_column=target,
            window_size=window_size
        )

        X, y = window_generator.run()

    lstm_trainer = LSTMTrainer(
        dataframe=filtered_df,
        targets=["PTS", "REB", "AST"],
        windows=[5, 10, 15, 20],
        feature_set_name="hybrid",
        tables_dir="outputs/tables",
        output_dir="outputs/metrics",
        models_dir="outputs/models"
    )

    lstm_results = lstm_trainer.run()

    print("\n[INFO] Resultados finais da LSTM:")
    print(lstm_results)

    print("\n================================================")
    print("FINAL NBA PREDICTIONS")
    print("================================================")

    selected_teams = [
        "Boston Celtics",
        "Denver Nuggets"
    ]

    prediction_reports = []

    for team_name in selected_teams:

        predictor = NBAPredictor(
            dataframe=filtered_df,
            team_name=team_name,
            feature_set_name="hybrid",
            tables_dir="outputs/tables",
            models_dir="outputs/models",
            metrics_dir="outputs/metrics"
        )

        team_report = predictor.run()

        prediction_reports.append(team_report)

    final_predictions_df = pd.concat(
        prediction_reports,
        ignore_index=True
    )

    print("\n================================================")
    print("FINAL PREDICTION TABLE")
    print("================================================")

    print(final_predictions_df)

    final_predictions_path = (
        "outputs/metrics/final_predictions.csv"
    )

    final_predictions_df.to_csv(
        final_predictions_path,
        index=False
    )

    print(
        f"\n[INFO] Previsões finais salvas em: "
        f"{final_predictions_path}"
    )

    plot_generator = PlotGenerator(
        predictions_dir="outputs/predictions",
        metrics_dir="outputs/metrics",
        output_dir="outputs/figures"
    )

    plot_generator.run_best_models()


if __name__ == "__main__":
    main()