"""
=========================================================
feature_sets.py
=========================================================

Constrói conjuntos de features para os experimentos.

Experimentos criados:
1. p-value
2. Random Forest
3. Interseção p-value + Random Forest
4. Top 10 Random Forest
5. Top 20 Random Forest
6. Híbrido
=========================================================
"""

from pathlib import Path
import pandas as pd


class FeatureSetBuilder:

    def __init__(
        self,
        target_column: str,
        tables_dir: str | Path = "outputs/tables",
        output_dir: str | Path = "outputs/tables"
    ):
        self.target_column = target_column.lower()
        self.tables_dir = Path(tables_dir)
        self.output_dir = Path(output_dir)
        self.feature_sets = {}

    def load_pvalue_features(self) -> list:
        file_path = (
            self.tables_dir /
            f"selected_features_pvalue_{self.target_column}.csv"
        )

        if not file_path.exists():
            print(f"[WARNING] Arquivo não encontrado: {file_path}")
            return []

        df = pd.read_csv(file_path)
        return df["feature"].dropna().tolist()

    def load_random_forest_features(self) -> pd.DataFrame:
        file_path = (
            self.tables_dir /
            f"rf_feature_importance_{self.target_column}.csv"
        )

        if not file_path.exists():
            print(f"[WARNING] Arquivo não encontrado: {file_path}")
            return pd.DataFrame(columns=["feature", "importance"])

        return pd.read_csv(file_path)

    def build_pvalue_set(self) -> list:
        return self.load_pvalue_features()

    def build_random_forest_set(self) -> list:
        rf_df = self.load_random_forest_features()

        if rf_df.empty:
            return []

        return (
            rf_df[rf_df["importance"] > 0]["feature"]
            .dropna()
            .tolist()
        )

    def build_intersection_set(self) -> list:
        pvalue_features = set(self.load_pvalue_features())
        rf_features = set(self.build_random_forest_set())

        return sorted(
            list(pvalue_features.intersection(rf_features))
        )

    def build_top_n_random_forest_set(self, n: int) -> list:
        rf_df = self.load_random_forest_features()

        if rf_df.empty:
            return []

        return (
            rf_df.head(n)["feature"]
            .dropna()
            .tolist()
        )

    def build_hybrid_set(self) -> list:
        pvalue_features = set(self.load_pvalue_features())
        top_20_rf = set(self.build_top_n_random_forest_set(20))

        technical_features = {
            "IS_HOME",
            "DAYS_REST",
            "WIN_STREAK",
            "LOSS_STREAK",
            "AST_TO_TOV",
            "REB_BALANCE_LAG1",
            "SHOOTING_EFFICIENCY_LAG1",
            "PTS_LAG1",
            "REB_LAG1",
            "AST_LAG1",
            "PLUS_MINUS_LAG1",
            "W_PCT_LAG1",
            "FG_PCT_LAG1",
            "FG3_PCT_LAG1",
            "FT_PCT_LAG1",
        }

        return sorted(
            list(
                pvalue_features
                .union(top_20_rf)
                .union(technical_features)
            )
        )

    def build_all_sets(self) -> dict:
        print("\n================================================")
        print(f"FEATURE SETS — TARGET: {self.target_column.upper()}")
        print("================================================")

        self.feature_sets = {
            "pvalue": self.build_pvalue_set(),
            "random_forest": self.build_random_forest_set(),
            "intersection": self.build_intersection_set(),
            "top_10_rf": self.build_top_n_random_forest_set(10),
            "top_20_rf": self.build_top_n_random_forest_set(20),
            "hybrid": self.build_hybrid_set()
        }

        for set_name, features in self.feature_sets.items():
            print(f"[INFO] {set_name}: {len(features)} features")

        return self.feature_sets

    def save_feature_sets(self):
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        rows = []

        for set_name, features in self.feature_sets.items():
            for feature in features:
                rows.append({
                    "target": self.target_column.upper(),
                    "feature_set": set_name,
                    "feature": feature
                })

        output_df = pd.DataFrame(rows)

        output_path = (
            self.output_dir /
            f"feature_sets_{self.target_column}.csv"
        )

        output_df.to_csv(
            output_path,
            index=False
        )

        print(f"[INFO] Feature sets salvos em: {output_path}")

    def run(self) -> dict:
        self.build_all_sets()
        self.save_feature_sets()

        return self.feature_sets