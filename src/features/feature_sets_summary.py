"""
Gera um quadro-resumo dos conjuntos de features por experimento.

O objetivo é facilitar:
- análise dos feature sets;
- comparação entre p-value e Random Forest;
- identificação das features da interseção;
- revisão do conjunto híbrido;
- uso posterior no relatório.
"""

from pathlib import Path
import pandas as pd


class FeatureSetsSummary:

    def __init__(
        self,
        tables_dir: str | Path = "outputs/tables",
        output_dir: str | Path = "outputs/tables"
    ):
        self.tables_dir = Path(tables_dir)
        self.output_dir = Path(output_dir)

    def load_feature_sets(self, target: str) -> pd.DataFrame:
        file_path = self.tables_dir / f"feature_sets_{target.lower()}.csv"

        if not file_path.exists():
            print(f"[WARNING] Arquivo não encontrado: {file_path}")
            return pd.DataFrame()

        return pd.read_csv(file_path)

    def create_summary_table(self, targets: list[str]) -> pd.DataFrame:
        rows = []

        for target in targets:
            df = self.load_feature_sets(target)

            if df.empty:
                continue

            for feature_set in df["feature_set"].unique():
                subset = df[df["feature_set"] == feature_set]

                features = subset["feature"].tolist()

                rows.append({
                    "target": target,
                    "experiment": feature_set,
                    "n_features": len(features),
                    "features": ", ".join(features)
                })

        return pd.DataFrame(rows)

    def save_summary(self, summary_df: pd.DataFrame):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.output_dir / "feature_sets_summary.csv"
        md_path = self.output_dir / "feature_sets_summary.md"

        summary_df.to_csv(csv_path, index=False)

        with open(md_path, "w", encoding="utf-8") as file:
            file.write("# Quadro-resumo dos Conjuntos de Features\n\n")
            file.write(summary_df.to_markdown(index=False))

        print(f"[INFO] Quadro CSV salvo em: {csv_path}")
        print(f"[INFO] Quadro Markdown salvo em: {md_path}")

    def run(self, targets: list[str]):
        summary_df = self.create_summary_table(targets)
        self.save_summary(summary_df)

        return summary_df