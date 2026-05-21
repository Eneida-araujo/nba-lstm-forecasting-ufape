"""
=========================================================
random_forest_importance.py
=========================================================

Responsável por calcular a importância das features usando
Random Forest.

Objetivo:
- identificar variáveis com maior poder preditivo;
- capturar relações não lineares;
- complementar a seleção estatística feita pelo p-value;
- gerar rankings de importância para os experimentos.

O p-value mede significância estatística linear.
O Random Forest mede importância prática preditiva.

Ambos serão usados em conjunto para selecionar os melhores
conjuntos de features para os modelos LSTM.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RandomForestFeatureImportance:
    """
    Calcula a importância das features usando RandomForestRegressor.

    Nesta etapa, o Random Forest não será o modelo final.
    Ele será usado como ferramenta auxiliar para selecionar variáveis.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        n_estimators: int = 300,
        random_state: int = 42,
        output_dir: str | Path = "outputs/tables"
    ):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset após engenharia de features e filtro de correlação.

        target_column : str
            Alvo de previsão. Exemplo: PTS, REB ou AST.

        n_estimators : int
            Quantidade de árvores da Random Forest.

        random_state : int
            Semente para reprodutibilidade.

        output_dir : str | Path
            Pasta onde os rankings de importância serão salvos.
        """

        self.df = dataframe.copy()
        self.target_column = target_column
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.output_dir = Path(output_dir)

        self.importance_df = None

    def get_forbidden_columns(self) -> list:
        """
        Define colunas que não devem ser usadas como features.

        Motivo:
        Essas colunas representam identificadores, targets ou estatísticas
        do próprio jogo, o que poderia causar vazamento de dados.
        """

        return [
            "SEASON_ID",
            "TEAM_ID",
            "GAME_ID",
            "VIDEO_AVAILABLE",

            "PTS",
            "REB",
            "AST",
            "FGM",
            "FGA",
            "FG_PCT",
            "FG3M",
            "FG3A",
            "FG3_PCT",
            "FTM",
            "FTA",
            "FT_PCT",
            "OREB",
            "DREB",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PLUS_MINUS",

            "W",
            "L",
            "W_PCT"
        ]

    def get_candidate_features(self) -> list:
        """
        Seleciona as features candidatas.

        Apenas colunas numéricas são utilizadas.
        Variáveis textuais e colunas proibidas são removidas.
        """

        numeric_columns = self.df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        forbidden_columns = self.get_forbidden_columns()

        candidate_features = [
            column
            for column in numeric_columns
            if column not in forbidden_columns
            and column != self.target_column
        ]

        return candidate_features

    def prepare_data(self):
        """
        Prepara X e y para o Random Forest.

        Etapas:
        - selecionar features candidatas;
        - substituir infinitos por NaN;
        - preencher valores ausentes pela mediana;
        - remover colunas constantes;
        - preservar ordem temporal dos dados.
        """

        features = self.get_candidate_features()

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

    def train_model(self):
        """
        Treina o Random Forest.

        O modelo será usado para extrair feature_importances_.
        """

        X, y = self.prepare_data()

        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )

        model.fit(X, y)

        importances = model.feature_importances_

        self.importance_df = (
            pd.DataFrame({
                "feature": X.columns,
                "importance": importances
            })
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

        return model, X, y

    def evaluate_temporal_baseline(self, model, X, y):
        """
        Avalia o Random Forest como baseline simples.

        Observação:
        Essa avaliação não substitui a validação final da LSTM.
        Ela serve apenas como referência inicial da capacidade
        preditiva das features.
        """

        split_index = int(len(X) * 0.80)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        baseline_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

        baseline_model.fit(X_train, y_train)

        predictions = baseline_model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print("[INFO] Baseline temporal Random Forest:")
        print(f"[INFO] MAE: {mae:.4f}")
        print(f"[INFO] RMSE: {rmse:.4f}")
        print(f"[INFO] R²: {r2:.4f}")

    def save_results(self):
        """
        Salva o ranking de importância das features.

        O arquivo será usado posteriormente para:
        - criar Top 10;
        - criar Top 20;
        - cruzar com p-value;
        - montar os experimentos.
        """

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            self.output_dir /
            f"rf_feature_importance_{self.target_column.lower()}.csv"
        )

        self.importance_df.to_csv(
            output_path,
            index=False
        )

        print(f"[INFO] Importância das features salva em: {output_path}")

    def run(self) -> pd.DataFrame:
        """
        Executa o pipeline completo:
        - prepara dados;
        - treina Random Forest;
        - calcula importâncias;
        - avalia baseline;
        - salva resultados.
        """

        print("\n================================================")
        print(f"RANDOM FOREST IMPORTANCE — TARGET: {self.target_column}")
        print("================================================")

        model, X, y = self.train_model()

        print(f"[INFO] Features candidatas: {len(X.columns)}")
        print("[INFO] Top 10 features mais importantes:")

        print(
            self.importance_df
            .head(10)
            .to_string(index=False)
        )

        self.evaluate_temporal_baseline(model, X, y)

        self.save_results()

        return self.importance_df