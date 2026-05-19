"""
=========================================================
main.py
=========================================================

Arquivo principal do projeto.

Fluxo:
1. Carregar bases.
2. Fazer merge.
3. Preprocessar.
4. Validar.
5. Preparar pipeline para Feature Engineering.

=========================================================
"""

from src.config.settings import (
    BASE_A_FILE,
    BASE_B_FILE
)

from src.data.loader import DataLoader

from src.data.merger import DataMerger

from src.data.preprocessing import DataPreprocessor

from src.data.validation import DataValidator


def main():

    print("\n================================================")
    print("NBA LSTM FORECASTING")
    print("================================================")

    # =====================================================
    # CARREGAMENTO DAS BASES
    # =====================================================

    df_a = DataLoader.load_csv(BASE_A_FILE)

    df_b = DataLoader.load_csv(BASE_B_FILE)

    # =====================================================
    # MERGE DAS BASES
    # =====================================================

    merged_df = DataMerger.merge_datasets(
        df_a,
        df_b
    )

    # =====================================================
    # PREPROCESSAMENTO
    # =====================================================

    preprocessor = DataPreprocessor(
        merged_df
    )

    processed_df = preprocessor.preprocess()

    # =====================================================
    # VALIDAÇÃO
    # =====================================================

    DataValidator.validate_dataframe(
        processed_df
    )


if __name__ == "__main__":

    main()