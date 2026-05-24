"""
=========================================================
settings.py
=========================================================

Arquivo central de configuração do projeto.

Objetivo:
- Centralizar caminhos;
- Definir parâmetros globais;
- Facilitar manutenção;
- Evitar valores hardcoded espalhados.

Boas práticas:
- Todo parâmetro importante deve vir daqui;
- Facilita reproducibilidade;
- Facilita experimentação.

=========================================================
"""

from pathlib import Path


# =========================================================
# DIRETÓRIO BASE DO PROJETO
# =========================================================
#
# Resolve automaticamente a raiz do projeto.
#
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# =========================================================
# DIRETÓRIOS DE DADOS
# =========================================================

DATA_RAW_DIR = BASE_DIR / "data" / "raw"

DATA_INTERIM_DIR = BASE_DIR / "data" / "interim"

DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"


# =========================================================
# DIRETÓRIOS DE SAÍDA
# =========================================================

OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR = OUTPUTS_DIR / "models"

# =========================================================
# ARQUIVOS DAS BASES
# =========================================================

BASE_A_FILE = DATA_RAW_DIR / "BaseA.csv"

BASE_B_FILE = DATA_RAW_DIR / "BaseB.csv"


# =========================================================
# RANDOM SEED
# =========================================================
#
# Garantir reproducibilidade.
#
# Sempre que possível usar a mesma seed:
# - Random Forest
# - divisão treino/teste
# - TensorFlow
# - NumPy
#
RANDOM_STATE = 42


# =========================================================
# EQUIPES SELECIONADAS
# =========================================================
#
# Equipes escolhidas inicialmente para o projeto.
#
# Critério:
# - estabilidade estatística;
# - boas campanhas;
# - alta consistência ofensiva.
#
SELECTED_TEAMS = [
    "Boston Celtics",
    "Denver Nuggets"
]

# =========================================================
# SLIDING WINDOWS
# =========================================================
#
# A prova exige:
# - 5
# - 10
# - 15
# - 20
#
WINDOW_SIZES = [5, 10, 15, 20]


# =========================================================
# TARGETS DE REGRESSÃO
# =========================================================
#
# RF3 da prova.
#
REGRESSION_TARGETS = [
    "PTS",
    "REB",
    "AST"
]