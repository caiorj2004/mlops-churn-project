from pathlib import Path

# Caminhos base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_rf.pkl"

# Hiperparâmetros e constantes
TEST_SIZE = 0.25
N_ESTIMATORS = 200
RANDOM_STATE = 42  # semente fixa: mesmo dado + mesmo codigo = mesma acuracia