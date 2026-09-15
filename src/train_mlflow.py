import subprocess
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data import load_clean_data
from src.evaluate import evaluate_model
from src.features import process_features


def get_dvc_md5(dvc_pointer_path: str = "data/raw/churn.csv.dvc") -> str:
    """Recupera o hash MD5 da versão do dado no DVC."""
    try:
        with open(dvc_pointer_path, "r") as f:
            doc = yaml.safe_load(f)
            return doc["outs"][0]["md5"]
    except Exception:
        return "unknown_dvc"


def get_git_commit() -> str:
    """Recupera o hash reduzido do commit do Git."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        return "unknown_git"


def run_experiment():
    # 1. Ativa o Autologging para Scikit-Learn
    mlflow.sklearn.autolog()

    # 2. Define o nome do experimento
    mlflow.set_experiment("churn-prediction")

    # 3. Inicia a execução rastreada
    with mlflow.start_run(run_name="rf-autolog-baseline"):
        # Registra a linhagem (DVC + Git)
        mlflow.set_tags(
            {"data_md5": get_dvc_md5(), "git_commit": get_git_commit()}
        )

        print("1. Carregando dados validados...")
        df = load_clean_data()

        print("2. Processando features...")
        X, y = process_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("3. Treinando modelo (rastreado pelo MLflow)...")
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        model.fit(X_train, y_train)

        print("4. Avaliando modelo...")
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    run_experiment()