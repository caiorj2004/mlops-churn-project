import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Evita travamento da GUI/Tkinter durante autologging

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.data import load_clean_data
from src.features import build_feature_contract, process_features
from src.train_mlflow import get_dvc_md5, get_git_commit

RELATORIO = Path("reports") / "grid_metricas.json"   # saida do estagio treino_e_tuning (dvc.yaml)


def metricas_teste(y_true, pred, prob) -> dict:
    """Acuracia (criterio da disciplina) + metricas que olham a classe minoritaria (churn = 32,5%)."""
    return {
        "test_accuracy": accuracy_score(y_true, pred),
        "test_roc_auc": roc_auc_score(y_true, prob),
        "test_precision": precision_score(y_true, pred, zero_division=0),
        "test_recall": recall_score(y_true, pred),
        "test_f1": f1_score(y_true, pred),
    }


def logar_modelo(model) -> None:
    """Grava o modelo no run como `model` (runs:/<id>/model), no MLflow 2.x e no 3.x."""
    # pickle: formato aceito no 2.x e no 3.x (o 3.x passou a usar skops por padrao, que exige lista de tipos confiaveis na carga)
    try:
        mlflow.sklearn.log_model(model, name="model", serialization_format="pickle")           # MLflow 3.x
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="pickle")  # MLflow 2.x


def run_grid():
    # Conecta ao mesmo banco SQLite utilizado pela API FastAPI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # autolog registra parametros e metricas de treino; o modelo e logado explicitamente abaixo
    # (no MLflow 3 o autolog nao grava mais o artefato `model` que o registry e o export usam)
    mlflow.sklearn.autolog(log_models=False)
    mlflow.set_experiment("churn-prediction")

    print("Carregando e processando dados...")
    df = load_clean_data()
    contrato = build_feature_contract(df)  # categorias + ordem das colunas do treino
    X, y = process_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    resultados = []
    n_estimators_list = [100, 200, 500]
    max_depth_list = [None, 8, 16]

    for n in n_estimators_list:
        for depth in max_depth_list:
            run_name = f"rf-n{n}-d{depth}"
            print(f"Executando run: {run_name}")

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({
                    "data_md5": get_dvc_md5(),
                    "git_commit": get_git_commit()
                })
                # Contrato de features salvo junto do modelo: a inferencia (APIs)
                # codifica cada cliente exatamente como neste treino
                mlflow.log_dict(contrato, "feature_contract.json")

                model = RandomForestClassifier(
                    n_estimators=n, max_depth=depth, random_state=42
                )
                model.fit(X_train, y_train)

                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:, 1]
                metricas = metricas_teste(y_test, preds, probs)
                mlflow.log_metrics(metricas)
                logar_modelo(model)
                resultados.append({"run": run_name, "n_estimators": n, "max_depth": depth,
                                   **{k: round(v, 4) for k, v in metricas.items()}})

    # Resumo do grid para o DVC (`dvc metrics show`) e para quem le o repositorio
    melhor = max(resultados, key=lambda r: r["test_accuracy"])
    RELATORIO.parent.mkdir(exist_ok=True)
    RELATORIO.write_text(json.dumps({"melhor_por_acuracia": melhor, "runs": resultados},
                                    indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resumo do grid salvo em {RELATORIO}")

if __name__ == "__main__":
    run_grid()
