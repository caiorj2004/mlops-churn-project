"""API de predicao de churn com o modelo @champion do MLflow Model Registry.

Encontro de FastAPI: o usuario informa o codigo do cliente e a API devolve a
predicao do modelo @champion junto com as informacoes desse modelo.

As features sao montadas com as MESMAS funcoes do treino (src/data.py +
src/features.py), a partir do registro real do cliente no churn.csv.
"""
from typing import Optional

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Path
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.data import load_clean_data
from src.features import numeric_features, process_features

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="Serviço de Predição de Churn - MLOps",
    description="API para inferência e consulta do modelo @champion",
    version="1.1.0",
)

MODEL_NAME = "churn-model"
ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{ALIAS}"
EXEMPLO_ID = "7015-DBER"

client = MlflowClient()
champion_model = None
champion_metadata: dict = {}

# Base historica: dado validado pelo Pandera e transformado como no treino.
df_base = load_clean_data()
X_base, _ = process_features(df_base.copy())
X_base.index = df_base.loc[X_base.index, "customerID"]
df_base = df_base.set_index("customerID")
print(f"Base histórica carregada com {len(X_base)} clientes.")


def load_champion():
    """Carrega o @champion do Model Registry. Sem fallback: se falhar, a API informa."""
    global champion_model, champion_metadata
    try:
        champion_model = mlflow.pyfunc.load_model(MODEL_URI)
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        run = client.get_run(mv.run_id)
        champion_metadata = {
            "model_name": MODEL_NAME,
            "alias": f"@{ALIAS}",
            "version": mv.version,
            "run_id": mv.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "test_accuracy": run.data.metrics.get("test_accuracy"),
            "hyperparameters": run.data.params,
            "data_md5": run.data.tags.get("data_md5"),
            "git_commit": run.data.tags.get("git_commit"),
            "status": "loaded",
        }
        print(f"Modelo carregado do Model Registry: {MODEL_URI} (versão {mv.version})")
    except Exception as e:
        champion_model = None
        champion_metadata = {"status": "indisponivel", "error": str(e)}
        print(f"Aviso: não foi possível carregar {MODEL_URI}: {e}")


load_champion()


class CustomerInput(BaseModel):
    customer_id: str = Field(..., examples=[EXEMPLO_ID])
    tenure: Optional[int] = Field(None, ge=0, examples=[6])
    monthly_charges: Optional[float] = Field(None, ge=0, examples=[85.0])
    total_charges: Optional[float] = Field(None, ge=0, examples=[510.0])


def run_inference(cust_id: str, tenure: Optional[int] = None,
                  monthly: Optional[float] = None, total: Optional[float] = None):
    if champion_model is None:
        raise HTTPException(status_code=503, detail=f"Modelo @champion indisponivel: {champion_metadata.get('error')}")
    if cust_id not in X_base.index:
        raise HTTPException(status_code=404, detail=f"Cliente '{cust_id}' não encontrado na base.")

    features = X_base.loc[[cust_id]].copy()
    raw = df_base.loc[cust_id]
    simulados = {k: v for k, v in {"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total}.items() if v is not None}

    if simulados:
        # Simulacao "e se": troca valores numericos do cliente e recalcula as features
        t = float(simulados.get("tenure", raw["tenure"]))
        m = float(simulados.get("MonthlyCharges", raw["MonthlyCharges"]))
        tc = float(simulados.get("TotalCharges", raw["TotalCharges"]))
        for coluna, valor in numeric_features(t, m, tc).items():
            features[coluna] = valor

    churn_result = int(champion_model.predict(features)[0])
    return {
        "customer_id": cust_id,
        "churn_prediction": churn_result,
        "label": "Churn" if churn_result == 1 else "No Churn",
        "actual_churn": raw["Churn"],
        "customer_source": "base_historica_com_simulacao" if simulados else "base_historica",
        "simulated_values": simulados,
        "champion_model_info": champion_metadata,
    }


@app.get("/")
def read_root():
    return {"status": "online", "message": "API MLOps ativa. Acesse /docs"}


@app.get("/model-info")
def get_model_info():
    if champion_model is None:
        raise HTTPException(status_code=503, detail=champion_metadata)
    return champion_metadata


@app.post("/predict")
def predict_churn_post(data: CustomerInput):
    return run_inference(data.customer_id, data.tenure, data.monthly_charges, data.total_charges)


@app.get("/predict/{customer_id}")
def predict_churn_get(customer_id: str = Path(..., examples=[EXEMPLO_ID])):
    return run_inference(customer_id)
