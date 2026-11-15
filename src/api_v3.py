"""API v3 - Arquitetura desacoplada em duas camadas (Two-Tier).

Camada de aplicacao (1a EC2): este servico FastAPI + modelo em memoria.
Camada de dados (2a EC2): PostgreSQL em Docker com a tabela `customers`
carregada a partir do churn.csv real (CSV -> postgres).

Diferenca para a v2: o lookup por codigo do cliente deixa de ler o arquivo
data/batch_predictions.parquet embutido na imagem e passa a executar uma query
SQL na rede, no banco da segunda maquina.
"""
import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.db import db_settings, get_engine
from src.inference import prepare_features, score

MODEL_PATH = os.path.join("models", "churn_pipeline.joblib")

app = FastAPI(
    title="Servico de Churn (v3 - Two-Tier)",
    description="Lookup de clientes no PostgreSQL remoto + inferencia em tempo real",
    version="3.0.0",
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Artefato do modelo nao encontrado em {MODEL_PATH}. Execute `python -m src.model_trainer` primeiro."
    )
model_pipeline = joblib.load(MODEL_PATH)

# Colunas da tabela `customers` (mesmos nomes do churn.csv, com aspas no Postgres)
CUSTOMER_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
_SELECT_COLS = ", ".join(f'"{c}"' for c in CUSTOMER_COLUMNS)
QUERY_BY_ID = text(f'SELECT {_SELECT_COLS} FROM customers WHERE "customerID" = :cid')


def _fetch_customer(customer_id: str) -> dict:
    """Executa o SELECT no PostgreSQL e devolve o registro como dicionario."""
    try:
        with get_engine().connect() as conn:
            row = conn.execute(QUERY_BY_ID, {"cid": customer_id}).mappings().first()
    except (SQLAlchemyError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Banco PostgreSQL indisponivel: {exc}") from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"Cliente '{customer_id}' nao encontrado no PostgreSQL.")
    record = dict(row)
    # NUMERIC chega como Decimal; converte para float para serializar em JSON
    if record.get("MonthlyCharges") is not None:
        record["MonthlyCharges"] = float(record["MonthlyCharges"])
    return record


class CustomerPayload(BaseModel):
    gender: str = "Male"
    SeniorCitizen: str = "0"
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = Field(default=12, ge=0)
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = Field(default=75.0, ge=0.0)
    TotalCharges: Optional[float] = None


@app.get("/health")
def health():
    cfg = db_settings()
    db = {"host": cfg["host"], "port": cfg["port"], "database": cfg["database"]}
    try:
        with get_engine().connect() as conn:
            db["records"] = conn.execute(text("SELECT count(*) FROM customers")).scalar_one()
        db["status"] = "connected"
    except Exception as exc:  # noqa: BLE001 - health deve sempre responder
        db["status"] = f"error: {exc.__class__.__name__}: {exc}".splitlines()[0]
    return {"status": "online", "model_loaded": model_pipeline is not None, "database": db}


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    """Consulta pura: devolve o registro do cliente exatamente como esta no banco."""
    return {"data_source": "postgresql", "customer": _fetch_customer(customer_id)}


@app.get("/results/{customer_id}")
def get_customer_score(customer_id: str):
    """Busca o cliente no PostgreSQL e calcula o score do modelo sobre esse registro."""
    record = _fetch_customer(customer_id)
    features = prepare_features(pd.DataFrame([record]))
    result = score(model_pipeline, features)
    return {
        "customer_id": record["customerID"],
        **result,
        "actual_churn": record["Churn"],
        "contract_type": record["Contract"],
        "monthly_charges": record["MonthlyCharges"],
        "total_charges": float(features["TotalCharges"].iloc[0]),
        "tenure": record["tenure"],
        "data_source": "postgresql",
        "customer": record,
    }


@app.post("/predict")
def predict_churn(customer: CustomerPayload):
    """Inferencia online para um cliente novo (nao depende do banco)."""
    data = customer.model_dump()
    if data["TotalCharges"] is None or data["TotalCharges"] <= 0:
        data["TotalCharges"] = float(data["tenure"] * data["MonthlyCharges"])
    try:
        result = score(model_pipeline, prepare_features(pd.DataFrame([data])))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na inferencia do modelo: {exc}") from exc
    return {**result, "inference_type": "real_time_online"}
