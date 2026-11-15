"""API do app-churn-v2 - arquitetura desacoplada em duas camadas (Two-Tier).

Camada de aplicacao (1a EC2): este servico FastAPI + modelo em memoria.
Camada de dados (2a EC2): PostgreSQL em Docker com a tabela `customers`
carregada a partir do churn.csv real (CSV -> postgres).

Mudanca desta atividade: o lookup por codigo do cliente deixa de ler o arquivo
data/batch_predictions.parquet embutido na imagem e passa a executar uma query
SQL na rede, no banco da segunda maquina. A inferencia online (POST /predict)
continua igual a da v2.

Modelo: o @champion do MLflow Model Registry, exportado para models/champion/
(src/export_champion.py) e copiado para a imagem no docker build.
"""
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.db import db_settings, get_engine
from src.inference import ChampionModel

app = FastAPI(
    title="Servico de Churn (v2 - Two-Tier)",
    description="Lookup de clientes no PostgreSQL remoto + inferencia em tempo real",
    version="2.1.0",
)

champion = ChampionModel()
print(f"Modelo carregado: {champion.info['model_name']} v{champion.info['version']} @champion ({champion.info['run_name']})")

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


@app.get("/")
def root():
    return {
        "servico": "Churn API v2 (Two-Tier: app + PostgreSQL)",
        "endpoints": {
            "GET /health": "status do modelo e do banco",
            "GET /customers/{customerID}": "registro do cliente no PostgreSQL",
            "GET /results/{customerID}": "registro do banco + score do modelo",
            "POST /predict": "inferencia online para cliente novo",
            "GET /model-info": "versao, metricas e linhagem do modelo @champion",
            "GET /docs": "documentacao interativa (Swagger)",
        },
    }


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
    return {"status": "online", "model": champion.model_summary(), "database": db}


@app.get("/model-info")
def model_info():
    """Metadados do @champion em uso: versao no registry, run, metricas e linhagem."""
    return champion.info


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    """Consulta pura: devolve o registro do cliente exatamente como esta no banco."""
    return {"data_source": "postgresql", "customer": _fetch_customer(customer_id)}


@app.get("/results/{customer_id}")
def get_customer_score(customer_id: str):
    """Busca o cliente no PostgreSQL e calcula o score do modelo sobre esse registro."""
    record = _fetch_customer(customer_id)
    entrada = {k: v for k, v in record.items() if k not in ("customerID", "Churn")}
    try:
        result = champion.score_one(entrada)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    total = pd.to_numeric(pd.Series([record["TotalCharges"]]), errors="coerce").fillna(0.0).iloc[0]
    return {
        "customer_id": record["customerID"],
        **result,
        "actual_churn": record["Churn"],
        "contract_type": record["Contract"],
        "monthly_charges": record["MonthlyCharges"],
        "total_charges": float(total),
        "tenure": record["tenure"],
        "data_source": "postgresql",
        "model": champion.model_summary(),
        "customer": record,
    }


@app.post("/predict")
def predict_churn(customer: CustomerPayload):
    """Inferencia online para um cliente novo (nao depende do banco)."""
    data = customer.model_dump()
    if data["TotalCharges"] is None or data["TotalCharges"] <= 0:
        data["TotalCharges"] = float(data["tenure"] * data["MonthlyCharges"])
    try:
        result = champion.score_one(data)
    except ValueError as exc:  # categoria fora do contrato de features
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**result, "inference_type": "real_time_online", "model": champion.model_summary()}
