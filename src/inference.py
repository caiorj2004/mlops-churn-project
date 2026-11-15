"""Preparacao de features e scoring compartilhados pela API (src/api_v2.py).

Replica exatamente o tratamento usado em src/model_trainer.py, para que um
registro vindo do PostgreSQL chegue ao modelo no mesmo formato do treino.
"""
from datetime import datetime

import numpy as np
import pandas as pd

NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "gasto_por_mes"]
CAT_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce"
    ).fillna(0.0)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(1)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0).astype(float)
    df["gasto_por_mes"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    for col in CAT_FEATURES:
        df[col] = df[col].astype(str)
    return df[NUM_FEATURES + CAT_FEATURES]


def risk_level(prob: float) -> str:
    return "Alto" if prob >= 0.6 else ("Medio" if prob >= 0.3 else "Baixo")


def score(pipeline, features: pd.DataFrame) -> dict:
    prob = float(pipeline.predict_proba(features)[0, 1])
    return {
        "churn_prediction": int(prob >= 0.5),
        "churn_probability": round(prob, 4),
        "risk_level": risk_level(prob),
        "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
