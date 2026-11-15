"""Pontuacao em lote (app-churn-v1): aplica o @champion a todos os clientes reais.

Gera data/batch_predictions.parquet, consultado pela API de lookup da v1
(src/api_lookup.py), que nao faz inferencia: apenas le os resultados prontos.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.inference import ChampionModel, risk_level

RAW_DATA_PATH = os.path.join("data", "raw", "churn.csv")
OUTPUT_PARQUET_PATH = os.path.join("data", "batch_predictions.parquet")

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Base de dados real nao encontrada em {RAW_DATA_PATH}")

champion = ChampionModel()
print(f"--> Modelo: {champion.info['model_name']} v{champion.info['version']} @champion ({champion.info['run_name']})")

print(f"--> Carregando dataset real ({RAW_DATA_PATH})...")
df = pd.read_csv(RAW_DATA_PATH)

print(f"--> Executando inferencia em lote para {len(df)} clientes reais...")
probs = champion.predict_proba(df.drop(columns=["customerID", "Churn"]))

results_df = pd.DataFrame({
    "customer_id": df["customerID"],
    "churn_prediction": (probs >= 0.5).astype(int),
    "churn_probability": np.round(probs, 4),
    "risk_level": [risk_level(p) for p in probs],
    "contract_type": df["Contract"],
    "monthly_charges": df["MonthlyCharges"],
    "total_charges": pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0),
    "tenure": df["tenure"].astype(int),
    "model_version": champion.info["version"],
    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
})

os.makedirs(os.path.dirname(OUTPUT_PARQUET_PATH), exist_ok=True)
results_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
print(f"--> Sucesso: {len(results_df)} predicoes salvas em {OUTPUT_PARQUET_PATH}")
