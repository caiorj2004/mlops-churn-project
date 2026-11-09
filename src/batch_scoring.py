import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

RAW_DATA_PATH = os.path.join("data", "raw", "churn.csv")
MODEL_PATH = os.path.join("models", "churn_pipeline.joblib")
OUTPUT_PARQUET_PATH = os.path.join("data", "batch_predictions.parquet")

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Base de dados real nao encontrada em {RAW_DATA_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Pipeline treinado nao encontrado em {MODEL_PATH}. Execute src/model_trainer.py primeiro.")

print(f"--> Carregando dataset real ({RAW_DATA_PATH})...")
df = pd.read_csv(RAW_DATA_PATH)
id_col = "customer_id" if "customer_id" in df.columns else "customerID"
df = df.rename(columns={id_col: "customer_id"})

print(f"--> Carregando modelo serializado ({MODEL_PATH})...")
pipeline = joblib.load(MODEL_PATH)

# Preparacao e tipagem identica ao pipeline de treino
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce").fillna(0.0)
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(1)
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
df["gasto_por_mes"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

cat_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]

for col in cat_features:
    df[col] = df[col].astype(str)

print(f"--> Executando inferencia em lote para {len(df)} clientes reais...")
probs = pipeline.predict_proba(df)[:, 1]
preds = (probs >= 0.5).astype(int)

# Construcao da base consolidada para consulta (Lookup)
results_df = pd.DataFrame({
    "customer_id": df["customer_id"],
    "churn_prediction": preds,
    "churn_probability": np.round(probs, 4),
    "risk_level": np.where(probs >= 0.6, "Alto", np.where(probs >= 0.3, "Medio", "Baixo")),
    "contract_type": df["Contract"],
    "monthly_charges": df["MonthlyCharges"],
    "total_charges": df["TotalCharges"],
    "tenure": df["tenure"].astype(int),
    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
})

os.makedirs(os.path.dirname(OUTPUT_PARQUET_PATH), exist_ok=True)
results_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
print(f"--> Sucesso: {len(results_df)} predicoes salvas em {OUTPUT_PARQUET_PATH}")
