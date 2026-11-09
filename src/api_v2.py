import os
import joblib
import pandas as pd
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Servico Hibrido de Churn (v2)",
    description="Endpoint de lookup historico e inferencia em tempo real"
)

PARQUET_PATH = os.path.join("data", "batch_predictions.parquet")
MODEL_PATH = os.path.join("models", "churn_pipeline.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Artefato do modelo nao encontrado em {MODEL_PATH}. Execute src/model_trainer.py primeiro.")

model_pipeline = joblib.load(MODEL_PATH)
print("Pipeline de inferencia online carregado em memoria.")

def get_batch_data() -> Optional[pd.DataFrame]:
    if os.path.exists(PARQUET_PATH):
        return pd.read_parquet(PARQUET_PATH)
    return None

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
    df_batch = get_batch_data()
    return {
        "status": "online",
        "model_loaded": model_pipeline is not None,
        "batch_records_available": len(df_batch) if df_batch is not None else 0
    }

# 1. Endpoint legado de lookup (batch)
@app.get("/results/{customer_id}")
def get_batch_result(customer_id: str):
    df_batch = get_batch_data()
    if df_batch is None:
        raise HTTPException(status_code=503, detail="Base de batch predictions nao carregada. Execute src.batch_scoring.")
    
    match = df_batch[df_batch["customer_id"] == customer_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Cliente '{customer_id}' nao localizado na base pre-computada.")
    return match.iloc[0].to_dict()

# 2. Endpoint de inferencia em tempo real (online)
@app.post("/predict")
def predict_churn(customer: CustomerPayload):
    data = customer.dict()
    
    if data["TotalCharges"] is None or data["TotalCharges"] <= 0:
        data["TotalCharges"] = float(data["tenure"] * data["MonthlyCharges"])
    
    data["gasto_por_mes"] = float(data["TotalCharges"] / max(data["tenure"], 1))
    
    cat_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    
    df_input = pd.DataFrame([data])
    for c in cat_cols:
        df_input[c] = df_input[c].astype(str)
        
    try:
        prob = float(model_pipeline.predict_proba(df_input)[0, 1])
        pred = int(prob >= 0.5)
        
        risk = "Alto" if prob >= 0.6 else ("Medio" if prob >= 0.3 else "Baixo")
            
        return {
            "churn_prediction": pred,
            "churn_probability": round(prob, 4),
            "risk_level": risk,
            "inference_type": "real_time_online",
            "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferencia do modelo: {str(e)}")
