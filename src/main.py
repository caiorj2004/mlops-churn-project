import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field
from typing import Optional

from src.data import load_clean_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="Serviço de Predição de Churn - MLOps",
    description="API para inferência e consulta do modelo @champion com busca por ID",
    version="1.0.0"
)

MODEL_NAME = "churn-model"
ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{ALIAS}"

champion_model = None
champion_metadata = {}
client = MlflowClient()

# Carrega a base limpa em memória para busca automática de clientes por ID
try:
    df_customers = load_clean_data()
    id_col = "customer_id" if "customer_id" in df_customers.columns else "customerID"
    df_customers = df_customers.rename(columns={id_col: "customer_id"})
    print(f"Base de clientes carregada com {len(df_customers)} registros.")
except Exception as e:
    df_customers = None
    print(f"Aviso ao carregar dataset histórico: {e}")

def load_champion():
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
            "test_accuracy": round(run.data.metrics.get("test_accuracy", 0.0), 4),
            "hyperparameters": run.data.params,
            "status": "loaded"
        }
        print(f"Modelo @champion carregado com sucesso (Versão {mv.version})")
    except Exception as e:
        champion_model = None
        champion_metadata = {"status": "indisponível", "error": str(e)}
        print(f"Aviso ao carregar modelo @champion: {e}")

load_champion()

class CustomerInput(BaseModel):
    customer_id: str = Field(..., example="7590-VHVEG")
    tenure: Optional[int] = Field(None, example=1)
    monthly_charges: Optional[float] = Field(None, example=29.85)
    total_charges: Optional[float] = Field(None, example=29.85)

def run_inference(cust_id: str, tenure: Optional[int], monthly: Optional[float], total: Optional[float]):
    if champion_model is None:
        raise HTTPException(status_code=503, detail="Modelo @champion não carregado.")
    
    # 1. Verifica se o cliente existe no dataset histórico
    row_data = None
    if df_customers is not None:
        matches = df_customers[df_customers["customer_id"] == cust_id]
        if not matches.empty:
            row_data = matches.iloc[0].to_dict()

    # Se não passou parâmetros e não encontrou no dataset
    if row_data is None and (tenure is None or monthly is None or total is None):
        raise HTTPException(
            status_code=404, 
            detail=f"Cliente '{cust_id}' não encontrado na base e atributos não foram informados."
        )

    # 2. Preenche variáveis (usa os parâmetros manuais se passados, senão usa os da base)
    tenure_val = tenure if tenure is not None else int(row_data.get("tenure", 1))
    monthly_val = monthly if monthly is not None else float(row_data.get("MonthlyCharges", row_data.get("monthly_charges", 29.85)))
    total_val = total if total is not None else float(row_data.get("TotalCharges", row_data.get("total_charges", monthly_val)))
    gasto_mes = float(total_val / tenure_val) if tenure_val > 0 else float(monthly_val)

    # 3. Monta o DataFrame com as 20 features exigidas pela assinatura do MLflow
    feature_dict = {
        "gender": int(row_data.get("gender", 0)) if row_data else 0,
        "SeniorCitizen": int(row_data.get("SeniorCitizen", 0)) if row_data else 0,
        "Partner": int(row_data.get("Partner", 0)) if row_data else 0,
        "Dependents": int(row_data.get("Dependents", 0)) if row_data else 0,
        "tenure": float(tenure_val),
        "PhoneService": int(row_data.get("PhoneService", 0)) if row_data else 0,
        "MultipleLines": int(row_data.get("MultipleLines", 0)) if row_data else 0,
        "InternetService": int(row_data.get("InternetService", 0)) if row_data else 0,
        "OnlineSecurity": int(row_data.get("OnlineSecurity", 0)) if row_data else 0,
        "OnlineBackup": int(row_data.get("OnlineBackup", 0)) if row_data else 0,
        "DeviceProtection": int(row_data.get("DeviceProtection", 0)) if row_data else 0,
        "TechSupport": int(row_data.get("TechSupport", 0)) if row_data else 0,
        "StreamingTV": int(row_data.get("StreamingTV", 0)) if row_data else 0,
        "StreamingMovies": int(row_data.get("StreamingMovies", 0)) if row_data else 0,
        "Contract": int(row_data.get("Contract", 0)) if row_data else 0,
        "PaperlessBilling": int(row_data.get("PaperlessBilling", 0)) if row_data else 0,
        "PaymentMethod": int(row_data.get("PaymentMethod", 0)) if row_data else 0,
        "MonthlyCharges": float(monthly_val),
        "TotalCharges": float(total_val),
        "gasto_por_mes": gasto_mes
    }

    df_input = pd.DataFrame([feature_dict])
    prediction = champion_model.predict(df_input)
    churn_result = int(prediction[0])

    return {
        "customer_id": cust_id,
        "churn_prediction": churn_result,
        "label": "Churn" if churn_result == 1 else "No Churn",
        "customer_source": "historico_database" if row_data is not None else "payload_manual",
        "champion_model_info": champion_metadata
    }

@app.get("/")
def read_root():
    return {"status": "online", "message": "API MLOps ativa. Acesse /docs"}

@app.get("/model-info")
def get_model_info():
    if champion_model is None:
        raise HTTPException(status_code=503, detail="Modelo @champion indisponível.")
    return champion_metadata

@app.post("/predict")
def predict_churn_post(data: CustomerInput):
    return run_inference(data.customer_id, data.tenure, data.monthly_charges, data.total_charges)

@app.get("/predict/{customer_id}")
def predict_churn_get(customer_id: str = Path(..., example="7590-VHVEG")):
    """Permite fazer predição fornecendo apenas o código do cliente na URL."""
    return run_inference(customer_id, None, None, None)
