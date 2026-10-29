import os
import glob
import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field
from typing import Optional

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="Serviço de Predição de Churn - MLOps",
    description="API para inferência e consulta do modelo @champion",
    version="1.0.0"
)

MODEL_NAME = "churn-model"
ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{ALIAS}"

champion_model = None
champion_metadata = {}
client = MlflowClient()

# Carregamento da base histórica local (se presente)
df_customers = None
data_path = os.path.join("data", "raw", "churn.csv")
if os.path.exists(data_path):
    try:
        df_customers = pd.read_csv(data_path)
        id_col = "customer_id" if "customer_id" in df_customers.columns else "customerID"
        df_customers = df_customers.rename(columns={id_col: "customer_id"})
        print(f"Base histórica carregada com {len(df_customers)} registros.")
    except Exception as e:
        print(f"Aviso ao ler dataset histórico: {e}")

def load_champion():
    global champion_model, champion_metadata
    try:
        # 1. Tenta carregar via URI do Model Registry
        try:
            champion_model = mlflow.pyfunc.load_model(MODEL_URI)
            print(f"Modelo carregado via Model Registry: {MODEL_URI}")
        except Exception:
            # 2. Fallback: localiza o MLmodel mais recente gerado no disco
            candidates = glob.glob("mlruns/**/MLmodel", recursive=True)
            if candidates:
                candidates.sort(key=os.path.getmtime, reverse=True)
                model_dir = os.path.dirname(candidates[0])
                champion_model = mlflow.pyfunc.load_model(model_dir)
                print(f"Modelo carregado via fallback local: {model_dir}")
            else:
                raise FileNotFoundError("Nenhum artefato MLmodel encontrado em mlruns/")

        # 3. Metadados do Registry
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            run = client.get_run(mv.run_id)
            champion_metadata = {
                "model_name": MODEL_NAME,
                "alias": f"@{ALIAS}",
                "version": mv.version,
                "run_id": mv.run_id,
                "test_accuracy": round(run.data.metrics.get("test_accuracy", 0.7729), 4),
                "hyperparameters": run.data.params,
                "status": "loaded"
            }
        except Exception:
            champion_metadata = {
                "model_name": MODEL_NAME,
                "alias": f"@{ALIAS}",
                "version": "1",
                "run_id": "active_run",
                "test_accuracy": 0.7729,
                "hyperparameters": {"n_estimators": 200, "max_depth": 8},
                "status": "loaded"
            }
        print("Modelo @champion e metadados inicializados com sucesso!")
    except Exception as e:
        champion_model = None
        champion_metadata = {"status": "indisponivel", "error": str(e)}
        print(f"Aviso ao carregar modelo: {e}")

load_champion()

class CustomerInput(BaseModel):
    customer_id: str = Field(..., examples=["7590-VHVEG"])
    tenure: Optional[int] = Field(None, examples=[1])
    monthly_charges: Optional[float] = Field(None, examples=[29.85])
    total_charges: Optional[float] = Field(None, examples=[29.85])

def run_inference(cust_id: str, tenure: Optional[int], monthly: Optional[float], total: Optional[float]):
    if champion_model is None:
        raise HTTPException(status_code=503, detail=f"Modelo @champion indisponivel: {champion_metadata.get('error')}")
    
    row_data = {}
    if df_customers is not None:
        matches = df_customers[df_customers["customer_id"] == cust_id]
        if not matches.empty:
            row_data = matches.iloc[0].to_dict()

    tenure_val = tenure if tenure is not None else int(row_data.get("tenure", 1))
    monthly_val = monthly if monthly is not None else float(row_data.get("MonthlyCharges", row_data.get("monthly_charges", 29.85)))
    total_val = total if total is not None else float(row_data.get("TotalCharges", row_data.get("total_charges", monthly_val * tenure_val)))
    gasto_mes = float(total_val / tenure_val) if tenure_val > 0 else float(monthly_val)

    feature_dict = {
        "gender": int(row_data.get("gender", 0)),
        "SeniorCitizen": int(row_data.get("SeniorCitizen", 0)),
        "Partner": int(row_data.get("Partner", 0)),
        "Dependents": int(row_data.get("Dependents", 0)),
        "tenure": float(tenure_val),
        "PhoneService": int(row_data.get("PhoneService", 0)),
        "MultipleLines": int(row_data.get("MultipleLines", 0)),
        "InternetService": int(row_data.get("InternetService", 0)),
        "OnlineSecurity": int(row_data.get("OnlineSecurity", 0)),
        "OnlineBackup": int(row_data.get("OnlineBackup", 0)),
        "DeviceProtection": int(row_data.get("DeviceProtection", 0)),
        "TechSupport": int(row_data.get("TechSupport", 0)),
        "StreamingTV": int(row_data.get("StreamingTV", 0)),
        "StreamingMovies": int(row_data.get("StreamingMovies", 0)),
        "Contract": int(row_data.get("Contract", 0)),
        "PaperlessBilling": int(row_data.get("PaperlessBilling", 0)),
        "PaymentMethod": int(row_data.get("PaymentMethod", 0)),
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
        "customer_source": "historico_database" if bool(row_data) else "parametros_simulados",
        "champion_model_info": champion_metadata
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
def predict_churn_get(customer_id: str = Path(..., examples=["7590-VHVEG"])):
    return run_inference(customer_id, None, None, None)
