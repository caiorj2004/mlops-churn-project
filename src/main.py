import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="Serviço de Predição de Churn - MLOps",
    description="API para inferência e consulta do modelo @champion",
    version="1.0.0"
)

MODEL_URI = "models:/churn-model@champion"
try:
    champion_model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"Modelo @champion carregado de {MODEL_URI}")
except Exception as e:
    champion_model = None
    print(f"Aviso ao carregar modelo: {e}")

class CustomerInput(BaseModel):
    customer_id: str = Field(..., example="7590-VHVEG")
    tenure: int = Field(..., example=1)
    monthly_charges: float = Field(..., example=29.85)
    total_charges: float = Field(..., example=29.85)

@app.get("/")
def read_root():
    return {"status": "online", "message": "API MLOps ativa. Acesse /docs"}

@app.get("/model-info")
def get_model_info():
    if champion_model is None:
        raise HTTPException(status_code=503, detail="Modelo @champion indisponível.")
    return {"model_name": "churn-model", "alias": "@champion", "uri": MODEL_URI, "status": "loaded"}

@app.post("/predict")
def predict_churn(data: CustomerInput):
    if champion_model is None:
        raise HTTPException(status_code=503, detail="Modelo @champion não carregado.")
    
    try:
        gasto_mes = float(data.total_charges / data.tenure) if data.tenure > 0 else float(data.monthly_charges)
        
        feature_dict = {
            "gender": 0,
            "SeniorCitizen": 0,
            "Partner": 0,
            "Dependents": 0,
            "tenure": float(data.tenure),
            "PhoneService": 0,
            "MultipleLines": 0,
            "InternetService": 0,
            "OnlineSecurity": 0,
            "OnlineBackup": 0,
            "DeviceProtection": 0,
            "TechSupport": 0,
            "StreamingTV": 0,
            "StreamingMovies": 0,
            "Contract": 0,
            "PaperlessBilling": 0,
            "PaymentMethod": 0,
            "MonthlyCharges": float(data.monthly_charges),
            "TotalCharges": float(data.total_charges),
            "gasto_por_mes": gasto_mes
        }

        df_input = pd.DataFrame([feature_dict])
        prediction = champion_model.predict(df_input)
        churn_result = int(prediction[0])

        return {
            "customer_id": data.customer_id,
            "churn_prediction": churn_result,
            "label": "Churn" if churn_result == 1 else "No Churn",
            "champion_model_info": {
                "model_name": "churn-model",
                "alias": "@champion",
                "uri": MODEL_URI,
                "status": "active"
            }
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {str(err)}")
