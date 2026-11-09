import os
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Servico de Consulta em Lote - Churn",
    description="Endpoint de lookup para resultados pre-computados (sem inferencia ativa)"
)

RESULTS_PATH = os.path.join("data", "batch_predictions.parquet")

if not os.path.exists(RESULTS_PATH):
    raise FileNotFoundError(f"Arquivo {RESULTS_PATH} nao encontrado. O batch scoring precisa ser executado.")

df_batch = pd.read_parquet(RESULTS_PATH)
print(f"Base de batch scoring carregada com sucesso: {len(df_batch)} registros autenticos.")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "records_available": len(df_batch),
        "source": RESULTS_PATH
    }

@app.get("/results/{customer_id}")
def get_batch_result(customer_id: str):
    match = df_batch[df_batch["customer_id"] == customer_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Cliente '{customer_id}' nao encontrado na base.")
    return match.iloc[0].to_dict()
