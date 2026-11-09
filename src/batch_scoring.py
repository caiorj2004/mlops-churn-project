import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

RAW_DATA_PATH = os.path.join("data", "raw", "churn.csv")
OUTPUT_PARQUET = os.path.join("data", "batch_predictions.parquet")

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Dataset nao encontrado em: {RAW_DATA_PATH}")

print(f"Carregando dataset de {RAW_DATA_PATH}...")
df = pd.read_csv(RAW_DATA_PATH)
id_col = "customer_id" if "customer_id" in df.columns else "customerID"
df = df.rename(columns={id_col: "customer_id"})

# Tratamento e tipagem real dos dados
df_proc = df.copy()
df_proc["TotalCharges"] = pd.to_numeric(df_proc["TotalCharges"].replace(" ", np.nan), errors="coerce").fillna(0.0)
df_proc["tenure"] = pd.to_numeric(df_proc["tenure"], errors="coerce").fillna(1)
df_proc["MonthlyCharges"] = pd.to_numeric(df_proc["MonthlyCharges"], errors="coerce").fillna(0.0)
df_proc["gasto_por_mes"] = df_proc["TotalCharges"] / df_proc["tenure"].replace(0, 1)

cols_categoricas = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]
for col in cols_categoricas:
    if col in df_proc.columns:
        df_proc[col] = df_proc[col].astype("category").cat.codes

features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges", "gasto_por_mes"
]

X = df_proc[features]
y = (df["Churn"] == "Yes").astype(int)

# Particionamento e treinamento do modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Modelo treinado com sucesso. Acuracia real de teste: {acc:.4f}")

# Inferencia em lote e extracao de probabilidades autenticas (predict_proba)
probs = model.predict_proba(X)[:, 1]
preds = (probs >= 0.5).astype(int)

# Classificacao de risco baseada no calculo real da probabilidade
def calcular_nivel_risco(p):
    if p >= 0.6:
        return "Alto"
    elif p >= 0.3:
        return "Medio"
    return "Baixo"

risk_levels = [calcular_nivel_risco(p) for p in probs]

# Consolidacao da base final de consulta em lote
df_batch = pd.DataFrame({
    "customer_id": df["customer_id"],
    "churn_prediction": preds,
    "churn_probability": np.round(probs, 4),
    "risk_level": risk_levels,
    "contract_type": df["Contract"],
    "monthly_charges": df["MonthlyCharges"],
    "total_charges": df_proc["TotalCharges"],
    "tenure": df["tenure"],
    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
})

os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
df_batch.to_parquet(OUTPUT_PARQUET, index=False)
print(f"Sucesso! {len(df_batch)} clientes pontuados e salvos em {OUTPUT_PARQUET}.")
