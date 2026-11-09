import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

RAW_DATA_PATH = os.path.join("data", "raw", "churn.csv")
MODEL_OUTPUT_PATH = os.path.join("models", "churn_pipeline.joblib")

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Dataset nao encontrado em: {RAW_DATA_PATH}")

print(f"--> Carregando dados de {RAW_DATA_PATH}...")
df = pd.read_csv(RAW_DATA_PATH)
id_col = "customer_id" if "customer_id" in df.columns else "customerID"
df = df.rename(columns={id_col: "customer_id"})

# Tratamento e tipagem numerica
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce").fillna(0.0)
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(1)
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
df["gasto_por_mes"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

num_features = ["tenure", "MonthlyCharges", "TotalCharges", "gasto_por_mes"]
cat_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]

for col in cat_features:
    df[col] = df[col].astype(str)

X = df[num_features + cat_features]
y = (df["Churn"] == "Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline de pre-processamento e modelo
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42))
])

print("--> Treinando RandomForestClassifier...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(f"--> Acuracia no teste: {accuracy_score(y_test, y_pred):.4f}")
print(f"--> ROC-AUC no teste: {roc_auc_score(y_test, y_prob):.4f}")

os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_OUTPUT_PATH)
print(f"--> Pipeline exportado com sucesso em: {MODEL_OUTPUT_PATH}")
