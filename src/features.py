import pandas as pd
from sklearn.preprocessing import LabelEncoder

def process_features(df: pd.DataFrame):
    """Executa a limpeza, engenharia de atributos e normalização."""
    # Remove coluna irrelevante
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Trata conversão e nulos de TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(2200)

    # Criação de feature nova
    df["gasto_por_mes"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Limpa valores nulos remanescentes
    df = df.dropna()

    # Codifica variáveis categóricas para numérico
    le = LabelEncoder()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = le.fit_transform(df[c])

    # Normalização das colunas numéricas
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = df["MonthlyCharges"] / 118.0
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"] / 8600.0
    if "tenure" in df.columns:
        df["tenure"] = df["tenure"] / 72.0

    # Separação entre Variável Alvo (y) e Features (X)
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    return X, y