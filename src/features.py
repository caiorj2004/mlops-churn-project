import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Constantes de normalizacao (aproximadamente o maximo de cada coluna na base)
TENURE_MAX = 72.0
MONTHLY_CHARGES_MAX = 118.0
TOTAL_CHARGES_MAX = 8600.0


def numeric_features(tenure: float, monthly_charges: float, total_charges: float) -> dict:
    """Engenharia e normalizacao das colunas numericas.

    Fonte unica da regra usada no treino (process_features) e na API (src/main.py),
    para que a inferencia receba as features exatamente no formato do treino.
    """
    return {
        "tenure": tenure / TENURE_MAX,
        "MonthlyCharges": monthly_charges / MONTHLY_CHARGES_MAX,
        "TotalCharges": total_charges / TOTAL_CHARGES_MAX,
        # calculada com os valores BRUTOS, antes da normalizacao
        "gasto_por_mes": total_charges / (tenure + 1),
    }


def process_features(df: pd.DataFrame):
    """Executa a limpeza, engenharia de atributos e normalização."""
    # Remove coluna irrelevante
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Trata conversão e nulos de TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Criação de feature nova + normalização das colunas numéricas
    numericas = numeric_features(df["tenure"], df["MonthlyCharges"], df["TotalCharges"])
    for coluna, valores in numericas.items():
        df[coluna] = valores

    # Limpa valores nulos remanescentes
    df = df.dropna()

    # Codifica TODAS as variáveis categóricas (object) para numérico
    le = LabelEncoder()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = le.fit_transform(df[c])

    # Separação entre Variável Alvo (y) e Features (X)
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    return X, y
