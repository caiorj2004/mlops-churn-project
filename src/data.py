import pandas as pd
from src.config import DATA_PATH
from src.schema import ChurnSchema


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset bruto de churn."""
    df = pd.read_csv(path)
    return df


def load_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset, realiza tratamento básico de tipos e valida contra o ChurnSchema."""
    df = load_data(path)

    # Tratamento pré-validação (garante que TotalCharges seja numérico e sem NaNs)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Validação do Contrato (lazy=True acumula todas as falhas antes de estourar a exceção)
    df_validado = ChurnSchema.validate(df, lazy=True)

    return df_validado