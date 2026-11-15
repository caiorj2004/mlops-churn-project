"""Engenharia de atributos do projeto de churn.

Fonte unica da transformacao usada no treino (MLflow) e na inferencia (APIs).
O "contrato de features" (mapeamento das categorias + ordem das colunas) e
salvo junto com o modelo no MLflow, para que a inferencia codifique um cliente
exatamente como o treino codificou.
"""
import pandas as pd

TARGET = "Churn"

# Constantes de normalizacao (aproximadamente o maximo de cada coluna na base)
TENURE_MAX = 72.0
MONTHLY_CHARGES_MAX = 118.0
TOTAL_CHARGES_MAX = 8600.0


def numeric_features(tenure, monthly_charges, total_charges) -> dict:
    """Engenharia e normalizacao das colunas numericas (aceita escalares ou Series)."""
    return {
        "tenure": tenure / TENURE_MAX,
        "MonthlyCharges": monthly_charges / MONTHLY_CHARGES_MAX,
        "TotalCharges": total_charges / TOTAL_CHARGES_MAX,
        # calculada com os valores BRUTOS, antes da normalizacao
        "gasto_por_mes": total_charges / (tenure + 1),
    }


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza + features numericas (antes do encoding das categorias)."""
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Trata conversão e nulos de TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Criação de feature nova + normalização das colunas numéricas
    for coluna, valores in numeric_features(df["tenure"], df["MonthlyCharges"], df["TotalCharges"]).items():
        df[coluna] = valores

    # Limpa valores nulos remanescentes
    return df.dropna()


def build_feature_contract(df: pd.DataFrame) -> dict:
    """Contrato de features aprendido no dado de treino.

    category_mappings: para cada coluna de texto, as categorias em ordem
    alfabetica (a mesma ordem que o LabelEncoder usa: codigo = posicao na lista).
    feature_columns: colunas de entrada do modelo, na ordem do treino.
    """
    df = _clean(df.copy())
    categoricas = df.select_dtypes(include=["object", "string"]).columns
    mappings = {c: sorted(df[c].astype(str).unique().tolist()) for c in categoricas}
    feature_columns = [c for c in df.columns if c != TARGET]
    return {"category_mappings": mappings, "feature_columns": feature_columns}


def encode_categories(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Aplica o contrato: troca cada categoria pelo seu codigo. Categoria desconhecida = erro."""
    for coluna, categorias in mappings.items():
        if coluna not in df.columns:
            continue
        codigos = {valor: i for i, valor in enumerate(categorias)}
        valores = df[coluna].astype(str)
        desconhecidos = sorted(set(valores) - set(codigos))
        if desconhecidos:
            raise ValueError(f"Categoria desconhecida em '{coluna}': {desconhecidos}. Validas: {categorias}")
        df[coluna] = valores.map(codigos).astype("int64")
    return df


def process_features(df: pd.DataFrame):
    """Treino: limpeza, engenharia de atributos, normalizacao e encoding."""
    contrato = build_feature_contract(df)
    df = encode_categories(_clean(df), contrato["category_mappings"])

    # Separação entre Variável Alvo (y) e Features (X)
    y = df[TARGET]
    X = df[contrato["feature_columns"]]
    return X, y


def features_for_inference(registros: pd.DataFrame, contrato: dict) -> pd.DataFrame:
    """Inferencia: transforma registros brutos (do banco ou do formulario) no formato do treino.

    `registros` tem as colunas do churn.csv (sem o alvo). TotalCharges vazio vira 0,
    como em src/data.py, onde o dado e limpo antes do treino.
    """
    df = registros.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="raise").astype("int64")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="raise").astype(float)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="raise").astype("int64")
    for coluna, valores in numeric_features(df["tenure"], df["MonthlyCharges"], df["TotalCharges"]).items():
        df[coluna] = valores
    mapeamentos = {c: v for c, v in contrato["category_mappings"].items() if c != TARGET}
    df = encode_categories(df, mapeamentos)
    return df[contrato["feature_columns"]]
