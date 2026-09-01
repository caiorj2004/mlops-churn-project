import pandera as pa
from pandera.typing import Series


class ChurnSchema(pa.DataFrameModel):
    """Contrato de dados (Schema) para o dataset de Churn utilizando Pandera."""

    # Unicidade do identificador
    customerID: Series[str] = pa.Field(unique=True)

    # Tipos e Faixas numéricas
    tenure: Series[int] = pa.Field(ge=0, le=72)
    MonthlyCharges: Series[float] = pa.Field(ge=18.0, le=120.0)
    TotalCharges: Series[float] = pa.Field(ge=0.0)

    # Categorias válidas
    Contract: Series[str] = pa.Field(
        isin=["Month-to-month", "One year", "Two year"]
    )
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config:
        coerce = True  # Converte tipos de dados automaticamente se possível
        strict = False  # Permite que colunas não mapeadas permaneçam no DataFrame