import pandera as pa
from pandera.typing import Series


class ChurnSchema(pa.DataFrameModel):
    """Contrato de dados (Schema) para o dataset de Churn utilizando Pandera."""

    tenure: Series[int] = pa.Field(ge=0, le=72)
    MonthlyCharges: Series[float] = pa.Field(ge=0.0)
    TotalCharges: Series[float] = pa.Field(ge=0.0)
    Contract: Series[str] = pa.Field(
        isin=["Month-to-month", "One year", "Two year"]
    )
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config:
        coerce = True  # Converte tipos de dados automaticamente se possível
        strict = False  # Permite que colunas adicionais permaneçam no DataFrame