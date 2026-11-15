"""Quebra de proposito do contrato de dados (Encontro 4).

Pega 3 linhas REAIS do churn.csv (ja validas), adultera algumas celulas e
confere que o ChurnSchema (pandera) barra exatamente essas adulteracoes.
"""
from pandera.errors import SchemaErrors

from src.data import load_clean_data
from src.schema import ChurnSchema

print("=== TESTE DE QUEBRA DO CONTRATO DE DADOS (PANDERA) ===")

df = load_clean_data().head(3).copy()  # 3 clientes reais, ja aprovados pelo contrato
print("\nLinhas reais usadas no teste:")
print(df[["customerID", "tenure", "MonthlyCharges", "Contract", "Churn"]].to_string(index=False))

# Adulteracoes propositais (cada uma viola uma regra do ChurnSchema)
df.loc[df.index[0], "tenure"] = 999                       # viola le=72
df.loc[df.index[1], "MonthlyCharges"] = -15.0             # viola ge=18
df.loc[df.index[2], "Contract"] = "Vitalicio"             # viola isin([...])
df.loc[df.index[2], "customerID"] = df.loc[df.index[0], "customerID"]  # viola unique=True

print("\nValidando o lote adulterado (lazy=True coleta todas as falhas)...")
try:
    ChurnSchema.validate(df, lazy=True)
    print("ERRO: o pandera deixou passar dados invalidos.")
except SchemaErrors as e:
    falhas = e.failure_cases[["column", "check", "failure_case"]]
    print("\nSUCESSO: o contrato barrou o lote. Falhas encontradas:")
    print(falhas.to_string(index=False))
