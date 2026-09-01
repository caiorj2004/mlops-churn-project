import pandas as pd
from src.schema import ChurnSchema

print("=== INICIANDO TESTE DE VALIDAÇÃO DE DADOS (PANDERA) ===")

# Criando um DataFrame propositalmente corrompido para testar o contrato
df_corrompido = pd.DataFrame(
    {
        "tenure": [12, 999],  # ERRO 1: 999 violará le=72
        "MonthlyCharges": [50.0, -15.0],  # ERRO 2: -15.0 violará ge=0
        "TotalCharges": [600.0, 1000.0],
        "Contract": ["Month-to-month", "Vitalicio"],  # ERRO 3: 'Vitalicio' violará isin
        "Churn": ["No", "Yes"],
    }
)

print("\nTentando validar lote de dados corrompidos...")

try:
    ChurnSchema.validate(df_corrompido, lazy=True)
    print("❌ ERRO: O Pandera falhou em barrar os dados corrompidos.")
except Exception as e:
    print("\n✅ SUCESSO: O Pandera capturou a quebra do contrato de dados.\n")
    print(e)