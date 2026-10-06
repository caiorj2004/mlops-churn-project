import mlflow
import pandas as pd
from src.data import load_clean_data
from src.features import process_features

def test_production_model():
    print("1. Carregando o modelo de produção (@champion)...")
    # Consome o modelo pelo contrato universal (pyfunc)
    model = mlflow.pyfunc.load_model("models:/churn-model@champion")

    print("2. Simulando novos dados de clientes...")
    df = load_clean_data()
    X, _ = process_features(df)
    
    # Isola 3 clientes fictícios
    clientes_novos = X.head(3)

    print("3. Executando inferência...")
    previsoes = model.predict(clientes_novos)
    
    print("\nResultados da previsão de Churn (0 = Não, 1 = Sim):")
    for i, prev in enumerate(previsoes):
        print(f"Cliente {i+1}: {prev}")

if __name__ == "__main__":
    test_production_model()