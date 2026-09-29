import mlflow
from mlflow.tracking import MlflowClient

def promote_best_model():
    # 1. Busca todos os runs do experimento, ordenando pela melhor acurácia no teste
    runs = mlflow.search_runs(
        experiment_names=["churn-prediction"], 
        order_by=["metrics.test_accuracy DESC"]
    )
    
    if runs.empty:
        print("Nenhum run encontrado. Execute o tuning primeiro.")
        return

    # 2. Isola o melhor run (a primeira linha do DataFrame pandas)
    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    best_acc = best_run["metrics.test_accuracy"]
    
    print(f"Melhor modelo encontrado! Run ID: {best_run_id} | Acurácia: {best_acc:.4f}")

    # 3. Registra o modelo vencedor no Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, "churn-model")
    
    # 4. Promove a versão registrada ao alias de produção (@champion)
    client = MlflowClient()
    client.set_registered_model_alias("churn-model", "champion", mv.version)
    
    print(f"Modelo registrado como 'churn-model' (Versão {mv.version})")
    print("Alias '@champion' aplicado com sucesso! Pronto para deploy.")

if __name__ == "__main__":
    promote_best_model()