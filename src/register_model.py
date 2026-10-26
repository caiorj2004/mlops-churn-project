import mlflow
from mlflow.tracking import MlflowClient

def promote_best_model():
    # Conecta ao banco SQLite compartilhado
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    runs = mlflow.search_runs(
        experiment_names=["churn-prediction"],
        order_by=["metrics.test_accuracy DESC"]
    )

    if runs.empty:
        print("Nenhum run encontrado. Execute o tuning primeiro.")
        return

    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    best_acc = best_run["metrics.test_accuracy"]

    print(f"Melhor modelo encontrado! Run ID: {best_run_id} | Acurácia: {best_acc:.4f}")

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, "churn-model")

    client = MlflowClient()
    client.set_registered_model_alias("churn-model", "champion", mv.version)

    print(f"Modelo registrado como 'churn-model' (Versão {mv.version})")
    print("Alias '@champion' aplicado com sucesso! Pronto para deploy.")

if __name__ == "__main__":
    promote_best_model()
