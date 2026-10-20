import matplotlib
matplotlib.use('Agg')  # Evita travamento da GUI/Tkinter durante autologging

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.data import load_clean_data
from src.features import process_features
from src.train_mlflow import get_dvc_md5, get_git_commit

def run_grid():
    # Conecta ao mesmo banco SQLite utilizado pela API FastAPI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.sklearn.autolog()
    mlflow.set_experiment("churn-prediction")

    print("Carregando e processando dados...")
    df = load_clean_data()
    X, y = process_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_estimators_list = [100, 200, 500]
    max_depth_list = [None, 8, 16]

    for n in n_estimators_list:
        for depth in max_depth_list:
            run_name = f"rf-n{n}-d{depth}"
            print(f"Executando run: {run_name}")

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({
                    "data_md5": get_dvc_md5(),
                    "git_commit": get_git_commit()
                })

                model = RandomForestClassifier(
                    n_estimators=n, max_depth=depth, random_state=42
                )
                model.fit(X_train, y_train)

                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                mlflow.log_metric("test_accuracy", acc)

if __name__ == "__main__":
    run_grid()
