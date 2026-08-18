import pickle
from sklearn.metrics import accuracy_score
from src.config import MODEL_PATH

def evaluate_model(model, X_test, y_test):
    """Calcula e exibe as métricas de avaliação do modelo."""
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Acurácia do modelo: {acc:.4f}")
    return acc

def save_model(model, path=MODEL_PATH):
    """Serializa e salva o modelo treinado em disco."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo salvo com sucesso em: {path}")