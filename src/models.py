from sklearn.ensemble import RandomForestClassifier
from src.config import N_ESTIMATORS

def train_model(X_train, y_train):
    """Instancia e treina o modelo RandomForest."""
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS)
    model.fit(X_train, y_train)
    return model