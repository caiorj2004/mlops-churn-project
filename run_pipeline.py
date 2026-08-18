from src.data import load_data
from src.features import process_features
from src.models import train_model
from src.evaluate import evaluate_model, save_model
from src.config import TEST_SIZE
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # 1. Carrega
    df_raw = load_data()
    
    # 2. Processa features
    X, y = process_features(df_raw)
    
    # 3. Divide treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    
    # 4. Treina
    model = train_model(X_train, y_train)
    
    # 5. Avalia e Salva
    evaluate_model(model, X_test, y_test)
    save_model(model)
