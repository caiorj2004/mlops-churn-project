from sklearn.model_selection import train_test_split
from src.data import load_clean_data
from src.evaluate import evaluate_model
from src.features import process_features
from src.models import train_model


def run():
    print("1. Carregando e validando dados com Pandera...")
    df = load_clean_data()

    print("2. Processando features...")
    X, y = process_features(df)

    # Separação Treino e Teste (Evita Overfitting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("3. Treinando modelo...")
    model = train_model(X_train, y_train)

    print("4. Avaliando modelo no conjunto de TESTE...")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    run()