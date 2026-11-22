"""Fixtures compartilhadas: dado real (DVC), um @champion de teste e a API ligada a um SQLite.

Os testes nao dependem da AWS, do PostgreSQL nem do mlflow.db local: o modelo de teste
e treinado aqui (Random Forest pequeno) com as mesmas funcoes do pipeline.
"""
import importlib
import json
import sqlite3

import mlflow.sklearn
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.config import DATA_PATH
from src.data import load_clean_data
from src.features import build_feature_contract, process_features


@pytest.fixture(scope="session")
def dados():
    if not DATA_PATH.exists():
        pytest.skip("data/raw/churn.csv ausente: rode `dvc pull` (remote `repositorio`, pasta dvc-publico/).")
    return load_clean_data()


@pytest.fixture(scope="session")
def champion_dir(dados, tmp_path_factory):
    """Pasta no mesmo formato de models/champion/ (src/export_champion.py)."""
    destino = tmp_path_factory.mktemp("champion") / "champion"
    X, y = process_features(dados)
    modelo = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=42).fit(X, y)
    mlflow.sklearn.save_model(modelo, str(destino), serialization_format="pickle")
    (destino / "feature_contract.json").write_text(json.dumps(build_feature_contract(dados)), encoding="utf-8")
    (destino / "champion_info.json").write_text(json.dumps({
        "model_name": "churn-model", "alias": "@champion", "version": 0, "run_name": "teste",
        "test_accuracy": None, "test_roc_auc": None}), encoding="utf-8")
    return destino


@pytest.fixture(scope="session")
def cliente_api(champion_dir, tmp_path_factory):
    """TestClient da API v2 com o banco trocado por um SQLite carregado do churn.csv (mesma tabela)."""
    from fastapi.testclient import TestClient

    banco = tmp_path_factory.mktemp("db") / "churn.db"
    with sqlite3.connect(banco) as con:
        pd.read_csv(DATA_PATH, dtype={"TotalCharges": str}).to_sql("customers", con, index=False)

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", f"sqlite:///{banco.as_posix()}")
    mp.setenv("CHAMPION_DIR", str(champion_dir))
    import src.db
    import src.inference
    src.db.get_engine.cache_clear()
    importlib.reload(src.inference)
    import src.api_v2
    importlib.reload(src.api_v2)
    yield TestClient(src.api_v2.app)
    mp.undo()
    src.db.get_engine.cache_clear()
