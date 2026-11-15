"""Scoring dos apps da AWS com o modelo @champion exportado do MLflow.

O modelo e o contrato de features vem de models/champion/ (gerado por
`python -m src.export_champion`). A transformacao dos dados e a mesma do
treino (src/features.py), entao a previsao de um cliente aqui e identica a
do modelo aplicado na base de treino.
"""
import json
from datetime import datetime
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from src.features import features_for_inference

CHAMPION_DIR = Path("models") / "champion"


class ChampionModel:
    def __init__(self, model_dir: Path = CHAMPION_DIR):
        if not (model_dir / "MLmodel").exists():
            raise FileNotFoundError(
                f"Modelo @champion nao encontrado em {model_dir}. Rode `python -m src.export_champion` primeiro."
            )
        # flavor sklearn: da acesso ao predict_proba (o pyfunc so devolve a classe)
        self.model = mlflow.sklearn.load_model(str(model_dir))
        self.contract = json.loads((model_dir / "feature_contract.json").read_text(encoding="utf-8"))
        self.info = json.loads((model_dir / "champion_info.json").read_text(encoding="utf-8"))

    def features(self, registros: pd.DataFrame) -> pd.DataFrame:
        return features_for_inference(registros, self.contract)

    def predict_proba(self, registros: pd.DataFrame):
        return self.model.predict_proba(self.features(registros))[:, 1]

    def score_one(self, registro: dict) -> dict:
        prob = float(self.predict_proba(pd.DataFrame([registro]))[0])
        return {
            "churn_prediction": int(prob >= 0.5),
            "churn_probability": round(prob, 4),
            "risk_level": risk_level(prob),
            "calculated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def model_summary(self) -> dict:
        return {k: self.info.get(k) for k in ("model_name", "alias", "version", "run_name", "test_accuracy")}


def risk_level(prob: float) -> str:
    return "Alto" if prob >= 0.6 else ("Medio" if prob >= 0.3 else "Baixo")
