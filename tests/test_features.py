"""Contrato de features: a inferencia transforma um cliente exatamente como o treino."""
import pandas as pd
import pytest

from src.features import build_feature_contract, features_for_inference, process_features


def test_inferencia_reproduz_o_treino(dados):
    contrato = build_feature_contract(dados)
    X_treino, _ = process_features(dados)
    brutos = pd.read_csv("data/raw/churn.csv").drop(columns=["Churn"])
    X_inf = features_for_inference(brutos, contrato)
    pd.testing.assert_frame_equal(X_inf.reset_index(drop=True), X_treino.reset_index(drop=True),
                                  check_dtype=False)


def test_categoria_fora_do_contrato_e_recusada(dados):
    contrato = build_feature_contract(dados)
    cliente = pd.read_csv("data/raw/churn.csv").drop(columns=["Churn"]).head(1)
    cliente["Contract"] = "Vitalicio"
    with pytest.raises(ValueError, match="Vitalicio"):
        features_for_inference(cliente, contrato)
