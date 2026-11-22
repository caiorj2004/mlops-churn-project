"""Data Pipeline: o contrato Pandera aceita o dado versionado e barra dado adulterado."""
import hashlib

import pytest
import yaml
from pandera.errors import SchemaErrors

from src.config import DATA_PATH
from src.schema import ChurnSchema


def test_csv_bate_com_o_ponteiro_dvc(dados):
    esperado = yaml.safe_load(open("data/raw/churn.csv.dvc", encoding="utf-8"))["outs"][0]["md5"]
    assert hashlib.md5(DATA_PATH.read_bytes()).hexdigest() == esperado


def test_dado_versionado_passa_no_contrato(dados):
    assert len(dados) == 7043
    assert dados["customerID"].is_unique
    assert set(dados["Churn"]) == {"Yes", "No"}


def test_contrato_barra_adulteracoes(dados):
    """Mesmas adulteracoes do test_schema.py (Encontro 4), agora como teste automatico."""
    df = dados.head(3).copy()
    df.loc[df.index[0], "tenure"] = 999
    df.loc[df.index[1], "MonthlyCharges"] = -15.0
    df.loc[df.index[2], "Contract"] = "Vitalicio"
    df.loc[df.index[2], "customerID"] = df.loc[df.index[0], "customerID"]
    with pytest.raises(SchemaErrors) as erro:
        ChurnSchema.validate(df, lazy=True)
    colunas = set(erro.value.failure_cases["column"])
    assert {"tenure", "MonthlyCharges", "Contract", "customerID"} <= colunas
