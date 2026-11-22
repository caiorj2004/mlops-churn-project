"""Software pipeline: a API v2 (consulta no banco + inferencia online) com o @champion de teste."""


def test_health_conecta_no_banco(cliente_api):
    saude = cliente_api.get("/health").json()
    assert saude["database"]["status"] == "connected"
    assert saude["database"]["records"] == 7043


def test_results_busca_no_banco_e_pontua(cliente_api):
    r = cliente_api.get("/results/7015-DBER")
    assert r.status_code == 200
    corpo = r.json()
    assert 0.0 <= corpo["churn_probability"] <= 1.0
    assert corpo["churn_prediction"] == int(corpo["churn_probability"] >= 0.5)
    assert corpo["actual_churn"] == "Yes"


def test_cliente_inexistente_da_404(cliente_api):
    assert cliente_api.get("/results/0000-NAOEXISTE").status_code == 404


def test_predict_online(cliente_api):
    r = cliente_api.post("/predict", json={"tenure": 2, "MonthlyCharges": 95.0})
    assert r.status_code == 200
    assert r.json()["inference_type"] == "real_time_online"


def test_predict_recusa_categoria_desconhecida(cliente_api):
    r = cliente_api.post("/predict", json={"Contract": "Vitalicio"})
    assert r.status_code == 422
    assert "Vitalicio" in r.json()["detail"]
