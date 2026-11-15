import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="App Churn v2 - Two-Tier", layout="wide")

st.title("📊 Painel de Retenção de Clientes (v2)")
st.caption("Arquitetura em duas camadas: aplicação + modelo numa EC2, PostgreSQL em Docker em outra EC2")


def legenda_modelo(resposta: dict) -> str:
    m = resposta.get("model") or {}
    acc = m.get("test_accuracy")
    acc_txt = f" · acurácia de teste {acc * 100:.2f}%" if acc is not None else ""
    return f"Modelo: `{m.get('model_name')}` v{m.get('version')} {m.get('alias')} ({m.get('run_name')}){acc_txt}"


# =========================================================================
# BLOCO 1: CONSULTA POR CODIGO NO POSTGRESQL (camada de dados - 2a EC2)
# =========================================================================
st.subheader("🔍 1. Consulta de Cliente no Banco (PostgreSQL)")
c_id_col, c_btn_col = st.columns([3, 1])

with c_id_col:
    search_id = st.text_input("Código do Cliente (customerID)", value="7015-DBER", help="Ex: 7015-DBER, 7000-QFVW")
with c_btn_col:
    st.write("")
    st.write("")
    btn_search = st.button("Buscar Cliente", type="secondary", use_container_width=True)

if btn_search:
    if not search_id.strip():
        st.error("Informe um ID valido.")
    else:
        try:
            res = requests.get(f"{API_URL}/results/{search_id.strip()}", timeout=10)
            if res.status_code == 200:
                data = res.json()
                st.success(f"Cliente **{search_id}** encontrado no PostgreSQL e avaliado pelo modelo")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Previsão do Modelo", "Churn (1)" if data.get("churn_prediction") == 1 else "Nao Churn (0)")
                m2.metric("Probabilidade", f"{data.get('churn_probability', 0.0) * 100:.2f}%")
                m3.metric("Risco", data.get("risk_level", "N/A"))
                m4.metric("Churn real (base)", data.get("actual_churn", "N/A"))
                st.caption(legenda_modelo(data))
                st.markdown("**Registro retornado pelo banco**")
                st.dataframe([data.get("customer", {})], use_container_width=True)
                with st.expander("Resposta completa da API (JSON)"):
                    st.json(data)
            elif res.status_code == 404:
                st.warning(f"Cliente '{search_id}' nao encontrado no banco.")
            elif res.status_code == 422:
                st.error(f"Dado fora do contrato do modelo: {res.json().get('detail')}")
            elif res.status_code == 503:
                st.error(f"Banco de dados indisponivel: {res.json().get('detail')}")
            else:
                st.error(f"Erro {res.status_code}: {res.text}")
        except requests.exceptions.ConnectionError:
            st.error("Servico backend indisponivel na porta 8000.")

st.divider()

# =========================================================================
# BLOCO 2: NOVA INFERÊNCIA EM TEMPO REAL (Online Serving)
# =========================================================================
st.subheader("⚡ 2. Nova Inferência em Tempo Real (Simulação de Novo Cliente)")
st.caption("Preencha os parametros abaixo para executar uma chamada ativa no modelo de Machine Learning:")

with st.form("form_nova_inferencia"):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Parametros Contratuais**")
        tenure = st.slider("Tempo de Contrato (Meses)", min_value=1, max_value=72, value=6)
        contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Metodo de Pagamento", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless = st.selectbox("Fatura Digital (Paperless)", ["Yes", "No"])

    with col_b:
        st.markdown("**Custos e Servicos Digitais**")
        monthly_charges = st.number_input("Valor Mensal ($)", min_value=15.0, max_value=150.0, value=85.0, step=1.0)
        total_charges = st.number_input("Valor Total Acumulado ($ - Deixe 0 para auto)", min_value=0.0, value=0.0, step=10.0)
        internet = st.selectbox("Servico de Internet", ["Fiber optic", "DSL", "No"])
        tech_support = st.selectbox("Suporte Tecnico", ["No", "Yes", "No internet service"])

    with col_c:
        st.markdown("**Seguranca e Perfil**")
        online_sec = st.selectbox("Seguranca Online", ["No", "Yes", "No internet service"])
        online_bak = st.selectbox("Backup Online", ["No", "Yes", "No internet service"])
        device_prot = st.selectbox("Protecao de Dispositivo", ["No", "Yes", "No internet service"])
        senior = st.selectbox("Cliente Idoso (SeniorCitizen)", ["0", "1"])

    btn_predict = st.form_submit_button("⚡ Fazer Chamada no Modelo", type="primary", use_container_width=True)

if btn_predict:
    payload = {
        "gender": "Female",
        "SeniorCitizen": senior,
        "Partner": "No",
        "Dependents": "No",
        "tenure": int(tenure),
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges) if total_charges > 0 else None
    }

    with st.spinner("Enviando parametros e computando inferencia no modelo..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                st.success("Inferência executada com sucesso pelo classificador!")

                r1, r2, r3 = st.columns(3)
                p_bin = result["churn_prediction"]
                p_prob = result["churn_probability"] * 100
                p_risk = result["risk_level"]

                r1.metric("Classificacao do Modelo", "Risco de Churn (1)" if p_bin == 1 else "Retencao Estavel (0)")
                r2.metric("Probabilidade Calculada", f"{p_prob:.2f}%")
                r3.metric("Nivel de Risco", p_risk)
                st.caption(legenda_modelo(result))

                st.json(result)
            else:
                st.error(f"Erro na API ({resp.status_code}): {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("Nao foi possivel conectar a API na porta 8000.")
