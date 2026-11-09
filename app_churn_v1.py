import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="App Churn v1 - Consulta em Lote", layout="wide")

st.title("📋 Painel de Consulta de Scores de Churn (v1)")
st.caption("Arquitetura AWS com Container Unificado (FastAPI + Streamlit via Supervisord)")

st.sidebar.header("Consulta de Resultados")
customer_id = st.sidebar.text_input("Codigo do Cliente (customer_id)", value="7015-DBER")
btn_buscar = st.sidebar.button("Buscar Resultado", type="primary")

if btn_buscar:
    if not customer_id.strip():
        st.error("Informe um codigo de cliente valido.")
    else:
        with st.spinner("Buscando score consolidado na API interna..."):
            try:
                res = requests.get(f"{API_URL}/results/{customer_id.strip()}", timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Resultado localizado para o cliente **{customer_id}**")

                    c1, c2, c3 = st.columns(3)
                    pred = data.get("churn_prediction", 0)
                    prob = data.get("churn_probability", 0.0)
                    risk = data.get("risk_level", "N/A")

                    c1.metric("Classificacao", "Churn (1)" if pred == 1 else "Nao Churn (0)")
                    c2.metric("Probabilidade de Cancelamento", f"{prob * 100:.2f}%")
                    c3.metric("Nivel de Risco", risk)

                    st.divider()
                    st.subheader("Detalhes da Consulta em Lote")
                    st.json(data)
                elif res.status_code == 404:
                    st.warning(f"O identificador '{customer_id}' nao consta na base consolidada.")
                else:
                    st.error(f"Erro {res.status_code}: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Falha ao comunicar com o backend interno na porta 8000.")
