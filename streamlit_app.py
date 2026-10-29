import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Painel MLOps - Predição de Churn", layout="wide")

st.title("📊 Painel de Predição de Churn - MLOps")
st.markdown("Consulte previsões em tempo real e inspecione a governança do modelo **@champion**.")

st.sidebar.header("Parâmetros de Entrada")
customer_id = st.sidebar.text_input("Código do Cliente (customer_id)", value="7590-VHVEG")

usar_manual = st.sidebar.checkbox("Inserir atributos manualmente (Simulação)", value=False)

tenure, monthly, total = None, None, None
if usar_manual:
    tenure = st.sidebar.number_input("Meses de Contrato (tenure)", min_value=1, max_value=120, value=1)
    monthly = st.sidebar.number_input("Cobrança Mensal (MonthlyCharges)", min_value=0.0, value=29.85, step=5.0)
    total = st.sidebar.number_input("Cobrança Total (TotalCharges)", min_value=0.0, value=29.85, step=10.0)

opcoes = st.sidebar.multiselect(
    "O que você deseja avaliar?",
    ["Predição de Churn", "Métricas do Champion", "Hiperparâmetros", "Metadados Gerais"],
    default=["Predição de Churn", "Métricas do Champion"]
)

btn_consultar = st.sidebar.button("Consultar Modelo", type="primary")

if btn_consultar:
    if not customer_id.strip():
        st.error("Por favor, informe um código de cliente válido.")
    else:
        with st.spinner("Consultando modelo @champion na API..."):
            try:
                if usar_manual:
                    payload = {
                        "customer_id": customer_id.strip(),
                        "tenure": int(tenure),
                        "monthly_charges": float(monthly),
                        "total_charges": float(total)
                    }
                    response = requests.post(f"{API_URL}/predict", json=payload)
                else:
                    response = requests.get(f"{API_URL}/predict/{customer_id.strip()}")

                if response.status_code == 200:
                    data = response.json()
                    champion = data.get("champion_model_info", {})

                    st.success(f"Dados processados com sucesso para o cliente: **{customer_id}**")
                    col1, col2 = st.columns(2)

                    if "Predição de Churn" in opcoes:
                        with col1:
                            st.subheader("Resultado da Predição")
                            label = data.get("label", "Desconhecido")
                            pred = data.get("churn_prediction", 0)

                            if pred == 1:
                                st.error(f"Classificação: **{label} (1)** - Alta probabilidade de cancelamento.")
                            else:
                                st.success(f"Classificação: **{label} (0)** - Baixo risco de cancelamento.")
                            
                            st.caption(f"Origem dos dados: `{data.get('customer_source', 'N/A')}`")

                    if "Métricas do Champion" in opcoes:
                        with col2:
                            st.subheader("Métricas do Modelo Ativo")
                            acc = champion.get("test_accuracy", 0.0)
                            st.metric(label="Acurácia de Teste (@champion)", value=f"{acc * 100:.2f}%")
                            st.info(f"Versão no Registry: **v{champion.get('version', 'N/A')}** | Tag: `{champion.get('alias', '@champion')}`")

                    if "Hiperparâmetros" in opcoes:
                        st.subheader("Hiperparâmetros de Treino")
                        params = champion.get("hyperparameters", {})
                        if params:
                            st.json(params)
                        else:
                            st.write("Nenhum hiperparâmetro retornado.")

                    if "Metadados Gerais" in opcoes:
                        st.subheader("Linhagem e Rastreamento")
                        st.write(f"- **Nome do Modelo:** `{champion.get('model_name', 'churn-model')}`")
                        st.write(f"- **MLflow Run ID:** `{champion.get('run_id', 'N/A')}`")
                        st.write(f"- **Status de Produção:** `{champion.get('status', 'N/A')}`")

                    st.divider()
                    st.subheader("📖 Explicação da Saída")
                    st.markdown("""
                    - **Churn (Sim / 1):** Padrões contratuais e financeiros indicam alto risco de cancelamento.
                    - **No Churn (Não / 0):** O perfil do cliente indica retenção e estabilidade.
                    - **Acurácia de Teste:** Eficácia medida em conjunto de teste não utilizado durante o treinamento.
                    """)

                elif response.status_code == 404:
                    st.warning(f"Cliente '{customer_id}' não encontrado na base. Marque a opção de inserção manual para simular.")
                elif response.status_code == 503:
                    st.error("O modelo @champion não está carregado na API.")
                else:
                    st.error(f"Erro na API ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"Não foi possível conectar à API em `{API_URL}`. Verifique se o serviço está rodando.")
