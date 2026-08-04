# Projeto Prático: Previsão de Churn (MLOps)

Repositório para o desenvolvimento do projeto prático da disciplina de **Engenharia de Machine Learning Aplicada - princípios de MLOps**. Ao longo dos encontros, este repositório evoluirá de um protótipo local para um pipeline completo de ponta a ponta: dados versionados, rastreamento de experimentos, orquestração, containerização, API de inferência e esteira de CI/CD.

---

## 🗺️ Arquitetura do Projeto (Os Três Pipelines)
Seguindo o modelo de referência do curso (`ml-ops.org`), o projeto atravessa três camadas principais:
1. **Data Pipeline:** Coleta, validação e versionamento de dados (DVC + Great Expectations/pandera).
2. **ML Pipeline:** Treinamento, experimentação e registro de modelos (MLflow).
3. **Code Pipeline / Deployment:** Empacotamento, API de serviço e automação (FastAPI + Docker + GitHub Actions).

---

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Contêineres:** Docker & Docker Compose
* **Versionamento de Dados:** DVC
* **Rastreamento de Experimentos:** MLflow
* **Orquestração:** Apache Airflow
* **Serviço de Modelo:** FastAPI
* **Automação:** GitHub Actions
