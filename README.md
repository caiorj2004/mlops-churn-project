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


## Visão Inicial da Estrutura do Projeto
mlops-churn-project/
├── .github/
│   └── workflows/        # CI/CD com GitHub Actions
├── data/
│   ├── raw/              # Dados brutos fixados pelo professor (versionados via DVC)
│   ├── interim/          # Dados intermediários / limpos
│   └── processed/        # Dados prontos para treino
├── docker/               # Dockerfiles específicos (Airflow, API, etc.)
├── notebooks/            # Notebook inicial (a ser refatorado no Encontro 2)
├── src/                  # Código modular do projeto (.py)
│   ├── __init__.py
│   ├── data/             # Scripts de coleta e validação
│   ├── features/         # Engenharia de atributos
│   ├── models/           # Treinamento, tracking (MLflow) e inferência
│   └── visualization/    # Gráficos e relatórios
├── tests/                # Testes unitários e de integração
├── .dvc/                 # Configuração do DVC
├── .gitignore
├── docker-compose.yml    # Orquestração local (Airflow, MLflow, etc.)
├── pyproject.toml        # Gerenciamento moderno de dependências
├── README.md             # Documentação principal do repositório
└── requirements.txt      # Dependências mínimas iniciais
