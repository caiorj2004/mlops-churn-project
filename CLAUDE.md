# Contexto do projeto (para o Claude Code)

Projeto prático da disciplina de **MLOps** (IESB, Ciência de Dados e IA) do Caio. Previsão de churn tabular que evolui por encontros: refatoração modular → Docker → Pandera → DVC → MLflow → FastAPI → Streamlit → AWS.

## Regras do aluno (seguir sempre)
- Responder em **português**.
- **Nunca gerar dados sintéticos/mock** para fazer algo "funcionar". Todo resultado deve vir do `data/raw/churn.csv` real e do modelo treinado.
- O dataset **NÃO é o Telco Customer Churn** do Kaggle. É o `churn.csv` do próprio projeto (versionado no DVC, 7.043 linhas, IDs no formato `7000-QFVW`, `7015-DBER`). Nunca baixar ou assumir outra base.
- Seguir o que é pedido em aula (materiais na pasta do Project). Não introduzir ferramentas fora do plano de ensino sem perguntar (ex.: DagsHub).
- Ao orientar comandos, deixar explícito **onde** rodar: terminal do VS Code (Windows/PowerShell) ou terminal da EC2 (Ubuntu).

## Ambientes
- **Casa:** `C:\Users\Caio\Downloads\mlops-churn-project`, Python 3.14, `.venv` local, perfil AWS `churn`.
- **Faculdade:** máquinas de laboratório com Python 3.8, PowerShell com ExecutionPolicy restrita (usar `Set-ExecutionPolicy -Scope Process RemoteSigned`), commits saem como "unknown" e são reatribuídos em casa.
- DVC remote é local (`C:/Users/Caio/dvc_remote`), então só funciona em casa; em outras máquinas o `churn.csv` precisa ser copiado manualmente.

## Estado atual (v3: arquitetura em duas camadas)
- Atividade: separar a camada de dados da camada de aplicação na AWS (`us-east-2`).
  - `churn-app-host` (EC2): container único com Supervisord rodando FastAPI (`src/api_v3.py`, :8000) + Streamlit (`app_churn_v3.py`, :8501). Imagem no ECR `app-churn-v1:latest`.
  - `churn-db-host` (EC2, IP privado 172.31.19.112): PostgreSQL 15 em Docker (`churn_postgres`, banco `churn_db`), tabela `customers` com os 7.043 registros carregados via `infra/db/init_customers.sql`.
- Credenciais do banco ficam só no `.env` (local) ou `~/churn.env` (EC2). Ver `.env.example`.
- v1/v2 (lookup em Parquet) continuam no repositório como histórico.

## Comandos úteis
```powershell
python -m src.model_trainer                              # treina e salva models/churn_pipeline.joblib
python -m uvicorn src.api_v3:app --port 8000 --reload    # API v3
streamlit run app_churn_v3.py                            # painel v3
.\infra\deploy_ecr.ps1 -AwsProfile churn                 # build + push para o ECR
```
Detalhes de infraestrutura: `infra/README.md`.
