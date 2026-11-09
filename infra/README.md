# Infraestrutura e Implantação (MLOps Churn App v1)

Este diretório contém os artefatos de infraestrutura como código e scripts de automação necessários para reproduzir a arquitetura do projeto na AWS.

---

## 1. Topologia da Arquitetura

```text
[Cliente / Navegador]
        │
        ▼ (Porta 8501 - HTTP)
┌── AWS EC2 (t3.micro - Ubuntu) ──────────────────────────────────────────┐
│                                                                        │
│   IAM Instance Profile: EC2-ECR-ReadRole (Sem chaves fixas em disco)  │
│                                                                        │
│   ┌── Docker Container (app-churn-v1) ──────────────────────────────┐  │
│   │                                                                 │  │
│   │   [Supervisord] (Gerenciador de Processos)                     │  │
│   │      ├── [FastAPI]   : Porta interna 8000 (Lookup em Parquet)   │  │
│   │      └── [Streamlit] : Porta exposta 8501 (Frontend interativo) │  │
│   │                                                                 │  │
│   │   Base de Dados: data/batch_predictions.parquet (7.043 registros)│
│   └─────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
        ▲
        │ docker pull (Autenticação temporária STS)
┌───────┴──────────────┐
│ Amazon ECR           │
│ app-churn-v1:latest  │
│ (Região: us-east-2)  │
└──────────────────────┘