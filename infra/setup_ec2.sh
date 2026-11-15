#!/usr/bin/env bash
# Deploy da camada de aplicacao (EC2 churn-app-host): baixa a imagem do ECR e
# sobe o container unificado (FastAPI + Streamlit) apontando para o PostgreSQL.
#
# Uso (no terminal da churn-app-host):
#   1. Crie ~/churn.env com as variaveis do banco (modelo em .env.example):
#        DB_HOST=<IP privado da churn-db-host>
#        DB_PORT=5432
#        DB_NAME=churn_db
#        DB_USER=postgres
#        DB_PASSWORD=<senha>
#   2. bash setup_ec2.sh
set -euo pipefail

REGION="us-east-2"
ACCOUNT_ID="841770988469"
REPO_NAME="app-churn-v1"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENV_FILE="${ENV_FILE:-$HOME/churn.env}"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${REPO_NAME}:${IMAGE_TAG}"

if [ ! -f "${ENV_FILE}" ]; then
    echo "ERRO: arquivo de ambiente ${ENV_FILE} nao encontrado (veja o cabecalho deste script)."
    exit 1
fi

echo "=== 1. Atualizando pacotes e instalando dependencias ==="
sudo apt-get update -y
sudo apt-get install -y docker.io unzip curl
sudo systemctl enable --now docker

echo "=== 2. Instalando AWS CLI v2 oficial ==="
if ! command -v aws &> /dev/null; then
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install
    rm -rf /tmp/aws /tmp/awscliv2.zip
fi

echo "=== 3. Autenticando no ECR via IAM Role (STS Instance Profile) ==="
aws ecr get-login-password --region "${REGION}" | sudo docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "=== 4. Baixando e iniciando o container (FastAPI + Streamlit) ==="
sudo docker pull "${IMAGE_URI}"
sudo docker stop churn_app 2>/dev/null || true
sudo docker rm churn_app 2>/dev/null || true
sudo docker run -d -p 8501:8501 --name churn_app --restart unless-stopped \
    --env-file "${ENV_FILE}" "${IMAGE_URI}"

echo "=== 5. Verificando a conexao com o banco ==="
sleep 8
sudo docker exec churn_app python -c "import requests; print(requests.get('http://127.0.0.1:8000/health', timeout=10).json())" || true

echo "=== Deploy concluido. Servico ativo na porta 8501 ==="
