#!/usr/bin/env bash
set -e

REGION="us-east-2"
ACCOUNT_ID="841770988469"
REPO_NAME="app-churn-v1"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${REPO_NAME}:latest"

echo "=== 1. Atualizando pacotes e instalando dependencias ==="
sudo apt-get update -y
sudo apt-get install -y docker.io unzip curl
sudo systemctl start docker
sudo systemctl enable docker

echo "=== 2. Instalando AWS CLI v2 oficial ==="
if ! command -v aws &> /dev/null; then
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install
    rm -rf /tmp/aws /tmp/awscliv2.zip
fi

echo "=== 3. Autenticando no ECR via IAM Role (STS Instance Profile) ==="
aws ecr get-login-password --region "${REGION}" | sudo docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "=== 4. Baixando e iniciando o container unificado (FastAPI + Streamlit) ==="
sudo docker pull "${IMAGE_URI}"
sudo docker stop churn_app 2>/dev/null || true
sudo docker rm churn_app 2>/dev/null || true
sudo docker run -d -p 8501:8501 --name churn_app --restart unless-stopped "${IMAGE_URI}"

echo "=== Deploy concluido. Servico ativo na porta 8501 ==="
