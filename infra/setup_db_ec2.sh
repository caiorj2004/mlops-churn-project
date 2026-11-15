#!/usr/bin/env bash
# Provisiona a camada de dados (arquitetura em duas camadas) na EC2 churn-db-host:
# PostgreSQL 15 em Docker + carga do churn.csv real (CSV -> postgres).
#
# Uso (no terminal da churn-db-host, dentro do clone do repositorio):
#   export DB_PASSWORD='sua-senha'
#   bash infra/setup_db_ec2.sh
#
# Requer data/raw/churn.csv presente (7.043 registros reais do projeto).
set -euo pipefail

: "${DB_PASSWORD:?Defina DB_PASSWORD antes de rodar (export DB_PASSWORD=...)}"
DB_NAME="${DB_NAME:-churn_db}"
DB_USER="${DB_USER:-postgres}"
CONTAINER="churn_postgres"
CSV_PATH="data/raw/churn.csv"

if [ ! -f "${CSV_PATH}" ]; then
    echo "ERRO: ${CSV_PATH} nao encontrado. Copie o churn.csv real para essa pasta antes."
    exit 1
fi

echo "=== 1. Instalando Docker ==="
if ! command -v docker &> /dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y docker.io
fi
sudo systemctl enable --now docker

echo "=== 2. Subindo PostgreSQL 15 (porta 5432, volume persistente pgdata) ==="
if ! sudo docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    sudo docker run -d --name "${CONTAINER}" \
        -e POSTGRES_DB="${DB_NAME}" \
        -e POSTGRES_USER="${DB_USER}" \
        -e POSTGRES_PASSWORD="${DB_PASSWORD}" \
        -p 5432:5432 \
        -v pgdata:/var/lib/postgresql/data \
        --restart unless-stopped \
        postgres:15
else
    sudo docker start "${CONTAINER}" > /dev/null
fi

echo "=== 3. Aguardando o banco aceitar conexoes ==="
until sudo docker exec "${CONTAINER}" pg_isready -U "${DB_USER}" -d "${DB_NAME}" > /dev/null 2>&1; do
    sleep 1
done

echo "=== 4. Carga CSV -> postgres ==="
sudo docker cp "${CSV_PATH}" "${CONTAINER}:/tmp/churn.csv"
sudo docker exec -i "${CONTAINER}" psql -v ON_ERROR_STOP=1 -U "${DB_USER}" -d "${DB_NAME}" < infra/db/init_customers.sql

echo "=== Camada de dados pronta. IP privado desta maquina: $(hostname -I | awk '{print $1}') ==="
