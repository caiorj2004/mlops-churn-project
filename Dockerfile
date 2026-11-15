# Imagem da aplicacao (AWS): FastAPI + Streamlit no mesmo container, via Supervisord.
# Serve o modelo @champion do MLflow Model Registry, exportado antes do build:
#   python -m src.export_champion   (feito automaticamente por infra/deploy_ecr.ps1)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends supervisor && rm -rf /var/lib/apt/lists/*

# Mesmas versoes de bibliotecas com que o @champion foi treinado (geradas pelo MLflow)
# + dependencias da aplicacao (banco, Streamlit)
COPY models/champion/requirements.txt ./champion-requirements.txt
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r champion-requirements.txt -r requirements-app.txt

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
# Sanitiza o arquivo removendo qualquer BOM UTF-8 e CRLF do Windows:
RUN sed -i '1s/^\xef\xbb\xbf//' /etc/supervisor/conf.d/supervisord.conf && sed -i 's/\r$//' /etc/supervisor/conf.d/supervisord.conf

COPY src/ ./src/
COPY models/champion/ ./models/champion/
COPY data/raw/churn.csv ./data/raw/churn.csv
COPY app_churn_v1.py app_churn_v2.py ./

# app-churn-v1: resultados pre-calculados pelo @champion para os 7.043 clientes (parquet)
RUN python -m src.batch_scoring

# Na v2 os dados dos clientes vem do PostgreSQL da 2a EC2. Conexao via variaveis
# de ambiente em tempo de execucao: docker run --env-file ...
EXPOSE 8501

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
