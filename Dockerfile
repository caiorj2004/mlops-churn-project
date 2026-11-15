FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends supervisor && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
# Sanitiza o arquivo removendo qualquer BOM UTF-8 e CRLF do Windows:
RUN sed -i '1s/^\xef\xbb\xbf//' /etc/supervisor/conf.d/supervisord.conf && sed -i 's/\r$//' /etc/supervisor/conf.d/supervisord.conf

COPY src/ ./src/
COPY data/ ./data/
COPY app_churn_v1.py app_churn_v2.py ./

# O modelo e treinado no build a partir do churn.csv real (obtido via `dvc pull`).
# O batch_scoring continua gerando o parquet usado pela v1 (src/api_lookup.py).
RUN python -m src.model_trainer
RUN python -m src.batch_scoring

# Na v2 os dados dos clientes vem do PostgreSQL da 2a EC2. Conexao via variaveis
# de ambiente em tempo de execucao: docker run --env-file .env ...
EXPOSE 8501

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
