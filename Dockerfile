FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends supervisor && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pyarrow fastparquet supervisor streamlit requests scikit-learn

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY src/ ./src/
COPY data/ ./data/
COPY app_churn_v1.py .

# Executa o batch scoring autentico durante a construcao da imagem
RUN python -m src.batch_scoring

EXPOSE 8501

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
