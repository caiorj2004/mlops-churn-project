FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends supervisor && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pyarrow fastparquet supervisor streamlit requests scikit-learn joblib

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY src/ ./src/
COPY data/ ./data/
COPY app_churn_v1.py .
COPY app_churn_v2.py .

# 1. Treina o pipeline de Machine Learning (gera models/churn_pipeline.joblib)
RUN python -m src.model_trainer

# 2. Computa a base consolidada de scores historicos (gera data/batch_predictions.parquet)
RUN python -m src.batch_scoring

EXPOSE 8501

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
