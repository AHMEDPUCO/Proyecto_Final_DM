# model_api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias
COPY model_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación
COPY model_api /app

# Copiar el modelo entrenado desde la raíz del proyecto
# 👇 IMPORTANTE: el archivo .pkl debe existir en la raíz cuando hagas `docker compose build`
COPY model_trading_AAPL_LogisticRegression.pkl /app/model_trading_pipeline.pkl

# Variables de entorno
ENV API_PORT=8000
ENV MODEL_PATH=/app/model_trading_pipeline.pkl

EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
