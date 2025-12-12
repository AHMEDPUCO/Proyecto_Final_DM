FROM python:3.11-slim

WORKDIR /app

# Opcional: copiar un requirements específico
COPY feature_builder/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY feature_builder /app/feature_builder

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "feature_builder.build_features"]
