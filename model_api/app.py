# model_api/app.py
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

# ========= Configuración =========

# Ruta del modelo dentro del contenedor
MODEL_PATH = os.getenv("MODEL_PATH", "model_trading_pipeline.pkl")

app = FastAPI(
    title="Trading Direction Model API",
    description="API para predecir si el día cerrará al alza (target_up=1) usando un modelo entrenado.",
    version="1.0.0",
)

model_pipeline = None  # se cargará en startup


# ========= Esquemas de entrada/salida =========
class Instance(BaseModel):
    # FEATURES NUMÉRICAS BASE
    open_lag1: float
    high_lag1: float
    low_lag1: float
    close_lag1: float
    volume_lag1: float
    return_close_open_lag1: float
    return_prev_close_lag1: float
    volatility_5d_lag1: float
    volume_ma_5: float
    range_hl_lag1: float

    # NUEVAS FEATURES TÉCNICAS (LAS QUE AGREGASTE EN EL NOTEBOOK)
    price_range_pct: float
    volume_change: float
    ma_5_vs_20: float
    momentum_5: float
    momentum_10: float
    volatility_10d: float
    volatility_20d: float
    rsi_14: float

    # FEATURES CATEGÓRICAS
    day_of_week: int  # 0..4
    month: int        # 1..12

class PredictRequest(BaseModel):
    instances: List[Instance]


class PredictResponseItem(BaseModel):
    prediction: int
    proba_up: Optional[float] = None


class PredictResponse(BaseModel):
    results: List[PredictResponseItem]


# ========= Eventos =========

@app.on_event("startup")
def load_model():
    """Carga el modelo entrenado al iniciar la API."""
    global model_pipeline

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            f"Asegúrate de que el archivo .pkl se haya copiado correctamente al contenedor."
        )

    model_pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Modelo cargado desde {MODEL_PATH}")


# ========= Endpoints =========

@app.get("/health")
def health_check():
    """Endpoint simple para verificar que el servicio está vivo."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Recibe una o varias instancias con las mismas features que el modelo espera
    y devuelve predicciones de target_up (0 = no sube, 1 = sube).
    """
    if model_pipeline is None:
        raise RuntimeError("El modelo aún no ha sido cargado. Revisa los logs de arranque.")

    # Convertir lista de Instance -> DataFrame
    data = [inst.dict() for inst in request.instances]
    df = pd.DataFrame(data)

    # Predicciones
    preds = model_pipeline.predict(df)

    # Probabilidades (si el modelo las soporta)
    proba_up = None
    has_proba = hasattr(model_pipeline, "predict_proba")

    if has_proba:
        try:
            proba = model_pipeline.predict_proba(df)[:, 1]
            proba_up = proba
        except Exception:
            has_proba = False

    results: List[PredictResponseItem] = []
    for i, p in enumerate(preds):
        item = PredictResponseItem(
            prediction=int(p),
            proba_up=float(proba_up[i]) if has_proba and proba_up is not None else None
        )
        results.append(item)

    return PredictResponse(results=results)
