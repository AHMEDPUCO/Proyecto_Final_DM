# TradingML Pipeline - Sistema Completo de Predicción de Mercados

Este proyecto implementa una solución end-to-end para predecir la **dirección diaria** de un activo (ej. AAPL). Abarca desde la ingesta automatizada de datos OHLCV, construcción de features técnicas, entrenamiento y optimización de modelos de ML, hasta el despliegue de una **API REST** de inferencia en tiempo real. Todo se orquesta con **Docker Compose** .

## Descripción General

El pipeline integra cuatro módulos especializados:

1. **Ingesta de Datos**: Descarga y normalización de datos OHLCV → `raw.prices_daily`
2. **Construcción de Features**: Ingeniería de features técnicas y temporales → `analytics.daily_features`
3. **Machine Learning**: Entrenamiento, comparación y optimización de 7 algoritmos → Modelo ganador serializado --> `aml_trading_classifier `
4. **API de Inferencia**: Servicio REST 
5. **Infraestructura**: Docker Compose

---

## 1. Stack Tecnológico
###  Librerías Python Principales

**Para Ingesta y Procesamiento:**

```
pandas
numpy
psycopg2-binary
python-dotenv
yfinance
```

**Para Machine Learning y Análisis:**

```
scikit-learn
xgboost
lightgbm
joblib
```

**Para API y Servicios:**

```
fastapi
uvicorn[standard]
pydantic
```

**Para Visualización:**

```
matplotlib
seaborn
```

### 1.3 Infraestructura

| Herramienta          | Función                                |
| -------------------- | -------------------------------------- |
| **PostgreSQL**       | Almacenamiento para `raw` y `analytics` |
| **Docker Compose**   | Orquestación multi-servicio            |
| **Jupyter Notebook** | Entorno interactivo                    |

---

## 2. Arquitectura General

El entorno se ejecuta mediante Docker Compose con los siguientes servicios:

| Servicio         | Descripción                                                    |
| ---------------- | -------------------------------------------------------------- |
| postgres         | Base de datos que almacena los esquemas `raw` y `analytics`.   |
| jupyter-notebook | Entorno para ejecutar notebooks (ingesta y ML).                |
| feature-builder  | Construye `analytics.daily_features` desde `raw.prices_daily`. |
| model_api        | API REST de inferencia que carga el `.pkl` exportado.          |

La comunicación entre servicios se realiza mediante redes internas definidas por Docker.

---

## 3. Configuración del Sistema

### Variables de Ambiente

El proyecto utiliza un archivo `.env` en la raíz. Ejemplo:

```env
PG_HOST=postgres
PG_PORT=5432
PG_DB=marketdb
PG_USER=postgres
PG_PASSWORD=postgres

PG_SCHEMA_RAW=raw
PG_SCHEMA_ANALYTICS=analytics

TICKERS=AAPL
START_DATE=2018-01-01
END_DATE=2024-12-31
RUN_ID=init_ingest

MODEL_PATH=model_trading_AAPL_LogisticRegression.pkl
```

Copiar el archivo base:

```bash
cp .env.example .env
```

---

## 4. Pipeline de Datos

### 4.1 Levantar Compose

Levantar todo el stack:

```bash
docker compose up -d --build
```

Verifica salud de la API:

```bash
curl http://localhost:8000/health
```

---

### 4.2 Ingesta de Datos de Mercado → `raw.prices_daily`

La ingesta se realiza con el notebook:

* `01_ingesta_prices_raw.ipynb`

Este paso:

* Descarga OHLCV desde Yahoo Finance (según `TICKERS`, `START_DATE`, `END_DATE`)
* Normaliza columnas (open, high, low, close, adj_close, volume)
* Inserta a `raw.prices_daily`
* Permite re-ejecución segura (idempotencia según lógica del notebook)
* Registra metadatos de trazabilidad (ej. run_id / timestamp)

**Ejecución**

1. Abre Jupyter (servicio `jupyter-notebook`)
2. Ejecuta `01_ingesta_prices_raw.ipynb`


### 4.3 Construcción de Features → `analytics.daily_features` 

El script `build_features.py` construye y/o actualiza `analytics.daily_features` desde `raw.prices_daily`, con control de idempotencia y logs.

#### 4.3.1 Especificación CLI (sin código)

**Argumentos mínimos:**

* `--mode {full, by-date-range}`
* `--ticker <string>`
* `--start-date YYYY-MM-DD`
* `--end-date YYYY-MM-DD`
* `--run-id <string>`
* `--overwrite {true,false}`

**Comportamiento:**

* `full`: (re)crea la tabla completa para el ticker/rango.
* `by-date-range`: solo procesa un subconjunto de fechas.
* **Idempotente**: sin duplicar filas (sobrescribir por `(ticker, date)` o borrar/reinsertar).

**Logs esperados:**

* Filas creadas/actualizadas.
* Fecha mín y máx procesada.
* Duración.

#### 4.3.2 Comandos de construcción (requerido)

> En Docker Compose :

```bash
docker compose run --rm feature-builder \
  --mode full \
  --ticker AAPL \
  --run-id build_v1 \
  --overwrite true
```

---

### 4.4 Breve explicación de columnas principales (`analytics.daily_features`)

La tabla analítica es un **One Big Table** (1 fila por día por ticker). Columnas clave:

| Columna                        | Descripción                                        |
| ------------------------------ | -------------------------------------------------- |
| date, ticker                   | Clave por día y activo                             |
| year, month, day_of_week       | Variables temporales (conocidas al inicio del día) |
| open, high, low, close, volume | OHLCV diarios                                      |
| return_close_open              | (close − open) / open                              |
| return_prev_close              | close_t / close_{t-1} − 1                          |
| volatility_5d                  | Volatilidad rolling (std de retornos)              |
| run_id                         | Identificador de corrida                           |
| ingested_at_utc                | Timestamp UTC (trazabilidad)                       |

---

## 5. Machine Learning y Modelado Predictivo

El módulo de ML está en el notebook:

* `ml_trading_classifier.ipynb`

### 5.1 Definición del Problema

Clasificación binaria diaria:

```bash
target_up = 1 si close > open
target_up = 0 en caso contrario
```

### 5.2 Features (por qué se escogieron)

Las features se diseñaron para **NO tener leakage**: solo usan información disponible **hasta el inicio del día** a predecir (principalmente datos del día anterior o ventanas rolling basadas en datos pasados). Incluyen:

* **Lags OHLCV (t−1):** capturan continuidad de precio/volumen.
* **Retornos previos:** señales de momentum / reversión.
* **Volatilidades (rolling):** detectan régimen de riesgo (mercado calmo vs volátil).
* **Indicadores técnicos simples (MA ratio, RSI, momentum):** resumen tendencia y fuerza del movimiento.
* **Calendario (día/mes):** permite capturar estacionalidad leve.

### 5.3 Split temporal (sin data leakage)

Split estrictamente por tiempo (ejemplo real en la ejecución mostrada):

* **Train:** 2018–2021
* **Validation:** 2022–2023
* **Test:** 2024 (último año disponible)

Además, se usa `TimeSeriesSplit` dentro del tuning para validación cruzada temporal.

### 5.4 Modelos entrenados (≥ 7)

Se comparan 7 modelos distintos:

1. Logistic Regression
2. Linear SVC
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. XGBoost
7. LightGBM

Se evalúa con métricas:

* Accuracy, Precision, Recall, F1, ROC-AUC
* Matriz de confusión + reporte de clasificación

### 5.5 Baseline obligatorio

Baseline: `DummyClassifier(strategy="most_frequent")`, que predice siempre la clase mayoritaria (en este caso tiende a **Up**).

Se usa para verificar si ML realmente añade valor vs una estrategia trivial.

---

## 6. Mejor Modelo (resultado del experimento)

### 6.1 Modelo ganador (por métrica de validación)

**Mejor modelo por F1 en Validation (entre los modelos ML):** **LogisticRegression**

* Fue el mejor de la tabla comparativa (ordenada por `val_f1`).

> Nota: El baseline obtuvo un F1 mayor que cualquier modelo ML, por lo que **ningún modelo superó al baseline** en F1 en validación y test. Aun así, se seleccionó LogisticRegression como el **mejor modelo ML** entrenado por desempeño relativo y porque permite interpretación (coeficientes + análisis de features).

### 6.2 Evaluación en Test (ejecución real)

En Test (2024) se reportó (aprox):

* Accuracy ~ 0.54
* F1 ~ 0.70
* Recall muy alto (sesgo a predecir Up)
* ROC-AUC < 0.50 (débil separación probabilística)

Conclusión: el modelo tiende a **predecir “Up”** con mucha frecuencia (alto recall para Up, baja performance para Down/Equal).

---

## 7. Serialización y Exportación del Modelo

El notebook exporta:

* `model_trading_<TICKER>_<MODEL>.pkl` (pipeline completo)
* `experiment_info_<TICKER>.json` (métricas, parámetros y trazabilidad)

Ejemplo:

* `model_trading_AAPL_LogisticRegression.pkl`
* `experiment_info_AAPL.json`

---

## 8. API REST para Inferencia

### 8.1 Levantar la API

Con Compose(se levanta con el resto de servicios):

```bash
docker compose up -d 
```
Health:

```bash
curl http://localhost:8000/health
```

### 8.2 Probar `/predict`

**Windows PowerShell :**

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "instances": [
      {
        "open_lag1": 250.41,
        "high_lag1": 253.28,
        "low_lag1": 249.42,
        "close_lag1": 252.20,
        "volume_lag1": 39480700,
        "return_close_open_lag1": 0.0072,
        "return_prev_close_lag1": 0.0031,
        "volatility_5d_lag1": 0.012,
        "volume_ma_5": 35000000,
        "range_hl_lag1": 0.012,

        "price_range_pct": 0.015,
        "volume_change": 1.02,
        "ma_5_vs_20": 1.05,
        "momentum_5": 0.01,
        "momentum_10": 0.02,
        "volatility_10d": 0.013,
        "volatility_20d": 0.017,
        "rsi_14": 55.0,

        "day_of_week": 1,
        "month": 1
      }
    ]
  }'
```

Respuesta esperada (ejemplo):

```json
{"results":[{"prediction":1,"proba_up":0.54}]}
```

---

## 9. Simulación de Inversión con USD 10,000 (2025 o último año disponible)

### 9.1 Requisito del proyecto

* Simular en **2025** o el **último año disponible** (en la ejecución mostrada fue **2024**).
* Ese año debe ser **solo Test** (no usado en train/val).

### 9.2 Regla simple de trading (usada)

Para cada día bursátil del período Test:

* Si `prediction = 1` ⇒ **comprar al open y vender al close** (intradía).
* Si `prediction = 0` ⇒ quedarse en **cash** (retorno 0).
* Sin comisiones, sin apalancamiento (capital nunca negativo).

### 9.3 Salidas

* Capital final
* Retorno total (%)
* Retorno anualizado
* Número de trades
* Curva de equity (capital vs tiempo)
* Métricas de riesgo: drawdown, volatilidad, Sharpe

**Interpretación corta (máx 3 líneas, incluye ganancias):**
La simulación mostró **ganancias**: el capital creció desde $10,000 hasta $12,966.53 (**+29.67%**). El **max drawdown** mide la peor caída desde un máximo (riesgo), y el **Sharpe** resume retorno vs volatilidad (consistencia del retorno). La **volatilidad anual** indica cuánto fluctúa la curva de capital.

---

## 10. Guía de Ejecución Completa (end-to-end)

### 10.1 Preparación

```bash
cp .env.example .env
docker compose up -d --build
```

### 10.2 Ingesta

* Ejecutar `01_ingesta_prices_raw.ipynb` en Jupyter.

### 10.3 Features (OBT)

```bash
python build_features.py --mode full --ticker AAPL --run-id run_full_aapl_1 --overwrite true
```

### 10.4 Entrenamiento + exportación

* Ejecutar `ml_trading_classifier.ipynb`:

  * entrenamiento, comparación de modelos, evaluación en test
  * simulación de inversión
  * exporta `.pkl` + `experiment_info_*.json`

### 10.5 API + pruebas

```bash
curl http://localhost:8000/health
```

y probar `/predict` (PowerShell o Postman).

---

## 11. Resumen

TradingML Pipeline es una solución completa de MLOps aplicada a finanzas:

* Pipeline de datos (raw → analytics)
* Entrenamiento y comparación de 7 modelos con validación temporal
* Explicabilidad básica (importancia por coeficientes / importancias)
* Backtesting de estrategia simple con USD 10,000
* API REST dockerizada para inferencia reproducible

---
