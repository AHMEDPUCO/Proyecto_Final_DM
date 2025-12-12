CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS analytics.daily_features (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,

    -- Identificación de día
    year INT,
    month INT,
    day_of_week INT,  -- 0=lunes ... 6=domingo

    -- Mercado (agregado diario, viniendo de raw.prices_daily)
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,

    -- Features derivadas
    return_close_open DOUBLE PRECISION,      -- (close - open) / open
    return_prev_close DOUBLE PRECISION,      -- close / close_lag1 - 1
    volatility_5d DOUBLE PRECISION,          -- std de retornos últimos 5 días

    -- Metadatos
    run_id TEXT,
    ingested_at_utc TIMESTAMP,

    PRIMARY KEY (date, ticker)
);
