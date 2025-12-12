import os
import argparse
from datetime import datetime, timedelta
import time
import sys

import psycopg2
import pandas as pd
import numpy as np
from psycopg2.extras import execute_values


def get_connection():
    """Establece conexión a PostgreSQL con manejo de errores"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "postgres"),
            port=os.getenv("PG_PORT", "5432"),
            dbname=os.getenv("PG_DB", "marketdb"),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD", "postgres"),
        )
        return conn
    except Exception as e:
        print(f" Error conectando a PostgreSQL: {str(e)}")
        sys.exit(1)
def analyze_data_coverage(df_raw: pd.DataFrame, ticker: str) -> dict:
    """
    Analiza la cobertura temporal de los datos y detecta huecos.
    """
    if df_raw.empty:
        return {
            "total_days": 0,
            "trading_days": 0,
            "missing_days": 0,
            "coverage_pct": 0,
            "date_range": "N/A - N/A",
            "gap_info": []
        }
    
    # Convertir a datetime y ordenar
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Encontrar todos los días entre min y max fecha
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    
    # Filtrar solo días de semana (lunes=0 a viernes=4)
    weekdays = all_dates[all_dates.dayofweek < 5]
    
    # Encontrar días faltantes
    existing_dates = set(df['date'])
    missing_dates = [date for date in weekdays if date not in existing_dates]
    
    # Calcular métricas
    total_trading_days = len(weekdays)
    existing_trading_days = len(existing_dates)
    missing_count = len(missing_dates)
    coverage_pct = (existing_trading_days / total_trading_days) * 100 if total_trading_days > 0 else 0
    
    # Analizar huecos consecutivos
    gap_info = []
    if missing_dates:
        missing_dates.sort()
        current_gap = [missing_dates[0]]
        
        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - missing_dates[i-1]).days == 1:
                current_gap.append(missing_dates[i])
            else:
                if len(current_gap) >= 3:  # Solo reportar huecos de 3+ días
                    gap_info.append({
                        'start': current_gap[0].strftime('%Y-%m-%d'),
                        'end': current_gap[-1].strftime('%Y-%m-%d'),
                        'length': len(current_gap)
                    })
                current_gap = [missing_dates[i]]
        
        # Último gap
        if len(current_gap) >= 3:
            gap_info.append({
                'start': current_gap[0].strftime('%Y-%m-%d'),
                'end': current_gap[-1].strftime('%Y-%m-%d'),
                'length': len(current_gap)
            })
    
    return {
        "total_days": total_trading_days,
        "trading_days": existing_trading_days,
        "missing_days": missing_count,
        "coverage_pct": coverage_pct,
        "date_range": f"{df['date'].min().strftime('%Y-%m-%d')} a {df['date'].max().strftime('%Y-%m-%d')}",
        "gap_info": gap_info[:5]  # Máximo 5 gaps para no saturar logs
    }


def log_detailed_summary(df_raw: pd.DataFrame, df_feat: pd.DataFrame, ticker: str, process_times: dict):
    """
    Genera un resumen detallado del proceso de construcción de features.
    """
    print("\n" + "="*70)
    print("📊 RESUMEN DETALLADO DEL PROCESO")
    print("="*70)
    
    # Información de cobertura temporal
    coverage = analyze_data_coverage(df_raw, ticker)
    
    print(f"📅 INFORMACIÓN DE COBERTURA TEMPORAL:")
    print(f"   • Rango total: {coverage['date_range']}")
    print(f"   • Días bursátiles esperados: {coverage['total_days']}")
    print(f"   • Días con datos: {coverage['trading_days']}")
    print(f"   • Días faltantes: {coverage['missing_days']}")
    print(f"   • Cobertura: {coverage['coverage_pct']:.2f}%")
    
    if coverage['gap_info']:
        print(f"   • Huecos significativos encontrados:")
        for gap in coverage['gap_info']:
            print(f"     - {gap['start']} a {gap['end']} ({gap['length']} días)")
    else:
        print(f"   • ✅ Sin huecos significativos detectados")
    
    # Conteos específicos
    print(f"\n📈 ESTADÍSTICAS DE DATOS:")
    print(f"   • Registros raw cargados: {len(df_raw):,}")
    print(f"   • Registros features generados: {len(df_feat):,}")
    print(f"   • Eficiencia de procesamiento: {(len(df_feat)/len(df_raw)*100 if len(df_raw) > 0 else 0):.1f}%")
    
    # Información de fechas
    if not df_feat.empty:
        feat_dates = pd.to_datetime(df_feat['date'])
        print(f"   • Rango features: {feat_dates.min().strftime('%Y-%m-%d')} a {feat_dates.max().strftime('%Y-%m-%d')}")
        print(f"   • Días únicos en features: {df_feat['date'].nunique()}")
    
    # Métricas de calidad de datos
    print(f"\n🔍 CALIDAD DE DATOS:")
    null_counts = df_feat.isnull().sum()
    problematic_cols = null_counts[null_counts > 0]
    if len(problematic_cols) > 0:
        print(f"   • Columnas con valores nulos:")
        for col, count in problematic_cols.items():
            pct = (count / len(df_feat)) * 100
            print(f"     - {col}: {count} nulos ({pct:.1f}%)")
    else:
        print(f"   • ✅ Sin valores nulos en features")
    
    # Tiempos de proceso
    print(f"\n⏱️  TIEMPOS DE PROCESO:")
    total_duration = sum(process_times.values())
    for stage, duration in process_times.items():
        pct = (duration / total_duration) * 100 if total_duration > 0 else 0
        print(f"   • {stage}: {duration:.2f}s ({pct:.1f}%)")
    print(f"   • TOTAL: {total_duration:.2f}s")
    
    print("="*70)
def ensure_analytics_table():
    """Crea el esquema y tabla analytics.daily_features si no existen"""
    sql = """
    CREATE SCHEMA IF NOT EXISTS analytics;

    CREATE TABLE IF NOT EXISTS analytics.daily_features (
        date DATE NOT NULL,
        ticker TEXT NOT NULL,

        year INT,
        month INT,
        day_of_week INT,

        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume BIGINT,

        return_close_open DOUBLE PRECISION,
        return_prev_close DOUBLE PRECISION,
        volatility_5d DOUBLE PRECISION,

        run_id TEXT,
        ingested_at_utc TIMESTAMP,

        PRIMARY KEY (date, ticker)
    );
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(sql)
        conn.commit()
        print("✅ Esquema 'analytics' y tabla 'daily_features' verificados")
    except Exception as e:
        print(f"❌ Error creando tabla: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def load_raw_prices(ticker: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """
    Carga datos desde raw.prices_daily para un ticker y rango de fechas.
    """
    conn = get_connection()

    base_query = """
        SELECT
            date,
            ticker,
            open,
            high,
            low,
            close,
            adj_close,
            volume
        FROM raw.prices_daily
        WHERE ticker = %s
    """

    params = [ticker]

    if start_date:
        base_query += " AND date >= %s"
        params.append(start_date)
    if end_date:
        base_query += " AND date <= %s"
        params.append(end_date)

    base_query += " ORDER BY date ASC;"

    try:
        df = pd.read_sql(base_query, conn, params=params)
        print(f"📥 Datos cargados desde raw.prices_daily: {len(df):,} registros para {ticker}")
        
        if not df.empty:
            dates = pd.to_datetime(df['date'])
            print(f"   • Rango raw: {dates.min().strftime('%Y-%m-%d')} a {dates.max().strftime('%Y-%m-%d')}")
            print(f"   • Días únicos: {df['date'].nunique()}")
            
        return df
    except Exception as e:
        print(f"❌ Error cargando datos raw: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()


def build_features(df_raw: pd.DataFrame, ticker: str, run_id: str) -> pd.DataFrame:
    """
    Construye las features diarias a partir de raw.prices_daily.
    """
    if df_raw.empty:
        print(f"⚠️ No hay datos en raw.prices_daily para ticker={ticker}.")
        return df_raw

    df = df_raw.copy()
    
    # Asegurar que date es datetime
    df["date"] = pd.to_datetime(df["date"]).dt.date

    print(f"🔄 Construyendo features para {len(df):,} registros...")

    # Identificación de día
    dt_index = pd.to_datetime(df["date"])
    df["year"] = dt_index.dt.year
    df["month"] = dt_index.dt.month
    df["day_of_week"] = dt_index.dt.dayofweek  # 0 = lunes, 6 = domingo

    # return_close_open = (close - open) / open
    df["return_close_open"] = (df["close"] - df["open"]) / df["open"]

    # return_prev_close = close / close_lag1 - 1
    df["close_lag1"] = df["close"].shift(1)
    df["return_prev_close"] = df["close"] / df["close_lag1"] - 1

    # volatilidad (std de retornos diarios últimos 5 días)
    df["ret_1d"] = df["return_prev_close"]
    df["volatility_5d"] = df["ret_1d"].rolling(window=5, min_periods=3).std()

    # metadatos
    df["run_id"] = run_id
    df["ingested_at_utc"] = datetime.utcnow()
    df["ticker"] = ticker  # asegurar ticker consistente

    # Seleccionar columnas finales
    cols = [
        "date",
        "ticker",
        "year",
        "month",
        "day_of_week",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_close_open",
        "return_prev_close",
        "volatility_5d",
        "run_id",
        "ingested_at_utc",
    ]
    
    # Filtrar filas con datos esenciales y eliminar NaN de features calculadas
    initial_count = len(df)
    df_final = df[cols].dropna(subset=["open", "close", "return_prev_close"])
    final_count = len(df_final)
    
    print(f"✅ Features construidas: {final_count:,} registros válidos")
    print(f"   • Registros descartados: {initial_count - final_count:,}")
    print(f"   • Eficiencia: {(final_count/initial_count*100):.1f}%")
    
    return df_final


def delete_existing_rows(ticker: str, start_date: str | None, end_date: str | None):
    """
    Borra filas existentes en analytics.daily_features para (ticker, rango de fechas),
    solo si overwrite=true.
    """
    conn = get_connection()
    cur = conn.cursor()

    query = """
        DELETE FROM analytics.daily_features
        WHERE ticker = %s
    """
    params = [ticker]

    if start_date:
        query += " AND date >= %s"
        params.append(start_date)
    if end_date:
        query += " AND date <= %s"
        params.append(end_date)

    try:
        # Contar antes de eliminar
        count_query = query.replace("DELETE", "SELECT COUNT(*)")
        cur.execute(count_query, params)
        count_before = cur.fetchone()[0]
        
        cur.execute(query, params)
        deleted = cur.rowcount
        conn.commit()
        print(f"🗑️  Filas eliminadas en analytics.daily_features: {deleted:,}")
        print(f"   • Verificación: {count_before:,} filas contadas antes de eliminar")
    except Exception as e:
        print(f"❌ Error eliminando filas existentes: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def upsert_features(df_feat: pd.DataFrame, overwrite: bool):
    """
    Inserta las filas en analytics.daily_features.
    - Si overwrite=False: ON CONFLICT DO NOTHING (no pisa).
    - Si overwrite=True: asumimos que ya se hizo DELETE antes.
    """
    if df_feat.empty:
        print("⚠️ DataFrame de features vacío. No se inserta nada.")
        return 0

    conn = get_connection()
    cur = conn.cursor()

    # Contar filas existentes antes de la inserción
    if not overwrite:
        ticker = df_feat.iloc[0]["ticker"]
        dates = df_feat["date"].unique()
        placeholders = ",".join(["%s"] * len(dates))
        
        count_query = f"""
            SELECT COUNT(*) 
            FROM analytics.daily_features 
            WHERE ticker = %s AND date IN ({placeholders})
        """
        cur.execute(count_query, [ticker] + dates.tolist())
        existing_before = cur.fetchone()[0]
    else:
        existing_before = 0

    # Si overwrite=False, usamos ON CONFLICT DO NOTHING para tener idempotencia.
    insert_sql = """
    INSERT INTO analytics.daily_features (
        date,
        ticker,
        year,
        month,
        day_of_week,
        open,
        high,
        low,
        close,
        volume,
        return_close_open,
        return_prev_close,
        volatility_5d,
        run_id,
        ingested_at_utc
    )
    VALUES %s
    """

    if not overwrite:
        insert_sql += """
        ON CONFLICT (date, ticker) DO NOTHING;
        """

    records = [
        (
            row["date"],
            row["ticker"],
            int(row["year"]),
            int(row["month"]),
            int(row["day_of_week"]),
            float(row["open"]) if pd.notnull(row["open"]) else None,
            float(row["high"]) if pd.notnull(row["high"]) else None,
            float(row["low"]) if pd.notnull(row["low"]) else None,
            float(row["close"]) if pd.notnull(row["close"]) else None,
            int(row["volume"]) if pd.notnull(row["volume"]) else None,
            float(row["return_close_open"]) if pd.notnull(row["return_close_open"]) else None,
            float(row["return_prev_close"]) if pd.notnull(row["return_prev_close"]) else None,
            float(row["volatility_5d"]) if pd.notnull(row["volatility_5d"]) else None,
            row["run_id"],
            row["ingested_at_utc"],
        )
        for _, row in df_feat.iterrows()
    ]

    inserted_count = 0
    try:
        print(f"💾 Insertando {len(records):,} filas en analytics.daily_features...")
        execute_values(cur, insert_sql, records)
        conn.commit()
        
        # Obtener conteo real de inserciones
        if overwrite:
            inserted_count = len(records)
            print(f"✅ Insert completado: {inserted_count:,} filas insertadas (overwrite=True)")
        else:
            # Para overwrite=False, contar las filas que realmente se insertaron
            cur.execute("SELECT COUNT(*) FROM analytics.daily_features WHERE run_id = %s", (df_feat.iloc[0]["run_id"],))
            total_with_run_id = cur.fetchone()[0]
            inserted_count = total_with_run_id - existing_before
            print(f"✅ Insert completado: {inserted_count:,} nuevas filas insertadas")
            print(f"   • Filas existentes evitadas: {len(records) - inserted_count:,}")
            
    except Exception as e:
        print(f"❌ Error en inserción: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
    
    return inserted_count
def main():
    # Diccionario para trackear tiempos de cada etapa
    process_times = {
        "setup": 0,
        "load_data": 0,
        "build_features": 0,
        "cleanup": 0,
        "insert_data": 0
    }
    
    start_time_total = time.time()
    setup_start = time.time()
    
    parser = argparse.ArgumentParser(description="Build daily features from raw.prices_daily")

    parser.add_argument("--mode", choices=["full", "by-date-range"], required=True,
                        help="full: procesa todo el histórico; by-date-range: solo un rango")
    parser.add_argument("--ticker", required=True, help="Ticker a procesar (ej: AAPL)")
    parser.add_argument("--start-date", required=False, help="YYYY-MM-DD (solo para by-date-range)")
    parser.add_argument("--end-date", required=False, help="YYYY-MM-DD (solo para by-date-range)")
    parser.add_argument("--run-id", required=True, help="Identificador de la corrida (run_id)")
    parser.add_argument("--overwrite", choices=["true", "false"], default="false",
                        help="Si true, borra y reescribe filas del rango.")

    args = parser.parse_args()

    # Validación de argumentos
    if args.mode == "by-date-range":
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date are required for by-date-range mode")
        
        # Validar formato de fechas
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            parser.error("Las fechas deben estar en formato YYYY-MM-DD")

    mode = args.mode
    ticker = args.ticker
    start_date = args.start_date
    end_date = args.end_date
    run_id = args.run_id
    overwrite = args.overwrite.lower() == "true"

    process_times["setup"] = time.time() - setup_start

    print("=" * 60)
    print(" FEATURE-BUILDER - INICIANDO EJECUCIÓN")
    print("=" * 60)
    print(f"Mode      : {mode}")
    print(f"Ticker    : {ticker}")
    print(f"StartDate : {start_date}")
    print(f"EndDate   : {end_date}")
    print(f"Run ID    : {run_id}")
    print(f"Overwrite : {overwrite}")
    print("=" * 60)

    try:
        # 1. Asegurar que existe la tabla analytics.daily_features
        ensure_analytics_table()

        # 2. Determinar rango de fechas
        load_start = time.time()
        if mode == "full":
            start_date_use = None
            end_date_use = None
            print("📅 Modo FULL: procesando todo el histórico")
        else:
            start_date_use = start_date
            end_date_use = end_date
            print(f"📅 Modo BY-DATE-RANGE: {start_date} a {end_date}")

        # 3. Cargar datos raw
        df_raw = load_raw_prices(ticker, start_date_use, end_date_use)
        process_times["load_data"] = time.time() - load_start

        if df_raw.empty:
            print("No hay datos en raw.prices_daily para los parámetros dados. Abortando.")
            return

        # 4. Construir features
        build_start = time.time()
        df_feat = build_features(df_raw, ticker, run_id)
        process_times["build_features"] = time.time() - build_start

        if df_feat.empty:
            print(" No se pudieron construir features. Abortando.")
            return

        # 5. Idempotencia: eliminar existentes si overwrite
        cleanup_start = time.time()
        if overwrite:
            delete_existing_rows(ticker, start_date_use, end_date_use)
        process_times["cleanup"] = time.time() - cleanup_start

        # 6. Insertar nuevas features
        insert_start = time.time()
        inserted_count = upsert_features(df_feat, overwrite=overwrite)
        process_times["insert_data"] = time.time() - insert_start

        # 7. Logs finales detallados
        total_duration = time.time() - start_time_total
        process_times["total"] = total_duration
        
        # Generar resumen detallado
        log_detailed_summary(df_raw, df_feat, ticker, process_times)
        
        # Resumen ejecutivo final
        print("\n RESUMEN EJECUTIVO:")
        print(f"   • Ticker: {ticker}")
        print(f"   • Modo: {mode}")
        print(f"   • Filas procesadas: {len(df_feat):,}")
        print(f"   • Filas insertadas: {inserted_count:,}")
        print(f"   • Duración total: {total_duration:.2f}s")
        print(f"   • Run ID: {run_id}")
        print(f"   • Estado: ✅ COMPLETADO EXITOSAMENTE")

    except Exception as e:
        print(f" ERROR CRÍTICO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()