import os
import psycopg2
from psycopg2.extras import execute_values

def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )

def insert_prices(df):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO raw.prices_daily
    (date, ticker, open, high, low, close, adj_close, volume, run_id, ingested_at_utc, source_name)
    VALUES %s
    ON CONFLICT (date, ticker) DO NOTHING;
    """

    records = [
        (
            row.date,
            row.ticker,
            row.open,
            row.high,
            row.low,
            row.close,
            row.adj_close,
            row.volume,
            row.run_id,
            row.ingested_at_utc,
            row.source_name
        )
        for _, row in df.iterrows()
    ]

    execute_values(cursor, query, records)
    conn.commit()
    cursor.close()
    conn.close()
