"""
Loads raw vibration signals into SQLite and computes windowed statistical
features using SQL (not pandas) so the SQL is doing real work, not just
acting as a file format.

Schema:
    runs(run_id, condition)
    signals(run_id, sample_idx, value)

Feature engineering strategy:
Each 12,000-sample run is split into fixed-size windows (default 500 samples
= ~24 windows/run). For each window we compute mean, std, min, max, and RMS
via SQL aggregate functions grouped by (run_id, window_id). This mirrors how
you'd do feature engineering against a real production database instead of
loading everything into memory.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = "data/processed/vibration.db"
WINDOW_SIZE = 500  # samples per window (~24 windows per 12,000-sample run)


def build_database(csv_path="data/raw/vibration_signals.csv", db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    df = pd.read_csv(csv_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS signals;
    DROP TABLE IF EXISTS runs;

    CREATE TABLE runs (
        run_id INTEGER PRIMARY KEY,
        condition TEXT NOT NULL
    );

    CREATE TABLE signals (
        run_id INTEGER NOT NULL,
        sample_idx INTEGER NOT NULL,
        value REAL NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );

    CREATE INDEX idx_signals_run ON signals(run_id);
    """)

    runs = df[["run_id", "condition"]].drop_duplicates()
    runs.to_sql("runs", conn, if_exists="append", index=False)
    df[["run_id", "sample_idx", "value"]].to_sql(
        "signals", conn, if_exists="append", index=False
    )
    conn.commit()
    conn.close()
    print(f"Loaded {len(runs)} runs, {len(df)} samples into {db_path}")


def extract_features(db_path=DB_PATH, window_size=WINDOW_SIZE):
    """
    Pure-SQL windowed feature extraction. Returns one row per
    (run_id, window_id) with mean/std/min/max/rms + the run's condition label.
    """
    conn = sqlite3.connect(db_path)

    query = f"""
    WITH windowed AS (
        SELECT
            run_id,
            sample_idx / {window_size} AS window_id,
            value
        FROM signals
    ),
    stats AS (
        SELECT
            run_id,
            window_id,
            AVG(value) AS mean_val,
            MIN(value) AS min_val,
            MAX(value) AS max_val,
            -- SQLite has no built-in STDDEV; compute via AVG(x^2) - AVG(x)^2
            SQRT(AVG(value * value) - AVG(value) * AVG(value)) AS std_val,
            SQRT(AVG(value * value)) AS rms_val,
            COUNT(*) AS n_samples
        FROM windowed
        GROUP BY run_id, window_id
    )
    SELECT
        s.run_id,
        s.window_id,
        r.condition,
        s.mean_val,
        s.std_val,
        s.min_val,
        s.max_val,
        s.rms_val
    FROM stats s
    JOIN runs r ON r.run_id = s.run_id
    ORDER BY s.run_id, s.window_id;
    """

    features_df = pd.read_sql_query(query, conn)
    conn.close()
    return features_df


if __name__ == "__main__":
    build_database()
    features = extract_features()
    print(features.head(10))
    print(f"\nTotal feature rows: {len(features)}")
    features.to_csv("data/processed/features.csv", index=False)
    print("Saved data/processed/features.csv")
