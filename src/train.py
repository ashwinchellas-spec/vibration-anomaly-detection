import os
import sqlite3
import numpy as np
import pandas as pd
import json

from model import build_autoencoder, make_windows

DB_PATH = "data/processed/vibration.db"
WINDOW_LENGTH = 200
STRIDE = 100
MODELS_DIR = "models"


def load_signal_windows(condition=None, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    if condition:
        query = """
            SELECT s.run_id, s.sample_idx, s.value
            FROM signals s JOIN runs r ON r.run_id = s.run_id
            WHERE r.condition = ?
            ORDER BY s.run_id, s.sample_idx
        """
        df = pd.read_sql_query(query, conn, params=(condition,))
    else:
        query = """
            SELECT s.run_id, s.sample_idx, s.value, r.condition
            FROM signals s JOIN runs r ON r.run_id = s.run_id
            ORDER BY s.run_id, s.sample_idx
        """
        df = pd.read_sql_query(query, conn)
    conn.close()

    all_windows = []
    labels = []
    for run_id, group in df.groupby("run_id"):
        values = group["value"].to_numpy()
        w = make_windows(values, WINDOW_LENGTH, STRIDE)
        all_windows.append(w)
        if not condition:
            labels.extend([group["condition"].iloc[0]] * len(w))

    windows = np.concatenate(all_windows, axis=0)
    return (windows, labels) if not condition else windows


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading healthy windows for training...")
    healthy_windows = load_signal_windows(condition="healthy")
    healthy_windows = healthy_windows[..., np.newaxis]  # (n, window_len, 1)

    # normalize using healthy-data statistics only (avoids leaking fault info)
    mean, std = healthy_windows.mean(), healthy_windows.std()
    healthy_norm = (healthy_windows - mean) / std

    split = int(0.85 * len(healthy_norm))
    X_train, X_val = healthy_norm[:split], healthy_norm[split:]
    print(f"Train windows: {X_train.shape}, Val windows: {X_val.shape}")

    model = build_autoencoder(window_length=WINDOW_LENGTH)
    model.summary()

    model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=8,
        batch_size=128,
        verbose=2,
    )

    # Determine anomaly threshold from healthy validation reconstruction error
    val_recon = model.predict(X_val, verbose=0)
    val_errors = np.mean(np.square(X_val - val_recon), axis=(1, 2))
    threshold = float(np.percentile(val_errors, 95))
    print(f"Anomaly threshold (95th pct of healthy val error): {threshold:.6f}")

    # Evaluate against all conditions
    print("\nEvaluating against all conditions...")
    all_windows, labels = load_signal_windows()
    all_windows = all_windows[..., np.newaxis]
    all_norm = (all_windows - mean) / std
    recon = model.predict(all_norm, verbose=0)
    errors = np.mean(np.square(all_norm - recon), axis=(1, 2))

    results = pd.DataFrame({"condition": labels, "error": errors})
    results["predicted_anomaly"] = results["error"] > threshold
    summary = results.groupby("condition").agg(
        mean_error=("error", "mean"),
        flagged_pct=("predicted_anomaly", "mean"),
    )
    summary["flagged_pct"] = (summary["flagged_pct"] * 100).round(1)
    print(summary)

    model.save(os.path.join(MODELS_DIR, "autoencoder.keras"))
    with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
        json.dump({
            "window_length": WINDOW_LENGTH,
            "stride": STRIDE,
            "mean": float(mean),
            "std": float(std),
            "threshold": threshold,
        }, f, indent=2)
    print(f"\nSaved model + config to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
