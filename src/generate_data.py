"""
Generates synthetic bearing vibration signals that mimic the structure of the
real CWRU (Case Western Reserve University) Bearing Data Center dataset:
https://engineering.case.edu/bearingdatacenter

Why synthetic data:
This sandbox environment cannot reach case.edu, so this script creates
physically-motivated stand-in signals (healthy + 3 common fault types) so the
full pipeline (SQL storage -> feature engineering -> LSTM autoencoder ->
Streamlit app) can be built and tested end-to-end right now.

To use the REAL dataset instead:
1. Download the 12k drive-end .mat files from the CWRU site above
   (Normal baseline + Inner Race / Ball / Outer Race fault files).
2. Replace this script's output with real signals loaded via scipy.io.loadmat.
3. Everything downstream (database schema, feature queries, model, app)
   works unchanged as long as you produce the same columns:
   run_id, condition, sample_idx, value
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
SAMPLE_RATE = 12000  # Hz, matches CWRU 12k drive-end sampling rate
DURATION_S = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_S)
T = np.arange(N_SAMPLES) / SAMPLE_RATE


def healthy_signal(noise_scale=0.05):
    """Smooth rotational vibration, low-amplitude broadband noise only."""
    base = 0.3 * np.sin(2 * np.pi * 30 * T)  # shaft rotation frequency
    noise = RNG.normal(0, noise_scale, N_SAMPLES)
    return base + noise


def fault_signal(fault_freq, impact_amplitude, noise_scale=0.08):
    """
    Bearing faults produce periodic IMPACT pulses at a characteristic defect
    frequency, riding on top of the normal rotational vibration. This is the
    real physical signature CWRU-style models are trained to detect.
    """
    base = 0.3 * np.sin(2 * np.pi * 30 * T)
    impacts = np.zeros(N_SAMPLES)
    period = int(SAMPLE_RATE / fault_freq)
    for start in range(0, N_SAMPLES, period):
        end = min(start + 20, N_SAMPLES)
        decay = np.exp(-np.linspace(0, 8, end - start))
        impacts[start:end] += impact_amplitude * decay
    noise = RNG.normal(0, noise_scale, N_SAMPLES)
    return base + impacts + noise


FAULT_TYPES = {
    "healthy": lambda: healthy_signal(),
    "inner_race_fault": lambda: fault_signal(fault_freq=157.9, impact_amplitude=1.2),
    "outer_race_fault": lambda: fault_signal(fault_freq=104.6, impact_amplitude=0.9),
    "ball_fault": lambda: fault_signal(fault_freq=68.3, impact_amplitude=0.7),
}


def generate_dataset(n_runs_per_condition=40):
    rows = []
    run_id = 0
    for condition, gen_fn in FAULT_TYPES.items():
        for _ in range(n_runs_per_condition):
            signal = gen_fn()
            for idx, value in enumerate(signal):
                rows.append((run_id, condition, idx, float(value)))
            run_id += 1
    df = pd.DataFrame(rows, columns=["run_id", "condition", "sample_idx", "value"])
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_dataset(n_runs_per_condition=40)
    out_path = "data/raw/vibration_signals.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {df['run_id'].nunique()} runs, {len(df)} rows -> {out_path}")
    print(df.groupby("condition")["run_id"].nunique())
