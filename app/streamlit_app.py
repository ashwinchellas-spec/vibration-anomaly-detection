"""
Streamlit demo: upload a CSV of vibration readings (single column of values,
or a 'value' column) and see whether the trained autoencoder flags it as
healthy or anomalous, with a reconstruction-error plot.

Run locally:
    streamlit run app/streamlit_app.py
"""

import json
import sys
import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import make_windows  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "autoencoder.keras")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "config.json")


@st.cache_resource
def load_model_and_config():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return model, config


def main():
    st.set_page_config(page_title="Vibration Anomaly Detector", layout="wide")
    st.title("Bearing Vibration Anomaly Detection")
    st.caption(
        "LSTM autoencoder trained only on healthy vibration signals. "
        "High reconstruction error = signal doesn't match learned healthy pattern."
    )

    model, config = load_model_and_config()
    window_length = config["window_length"]
    stride = config["stride"]
    mean, std = config["mean"], config["std"]
    threshold = config["threshold"]

    st.sidebar.header("Input")
    source = st.sidebar.radio("Signal source", ["Upload CSV", "Use sample synthetic signal"])

    values = None
    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("CSV with a numeric 'value' column (or single column)", type="csv")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            col = "value" if "value" in df.columns else df.columns[0]
            values = df[col].to_numpy(dtype=float)
    else:
        sample_type = st.sidebar.selectbox(
            "Condition", ["healthy", "inner_race_fault", "outer_race_fault", "ball_fault"]
        )
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
        from generate_data import FAULT_TYPES  # noqa: E402
        values = FAULT_TYPES[sample_type]()

    if values is None:
        st.info("Upload a CSV or pick a sample signal from the sidebar to run detection.")
        return

    if len(values) < window_length:
        st.error(f"Signal needs at least {window_length} samples, got {len(values)}.")
        return

    windows = make_windows(values, window_length, stride)[..., np.newaxis]
    windows_norm = (windows - mean) / std
    recon = model.predict(windows_norm, verbose=0)
    errors = np.mean(np.square(windows_norm - recon), axis=(1, 2))

    n_anomalous = int((errors > threshold).sum())
    pct_anomalous = 100 * n_anomalous / len(errors)

    col1, col2, col3 = st.columns(3)
    col1.metric("Windows analyzed", len(errors))
    col2.metric("Flagged anomalous", f"{n_anomalous} ({pct_anomalous:.0f}%)")
    verdict = "ANOMALY DETECTED" if pct_anomalous > 20 else "Signal looks healthy"
    col3.metric("Verdict", verdict)

    st.subheader("Raw signal")
    st.line_chart(pd.DataFrame({"value": values[:2000]}))

    st.subheader("Reconstruction error per window")
    error_df = pd.DataFrame({"window": range(len(errors)), "error": errors, "threshold": threshold})
    st.line_chart(error_df.set_index("window")[["error", "threshold"]])


if __name__ == "__main__":
    main()
