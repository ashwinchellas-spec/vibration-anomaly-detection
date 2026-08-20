"""
LSTM Autoencoder for vibration anomaly detection.

Approach:
Train ONLY on healthy signal windows. The autoencoder learns to compress and
reconstruct normal vibration patterns well. When fed a faulty window, it
reconstructs poorly (high reconstruction error) because it never learned that
pattern -> reconstruction error becomes the anomaly score. This is a standard
unsupervised approach used in real SHM/predictive-maintenance systems, and is
the same underlying idea as the damage-detection logic in physics-informed
SHM digital twins: deviation from a learned "healthy" baseline signals damage.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(window_length, n_features=1, latent_dim=8):
    inputs = keras.Input(shape=(window_length, n_features))

    # Encoder
    x = layers.LSTM(32, return_sequences=True)(inputs)
    x = layers.LSTM(latent_dim, return_sequences=False)(x)

    # Bottleneck repeated across the sequence so the decoder LSTM has
    # something to unroll against
    x = layers.RepeatVector(window_length)(x)

    # Decoder
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def make_windows(raw_values, window_length, stride):
    """Slice a 1D signal array into overlapping windows for the autoencoder."""
    windows = []
    for start in range(0, len(raw_values) - window_length + 1, stride):
        windows.append(raw_values[start:start + window_length])
    return np.array(windows)
