# Bearing Vibration Anomaly Detection

An LSTM autoencoder that learns what "healthy" bearing vibration looks like
and flags anomalous signals — the same class of technique used in real
predictive-maintenance and structural health monitoring (SHM) systems.

**[Live demo](#deployment)** · Build → Train → Deploy pipeline using Python, SQL, and TensorFlow.

## Problem

Rotating machinery (motors, gearboxes, bearings) develops faults long before
they cause a failure. Vibration sensors pick up characteristic signatures of
these faults — but labeling every possible fault type is expensive and
impractical in production. This project uses an **unsupervised** approach:
train only on healthy signals, then flag anything that doesn't reconstruct
well as anomalous.

## Architecture

```
Raw signals (CSV) → SQLite (SQL feature engineering) → LSTM Autoencoder (TensorFlow)
                                                              │
                                                    reconstruction error
                                                              │
                                                     anomaly threshold
                                                              │
                                                      Streamlit demo app
```

- **Data**: Synthetic vibration signals modeled after the structure of the
  public [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
  dataset — healthy + inner race / outer race / ball fault conditions, each
  with a physically distinct periodic-impact signature. See
  `src/generate_data.py` for how to swap in the real CWRU `.mat` files.
- **SQL**: Signals are loaded into SQLite; all windowed statistical features
  (mean, std, min, max, RMS) are computed with a single SQL query using
  window bucketing and aggregate functions — not pandas. See `src/database.py`.
- **Model**: LSTM autoencoder (TensorFlow/Keras) trained only on healthy
  windows. Anomaly score = mean squared reconstruction error. Threshold set
  at the 95th percentile of healthy validation error. See `src/model.py` and
  `src/train.py`.
- **Deployment**: Streamlit app for interactive inspection — upload a signal
  CSV or pick a sample condition, see the anomaly verdict and reconstruction
  error plot. See `app/streamlit_app.py`.

## Results

Evaluated on held-out windows across all four conditions:

| Condition          | Mean reconstruction error | % windows flagged anomalous |
|---------------------|---------------------------|------------------------------|
| healthy              | 0.19                       | 5%   (≈ false-positive rate, by threshold design) |
| ball_fault           | 0.39                       | 100% |
| outer_race_fault     | 0.60                       | 100% |
| inner_race_fault     | 1.04                       | 100% |

## Project structure

```
vibration-anomaly-detection/
├── src/
│   ├── generate_data.py   # synthetic signal generator (swap for real CWRU data)
│   ├── database.py        # SQLite loading + SQL feature engineering
│   ├── model.py            # LSTM autoencoder architecture
│   └── train.py            # training + evaluation + threshold selection
├── app/
│   └── streamlit_app.py    # deployment: interactive demo
├── models/                 # trained model + config (committed, ~200KB)
├── data/                   # raw/processed data (gitignored, regenerate locally)
├── requirements.txt
└── README.md
```

## Running locally

```bash
git clone <this-repo-url>
cd vibration-anomaly-detection
pip install -r requirements.txt

# Regenerate data + retrain (optional — a trained model is already committed)
python src/generate_data.py
python src/database.py
python src/train.py

# Launch the demo
streamlit run app/streamlit_app.py
```

## Deployment

Deployed as a free-tier Streamlit app on [Hugging Face Spaces](https://huggingface.co/spaces).
To deploy your own copy: create a new Space (SDK: Streamlit), push this repo
to it, and it builds automatically from `requirements.txt` and
`app/streamlit_app.py`.

## Notes on the synthetic data

The CWRU dataset's host (`engineering.case.edu`) wasn't reachable from the
environment this was built in, so `src/generate_data.py` generates
physically-motivated synthetic signals instead: a base rotational sine wave
plus periodic decaying-impact pulses at realistic bearing defect
frequencies, matching the *structure* of real bearing fault data even though
the exact numbers are synthetic. Swapping in the real `.mat` files from CWRU
requires no changes downstream — just produce the same
`(run_id, condition, sample_idx, value)` format.
