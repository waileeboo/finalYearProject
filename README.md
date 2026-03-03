# Drift-Resilient Time Series Forecasting

**Author:** Wai Lee Boo (2625170)  
**Supervisor:** Professor Leandro Minku  
**Institution:** University of Birmingham  

---

## Overview

This project explores drift-adaptive forecasting for non-stationary time series, with primary emphasis on controlled synthetic drift scenarios. Four main model variants, PSO-LSTM, PSO-ELM, baseline LSTM, and baseline ELM are evaluated with trial-based online adaptive framewrok designed to detect and respond to distribuitional change.

The experimental design systematically evaluates model robustness under both linear and non-linear concept drift, including abrupt and gradual transitions. This controlled setting enables precise assessment of how optimisation and retraining strategies respond to different structural shifts in the data-generating process.

The research is guided by the following questions:

- **RQ1:** To what extent does PSO-based optimisation of output layer weights improve the robustness of neural forecasting models (ELM and LSTM) under different types of concept drift?
- **RQ2:** How effectively does drift-adaptive retraining improve forecasting performance compared to non-adaptive baselines across synthetic and real financial data?

---

## Project Structure


```
FINALYEARPROJECT/
├── backend/
│   ├── data/
│   │   ├── graph/                        # Generated plots and figures
│   │   ├── raw/                          # Downloaded stock price CSVs (GSPC, AAPL)
│   │   ├── results/                      # Experiment result CSVs
│   │   └── synthetic/                    # Generated synthetic drift datasets
│   │       ├── linear_abrupt_drift/
│   │       ├── linear_gradual_drift/
│   │       ├── nonlinear_abrupt_drift/
│   │       └── nonlinear_gradual_drift/
│   ├── notebooks/                        # Exploratory notebooks
│   └── src/
│       ├── data_utils/
│       │   ├── data_loader.py            # Load and split data
│       │   ├── fetch_data.py             # Download stock data via yfinance
│       │   ├── preprocess.py             # Feature engineering and scaling
│       │   ├── synthetic_generator.py    # Synthetic drift dataset generator
│       │   └── windowing.py              # Sliding window creation
│       ├── detectors/
│       │   └── drift_detector.py         # ADWIN, Page-Hinkley, KSWIN wrappers
│       ├── eda/                          # Exploratory data analysis scripts
│       ├── models/
│       │   ├── baselines/
│       │   │   ├── lstm_base.py          # Baseline LSTM
│       │   │   ├── elm_base.py           # Baseline ELM
│       │   │   └── arima_base.py         # Baseline ARIMA
│       │   ├── optimisers/
│       │   │   └── PSO.py                # Particle Swarm Optimisation core
│       │   ├── PSO_ELM.py                # PSO-optimised ELM
│       │   └── PSO_LSTM.py               # PSO-optimised LSTM
│       ├── training/
│       │   ├── online_eval.py            # Online evaluation loop (Phase 1 & 2)
│       │   ├── train_basearima.py        # Train baseline ARIMA (RQ1)
│       │   ├── train_baseelm.py          # Train baseline ELM (RQ1)
│       │   ├── train_baselstm.py         # Train baseline LSTM (RQ1)
│       │   ├── train_drift_real.py       # RQ2 — adaptive experiments on real data
│       │   ├── train_drift_synthetic.py  # RQ2 — detector comparison on synthetic data
│       │   ├── train_psoelm.py           # Train PSO-ELM (RQ1)
│       │   ├── train_psolstm.py          # Train PSO-LSTM (RQ1)
│       │   ├── train_utils.py            # Shared training helpers
│       │   └── tune_lstm.py              # Optuna hyperparameter tuning
│       └── utils/                        # Evaluation, logging, statistical tests
├── docs/
│   ├── MSci_Project_Proposal_v1.pdf
│   └── MSci_Project_report_final.pdf
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

### Install dependencies

### Fetch stock data

---

## Synthetic Drift Types

---

## Adaptive Framework Design

The online evaluation loop [`online_eval.py`](./src/training/online_eval.py) implements a **trial-based model selection** strategy:

1. A drift detector continuously monitors the prediction error stream. When drift is detected, the current active model is copied and retrained using a recent sliding window of data.
2. The retrained model is introduced into a challenger pool and runs silently alongside the active model for a fixed trial period of 20 steps. The challenger pool maintains a maximum of two candidate models at any time.
3. After the trial phase, the model achieving the lower mean absolute error (abs(MAE)) over the trial window is promoted to become the active model.
4. The displaced model is retained in the challenger pool with its error history cleared, allowing recovery if the data distribution reverts to a prior regime.
5. If the challenger pool is full when a new retrained model is generated:
    - The challenger with the highest trial MAE (the worst performing model) is removed.
    - The new challenger is inserted into the pool
    - This ensures bounded memory usage while preserving stronger historical candidates.
6. A cooldown period suppresses further drift detection after each trial to prevent cascading false alarms.

Notes: This is distinct from ensemble learning as only one model makes predictions at any time. Challenger models are evaluated passively and do not influence live outputs.

---

## Running Experiments

### RQ1 - To what extent does PSO-based optimisation of output layer weights improve the robustness of neural forecasting models (ELM and LSTM) under different types of concept drift?

Train and evaluate each model on real financial data and 4 different synthetic data without drift adaptation:

Results saved to `data/results/experiment_results.csv`.

### RQ2 — Drift-Adaptive Comparison

**Phase 1** — Compare drift detectors (ADWIN, Page-Hinkley, KSWIN) on synthetic data:

Results saved to `data/results/rq2_phase1_results.csv`.

**Phase 2** — Compare all four model types using KSWIN across 30 random seeds:

Results saved to `data/results/rq2_phase2_results.csv`.

## Key Configuration

Edit `config.py` to change feature columns or tickers.
Edit `paths.py` to change data and results directories.

---

## Documents

[📄 View Proposal](./docs/MSci_Project_Proposal_v1.pdf)
[📄 Final Report](./docs/MSci_Project_Proposal_v1.pdf)