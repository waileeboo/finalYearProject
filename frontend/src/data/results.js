// ── RQ1: Static Baseline Rankings (Return MAE, Friedman/Wilcoxon) ────────────
export const rq1Rankings = [
  { model: 'PSO-ELM',  rank: 1.79, color: '#c9a84c' },
  { model: 'LSTM',     rank: 2.58, color: '#9d7bb5' },
  { model: 'PSO-LSTM', rank: 2.73, color: '#5a9e7c' },
  { model: 'ELM',      rank: 2.96, color: '#c47a3a' },
  { model: 'ARIMA',    rank: 4.94, color: '#b05c5c' },
]

export const rq1ReturnMAE = [
  { model: 'PSO-ELM',  returnMAE: 0.00512, color: '#c9a84c' },
  { model: 'LSTM',     returnMAE: 0.00598, color: '#9d7bb5' },
  { model: 'PSO-LSTM', returnMAE: 0.00613, color: '#5a9e7c' },
  { model: 'ELM',      returnMAE: 0.00631, color: '#c47a3a' },
  { model: 'ARIMA',    returnMAE: 0.01120, color: '#b05c5c' },
]

// ── RQ2: Adaptive vs Baseline on Real S&P 500 data (Price MAE) ──────────────
export const rq2RealPriceMAE = [
  { model: 'LSTM',     baseline: 33.9, adaptive: 19.8, improvement: 41 },
  { model: 'PSO-LSTM', baseline: 28.6, adaptive: 19.7, improvement: 31 },
  { model: 'PSO-ELM',  baseline: 20.2, adaptive: 20.2, improvement: 0  },
  { model: 'ELM',      baseline: 20.1, adaptive: 20.1, improvement: 0  },
]

export const rq2ReturnMAERankings = [
  { model: 'PSO-ELM (static)',  rank: 2.23, color: '#c9a84c' },
  { model: 'LSTM (static)',     rank: 3.90, color: '#9d7bb5' },
  { model: 'PSO-ELM adaptive', rank: 4.10, color: '#5a9e7c' },
  { model: 'ELM (static)',     rank: 4.41, color: '#c47a3a' },
  { model: 'PSO-LSTM adaptive',rank: 4.98, color: '#b05c5c' },
  { model: 'LSTM adaptive',    rank: 5.41, color: '#f59e0b' },
  { model: 'ELM adaptive',     rank: 6.48, color: '#5a8fa8' },
]

export const rq2RetrainTimeRankings = [
  { model: 'ELM adaptive',      rank: 1.00, color: '#c9a84c' },
  { model: 'LSTM adaptive',     rank: 2.00, color: '#9d7bb5' },
  { model: 'PSO-ELM adaptive',  rank: 3.01, color: '#5a9e7c' },
  { model: 'PSO-LSTM adaptive', rank: 3.99, color: '#b05c5c' },
]

// ── RQ3: TDTR vs Fixed-Interval Retraining ───────────────────────────────────
export const rq3Comparison = [
  { label: 'TDTR (KSWIN)',         priceMAE: 19.8, retrains: 8,  totalTime: 62,   color: '#c9a84c' },
  { label: 'Fixed-Interval',       priceMAE: 19.6, retrains: 18, totalTime: 130,  color: '#b05c5c' },
]

export const rq3EfficiencyData = [
  { metric: 'Price MAE',       tdtr: 19.8, fixed: 19.6 },
  { metric: 'Return MAE (×100)', tdtr: 0.72, fixed: 0.71 },
  { metric: 'Retrains',        tdtr: 8,    fixed: 18   },
  { metric: 'Total Time (s)',  tdtr: 62,   fixed: 130  },
]

// ── RQ4: Conventional ML on Real Data (GSPC) ─────────────────────────────────
export const rq4RealResults = [
  { model: 'RF (baseline)',   priceMAE: 20.8, returnMAE: 0.00753, color: '#c47a3a' },
  { model: 'RF (TDTR)',       priceMAE: 20.4, returnMAE: 0.00745, color: '#c9a84c' },
  { model: 'SVR (baseline)',  priceMAE: 23.7, returnMAE: 0.00913, color: '#c47a3a' },
  { model: 'SVR (TDTR)',      priceMAE: 23.7, returnMAE: 0.00911, color: '#b05c5c' },
]

// ── RQ5: Limited Training (20%) Results ──────────────────────────────────────
export const rq5UnseenGain = [
  { model: 'LSTM',     baseline: 0.112, adaptive: 0.048, gain: 57 },
  { model: 'PSO-LSTM', baseline: 0.098, adaptive: 0.029, gain: 70 },
  { model: 'ELM',      baseline: 0.087, adaptive: 0.071, gain: 18 },
  { model: 'PSO-ELM',  baseline: 0.079, adaptive: 0.068, gain: 14 },
]

// ── RQ2: Synthetic Return MAE table — matches report Table (mean, std over 30 runs) ──
export const rq2SyntheticTable = {
  staticModels:   ['ARIMA', 'LSTM', 'ELM', 'PSO-LSTM', 'PSO-ELM'],
  adaptiveModels: ['LSTM-A', 'PSO-LSTM-A', 'ELM-A', 'PSO-ELM-A'],
  rows: [
    {
      drift: 'Linear Abrupt',
      static:   [
        { mean: 0.200, std: 0.117 },
        { mean: 0.040, std: 0.038 },
        { mean: 0.085, std: 0.064 },
        { mean: 0.048, std: 0.006 },
        { mean: 0.050, std: 0.031 },
      ],
      adaptive: [
        { mean: 0.037, std: 0.009 },
        { mean: 0.053, std: 0.033 },
        { mean: 0.127, std: 0.121 },
        { mean: 0.090, std: 0.056 },
      ],
    },
    {
      drift: 'Linear Gradual',
      static:   [
        { mean: 0.211, std: 0.088 },
        { mean: 0.029, std: 0.005 },
        { mean: 0.037, std: 0.010 },
        { mean: 0.088, std: 0.006 },
        { mean: 0.023, std: 0.002 },
      ],
      adaptive: [
        { mean: 0.027, std: 0.006 },
        { mean: 0.030, std: 0.012 },
        { mean: 0.067, std: 0.054 },
        { mean: 0.041, std: 0.015 },
      ],
    },
    {
      drift: 'Non-linear Abrupt',
      static:   [
        { mean: 0.227, std: 0.162 },
        { mean: 0.083, std: 0.050 },
        { mean: 0.057, std: 0.033 },
        { mean: 0.034, std: 0.056 },
        { mean: 0.034, std: 0.005 },
      ],
      adaptive: [
        { mean: 0.059, std: 0.033 },
        { mean: 0.045, std: 0.018 },
        { mean: 0.082, std: 0.074 },
        { mean: 0.075, std: 0.054 },
      ],
    },
    {
      drift: 'Non-linear Gradual',
      static:   [
        { mean: 0.124, std: 0.024 },
        { mean: 0.022, std: 0.005 },
        { mean: 0.019, std: 0.001 },
        { mean: 0.019, std: 0.001 },
        { mean: 0.017, std: 0.001 },
      ],
      adaptive: [
        { mean: 0.020, std: 0.003 },
        { mean: 0.019, std: 0.002 },
        { mean: 0.020, std: 0.002 },
        { mean: 0.019, std: 0.002 },
      ],
    },
  ],
}

// ── RQ5: Synthetic Return MAE — Baseline vs TDTR (mean over 30 runs) ─────────
export const rq5SyntheticTable = [
  {
    model: 'LSTM',
    rows: [
      { drift: 'Linear Abrupt',      base: 0.13657, tdtr: 0.04624 },
      { drift: 'Linear Gradual',     base: 0.06916, tdtr: 0.04105 },
      { drift: 'Nonlinear Abrupt',   base: 0.15577, tdtr: 0.06625 },
      { drift: 'Nonlinear Gradual',  base: 0.03019, tdtr: 0.02277 },
    ],
    color: '#e8e8e8',
  },
  {
    model: 'PSO-LSTM',
    rows: [
      { drift: 'Linear Abrupt',      base: 0.07585, tdtr: 0.08570 },
      { drift: 'Linear Gradual',     base: 0.05884, tdtr: 0.04974 },
      { drift: 'Nonlinear Abrupt',   base: 0.05169, tdtr: 0.05910 },
      { drift: 'Nonlinear Gradual',  base: 0.01777, tdtr: 0.01866 },
    ],
    color: '#e8e8e8',
  },
  {
    model: 'ELM',
    rows: [
      { drift: 'Linear Abrupt',      base: 0.30553, tdtr: 0.09024 },
      { drift: 'Linear Gradual',     base: 0.26302, tdtr: 0.05214 },
      { drift: 'Nonlinear Abrupt',   base: 0.19815, tdtr: 0.08463 },
      { drift: 'Nonlinear Gradual',  base: 0.02424, tdtr: 0.01982 },
    ],
    color: '#e8e8e8',
  },
  {
    model: 'PSO-ELM',
    rows: [
      { drift: 'Linear Abrupt',      base: 0.07420, tdtr: 0.06074 },
      { drift: 'Linear Gradual',     base: 0.03357, tdtr: 0.03938 },
      { drift: 'Nonlinear Abrupt',   base: 0.04884, tdtr: 0.04338 },
      { drift: 'Nonlinear Gradual',  base: 0.01757, tdtr: 0.01893 },
    ],
    color: '#e8e8e8',
  },
]

// ── Diagrams mapping ──────────────────────────────────────────────────────────
export const diagrams = {
  systemOverview:      '/diagrams/FYP_System_overview.png',
  arimaBaseline:       '/diagrams/arima_baseline.png',
  syntheticDrift:      '/diagrams/synthetic_drift_series.png',
  poolActivity:        '/diagrams/plot_pool_activity.png',
  rq1CdReturnMAE:      '/diagrams/rq1_cd_diagram_return_mae.png',
  rq2CdReturnMAE:      '/diagrams/rq2_cd_diagram_total_return_mae.png',
  rq2CdRetrainTime:    '/diagrams/rq2_cd_diagram_total_retrain_time3.png',
  rq5ELMRolling:       '/diagrams/fig_rq5_concept_structure_validation.png',
  rq5LSTMRolling:      '/diagrams/fig_rq5_concept_structure_validation_base_lstm.png',
}
