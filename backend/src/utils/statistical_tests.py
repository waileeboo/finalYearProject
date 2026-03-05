    
"""
Statistical Test for RQ1 and RQ2 

Run two Friedman + Nemenyi tests and produce two CD Diagrams: 
1. Return_MAE
2. total_retrain_time 

All 8 configurations are tested together in one combined analysis: 
    Static (RQ1): PSO-LSTM, LSTM, ELM, PSO-ELM, ARIMA
    Adaptive (RQ2): PSO-LSTM-K, LSTM-K, ELM-K, PSO-ELM-K
"""


import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.paths import RESULTS_DIR, RESULTS_FILE_RQ1, RESULTS_FILE_RQ2_PHASE2

# Configuration 
ALPHA = 0.05 # Significance level for Friedman test

# Metrics column names in the CSV
METRIC_MAE = "Return_MAE"
METRIC_TIME = "total_retrain_time"

# ARIMA is static but excluded from formal tests (single run, no variance)
ARIMA_MODEL = "ARIMA"

# Static models (RQ1) from RESULTS_FILE_RQ1 
# Names must match the model column in the results CSV exactly
STATIC_MODELS = [
    "PSO_LSTM",
    "PSO_ELM",
    "LSTM",
    "ELM",
]

# Adaptive models (RQ2) from RESULTS_FILE_RQ2_PHASE2
# Renamed from {model}_kswin to {model}_adaptive for clarity
ADAPTIVE_MODELS = [
    "PSO_LSTM_adaptive",
    "PSO_ELM_adaptive",
    "LSTM_adaptive",
    "ELM_adaptive",
]


# All models included in formal statistical tests (excludes ARIMA)
ALL_MODELS = STATIC_MODELS + ADAPTIVE_MODELS


# add significance label 
def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "highly significant"
    elif p_value < 0.01:
        return "very significant"
    elif p_value < 0.05:
        return "significant"
    else:
        return "not significant"

# extract drift type name 
def extract_drift_type(dataset: str) -> str:
    """Strip suffix from dataset name."""
    parts = dataset.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return dataset


# Data Loading 
def load_synthetic_results() -> pd.DataFrame:
    """ Load and combine RQ1 and RQ2 results 
    Normalise model names so they match all models
    """
    # RQ1 static results and normalise names
    rq1_df = pd.read_csv(RESULTS_FILE_RQ1)
    rq1_df["model"] = rq1_df["model"].str.replace("_Baseline", "", regex=False)

    rq1_df = rq1_df[rq1_df["dataset"] != "GSPC"].copy()  # exclude real data
    
    # RQ2 adaptive results and normalise names
    rq2_df = pd.read_csv(RESULTS_FILE_RQ2_PHASE2)
    rq2_df["model"] = rq2_df["model"].str.replace("_kswin", "_adaptive", regex=False)
    rq2_df = rq2_df[rq2_df["dataset"] != "GSPC"].copy()  # exclude real data

    # Combine into one DataFrame
    df = pd.concat([rq1_df, rq2_df], ignore_index=True)
    
    # add drift_type column for stripping seed suffix from dataset name 
    df["drift_type"] = df["dataset"].apply(extract_drift_type)
    
    # Debug — find suspicious MAE values
    print("\nMax MAE per model:")
    print(df.groupby("model")[METRIC_MAE].max().round(4).to_string())

    print("\nRows with MAE > 1.0 (outliers):")
    bad_rows = df[df[METRIC_MAE] > 0.3][["model", "dataset", METRIC_MAE]]
    print(bad_rows.to_string())

    print("\nRow counts per model per drift type:")
    print(df.groupby(["model", "drift_type"]).size().to_string())

    print(f"RQ1 rows loaded      : {len(rq1_df)}")
    print(f"RQ2 rows loaded      : {len(rq2_df)}")
    print(f"Total rows           : {len(df)}")
    print(f"All models found     : {sorted(df['model'].unique().tolist())}")
    print(f"Models for testing   : {ALL_MODELS}")
    
    return df


# desriptive statistics 
def print_descriptive_statistics(df: pd.DataFrame, metric: str) -> None:
    """
    print mean, std, median, q1, q3, IQR grouped by drift_type and model 
    sort by mean ascendign with each drift type
    """
    print(f"\nDescriptive Statistics for {metric}:\n")
    
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in DataFrame columns.")
        return
    
    summary = df.groupby(["drift_type", "model"])[metric].agg(
        Count="count",
        Mean="mean",
        Std="std",
        Median="median",
    )
    
    summary["Q1"] = df.groupby(["drift_type", "model"])[metric].quantile(0.25)
    summary["Q3"] = df.groupby(["drift_type", "model"])[metric].quantile(0.75)
    summary["IQR"] = summary["Q3"] - summary["Q1"]
    
    # Sort by drift_type then mean asscending
    summary = summary.sort_values(["drift_type", "Mean"])
    
    print(summary.round(6).to_string())


# pivot table 

def build_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build pivot table where row = seed and columns = models"""
     
    if metric not in df.columns:
        print(f"  Metric '{metric}' not found. Cannot build pivot.")
        return pd.DataFrame()

    # Assign seed numbers if not already present in the CSV
    if "seed" not in df.columns:
        df = df.copy()
        df["seed"] = df.groupby("model").cumcount() + 1

    pivot = df.pivot_table(
        index="seed",
        columns="model",
        values=metric,
        aggfunc="mean",
    )

    # Filter to only the 8 testable models — ARIMA dropped automatically here
    cols_present = [m for m in ALL_MODELS if m in pivot.columns]
    pivot = pivot[cols_present].dropna()

    return pivot
    



# Friedman Test
def run_friedman_test(pivot: pd.DataFrame, metric: str) -> tuple:
    """
    Run the Friedman test across all 8 model configurations.
    Returns (chi_square_statistic, p_value, avg_ranks)
    """
    
    print(f"\nFriedman Test — {metric}")
    print("################################################################")
    print(f"Number of seeds (blocks) : {pivot.shape[0]}")
    print(f"Number of models         : {pivot.shape[1]}")
    print(f"Models tested            : {pivot.columns.tolist()}\n")

    # Rank models within each seed — rank 1 = best = lowest metric value
    ranks     = pivot.rank(axis=1)
    avg_ranks = ranks.mean().sort_values()

    print("Average Ranks (lower = better):")
    for model, rank in avg_ranks.items():
        print(f"  {model:<35} {rank:.4f}")

    # Run Friedman — pass each model column as a separate array
    stat, p_value = friedmanchisquare(*[pivot[col].values for col in pivot.columns])

    label = significance_label(p_value)
    print(f"\nFriedman chi-square = {stat:.4f}")
    print(f"p-value             = {p_value:.2e}  ({label})")

    if p_value < ALPHA:
        print(f"\nResult: Significant difference detected (p < {ALPHA}).")
        print("Proceeding to Nemenyi post-hoc test.")
    else:
        print(f"\nResult: No significant difference detected (p >= {ALPHA}).")
        print("Stopping — post-hoc test not applicable.")

    return stat, p_value, avg_ranks
    


# Nemenyi Post-hoc Test
def run_nemenyi_test(pivot: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Run Nemenyi post-hoc test after a significatn friedman test"""
    print(f"\nNemenyi Post-hoc Test — {metric}")
    print("################################################################")
    nemenyi         = sp.posthoc_nemenyi_friedman(pivot.values)
    nemenyi.index   = pivot.columns
    nemenyi.columns = pivot.columns

    print("Pairwise p-values (p < 0.05 = significantly different):\n")
    print(nemenyi.round(4).to_string())

    # Print a clean list of only the significant pairs
    print("\nSignificant pairs (p < 0.05):")
    found_any = False
    models = pivot.columns.tolist()
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            p = nemenyi.iloc[i, j]
            if p < ALPHA:
                label = significance_label(p)
                print(f"  {models[i]:<35} vs  {models[j]:<35} p={p:.4f} ({label})")
                found_any = True
    if not found_any:
        print("  None found.")

    return nemenyi


def plot_cd_diagram(avg_ranks: pd.Series, nemenyi: pd.DataFrame, metric: str, title: str) -> None:
    print(f"\nCD Diagram — {metric}")
    print("-" * 50)

    fig, ax = plt.subplots(figsize=(14, 5))

    try:
        sp.critical_difference_diagram(
            avg_ranks,
            sig_matrix=nemenyi,
            label_fmt_left="{label} ({rank:.2f})",
            label_fmt_right="({rank:.2f}) {label}",
            ax=ax,
        )
    except Exception as e:
        print(f"CD diagram plotting failed: {e}")
        plt.close()
        return

    ax.set_title(title, fontsize=13, pad=15)
    plt.tight_layout()

    safe_metric = metric.lower().replace(" ", "_")
    save_path   = RESULTS_DIR / f"cd_diagram_{safe_metric}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"CD diagram saved to: {save_path}")
    plt.show()
    plt.close()


# Full synthetic dataset and model analysis pipeline
def run_friedman_analysis(df: pd.DataFrame, metric: str, diagram_title: str) -> None:
    """
    Run the complete Friedman statistical anaysis pipeline for 1 metric: 
    1. descritptive statistics per drift type and model include ARIMA
    2. Build piviot table
    3. Friedman test on all models except ARIMA 
    4. Nemenyi post-hoc test and CD diagram (only if friedman is significant)
    """
    print(f"\nFriedman Analysis for {metric}")
    
    # 1. Descriptive statistics
    print_descriptive_statistics(df, metric)
    
    # 2. Build pivot table
    pivot = build_pivot(df, metric)
    
    if pivot.empty:
        print("Pivot table is empty. Cannot run statistical test.")
        return
    
    if pivot.shape[1] < 3:
        print(f"Not enough models with data to compare. Need at least 3, got {pivot.shape[1]}.")
        return
    
    # 3. Friedman test (exclude ARIMA)
    stat, p_value, avg_ranks = run_friedman_test(pivot, metric)
    
    # 4. Nemenyi post-hoc test and CD diagram (only if friedman is significant)
    if p_value >= ALPHA:
        print(f"Friedman test not significant (p={p_value:.4f}). No further tests.")
        return 
    
    nemenyi = run_nemenyi_test(pivot, metric)
    plot_cd_diagram(avg_ranks, nemenyi, metric, diagram_title)
        
        


# Full real dataset and model analysis pipeline 
def run_wilcoxon_analysis(df: pd.DataFrame, metric: str) -> None:
    pass


def main():
    print("####################################################################")
    print("Statistical Test Analysis for RQ1 and RQ2 Combined")
    print("####################################################################\n")
    
    # Load and normalise results 
    df = load_synthetic_results()
    
    # Analysis 1: Return MAE
    run_friedman_analysis(df, METRIC_MAE, "Critical Difference Diagram - Return MAE")
    
    # analysis 2: Total Retrain Time
    run_friedman_analysis(df, METRIC_TIME, diagram_title = "Critical Difference Diagram - Total Retrain Time (seconds)")
    
    print("\n#####################################################################")
    print("  ANALYSIS COMPLETE")
    print("#####################################################################\n")
    print(f"CD diagrams saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
    
    

