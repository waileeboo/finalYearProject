import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations  
import scikit_posthocs as sp 

from src.utils.paths import RESULTS_FILE

# Configuraiton for statistical tests
ALPHA = 0.05
SYNTHETIC_METRIC = "Return_MAE"
REAL_METRIC = "MAE"


def extract_drift_type(dataset: str) -> str:
    """
    Extract drift type from dataset name
    """
    if dataset == "GSPC":
        return "GSPC"
    parts = dataset.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return dataset


def print_descriptive_table(df: pd.DataFrame, metric: str) -> None:
    summary = df.groupby(["drift_type", "model"])[metric].agg(
        ["count", "mean", "std", "median"]
    )
    q1 = df.groupby(["drift_type", "model"])[metric].quantile(0.25)
    q3 = df.groupby(["drift_type", "model"])[metric].quantile(0.75)
    summary["Q1"] = q1
    summary["Q3"] = q3
    summary["IQR"] = q3 - q1
    print(summary.round(6).to_string())


# Friedman Test + Post-hoc Nemenyi Test
def run_friedman_nemenyi(pivot: pd.DataFrame, group_name: str, metric: str, save_plot: bool = True) -> None:
    if pivot.shape[1] < 3:
        print(f"\nNeed at least 3 models for Friedman test, got {pivot.shape[1]}. Skipping.")
        return
    if pivot.shape[0] < 3:
        print(f"\nNeed at least 3 datasets for Friedman test, got {pivot.shape[0]}. Skipping.")
        return
    
    print(f"Datasets: {pivot.shape[0]} | Models: {list(pivot.columns)}")
    
    # Average Ranks (rank 1 = lowerst MAE = Best) rank across rows (Datasets) and then average rank for each model
    ranks = pivot.rank(axis=1)
    avg_ranks = ranks.mean().sort_values()
    print("\nAverage Ranks (lower is better):")
    for model, rank in avg_ranks.items():
        print(f"  {model}: {rank:.3f}")
    # give the function n list of array where n is the number of models and each array is the metric values 
    # \* the break the list to individual array for each model
    try:
        stat, p_value = friedmanchisquare(*[pivot[col].values for col in pivot.columns])
    except ValueError as e:
        print(f"Friedman test failed: {e}")
    # chi suqre represent how much the average ranks of your models deviate from waht we would expect, A larger number mean a greater difference between models. 
    print(f"\nFriedman Test: chi_square = {stat:.4f}, p = {p_value:.6f} {significance_stars(p_value)}")

    if p_value >= ALPHA:
        print("No significant difference among models.")
        return

    print("Significant difference detected. Running Nemenyi post-hoc...\n")
    nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
    nemenyi.index = pivot.columns
    nemenyi.columns = pivot.columns

    print("Nemenyi p-values:")
    print(nemenyi.round(4).to_string(index=True))
    
    if save_plot:
        try:
            fig, ax = plt.subplots(figsize=(10, 3))
            sp.critical_difference_diagram(
                avg_ranks,
                sig_matrix=nemenyi,
                label_fmt_left="{label} ({rank:.2f})",
                label_fmt_right="({rank:.2f}) {label}",
                ax=ax,
            )
            safe_name = group_name.replace(" ", "_").lower()
            save_path = f"cd_diagram_{safe_name}_{metric.lower()}.png"
            ax.set_title(f"Critical Difference - {group_name} ({metric})", fontsize=12)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\nCD diagram saved to {save_path}")
            plt.show()
        except Exception as e:
            print(f"\nCD diagram failed: {e}")

def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"
    
# summary statistics (count, mean, std, medium, q1, q3) for each model and metric
def print_summary_statistics(df: pd.DataFrame, metric: str,) -> None:
    summary = df.groupby(["drift_type","model"])[metric].agg(["count", "mean", "std", "median"])
    q1 = df.groupby(["drift_type","model"])[metric].quantile(0.25)
    q3 = df.groupby(["drift_type","model"])[metric].quantile(0.75)
    summary["Q1"] = q1
    summary["Q3"] = q3
    print(summary)
    
    


# Wilcoxon Signed-Rank Test for pairwise comparisons
def run_wilcoxon_real(df: pd.DataFrame, metric: str) -> None:
    summary = df.groupby("model")[metric].agg(["count", "mean", "std", "median"])
    print(summary.round(6).to_string())
    models = df["model"].unique()
    # number of comparision = n choose 2 where n is the number of models being compared 
    k = len(models)
    n_comparisons = (k * (k - 1)) // 2
    
    # Adjust alpha for multiple comparisons using Bonferroni correction to make it stricter 
    bonferroni_alpha = ALPHA / n_comparisons if n_comparisons > 0 else ALPHA

    print(f"\nPairwise Wilcoxon Signed-Rank Tests (Bonferroni α = {bonferroni_alpha:.4f}):")

    # take every unique pair 
    for m1, m2 in combinations(models, 2):
        # filter row for each model and extract metric values, dropna to remove any missing values
        vals1 = df[df["model"] == m1][metric].dropna().values
        vals2 = df[df["model"] == m2][metric].dropna().values
        
        # make sure we have at least 5 pairs lesser pair less reliable 
        n = min(len(vals1), len(vals2))
        if n < 5:
            print(f"  {m1} vs {m2}: insufficient data ({n} pairs)")
            continue
        
        # make it same length if not 
        vals1 = vals1[:n]
        vals2 = vals2[:n]

        try:
            stat, p_value = wilcoxon(vals1, vals2)
            sig = significance_stars(p_value)
            bonf_sig = "yes" if p_value < bonferroni_alpha else "no"
            better = m1 if np.median(vals1) < np.median(vals2) else m2

            print(f"  {m1} vs {m2}: p={p_value:.10f} {sig} "
                  f"| Bonferroni sig: {bonf_sig} | better: {better}")
        except ValueError as e:
            print(f"  {m1} vs {m2}: test failed ({e})")


def main():
    print("\n###################################################################")
    print("Statistical Tests for Model Performance Comparison")
    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded results from {RESULTS_FILE}\n")
    print(f"Models: {df['model'].unique().tolist()}\n")
    # there should be 30 runs for each model dataset combined (121 because arima only runs 1 time on real data)
    print(f"Datasets: {df['dataset'].nunique()} unique")

    # remove the _1, _2, _3 suffixes to get the drift type
    df["drift_type"] = df["dataset"].apply(extract_drift_type)
    # collect unique drift types
    drift_types = df["drift_type"].unique()
    print(f"Drift types: {drift_types.tolist()}")
    
    # Summary Statistics
    print("\nSummary Statistics:")
    for metric in [REAL_METRIC, SYNTHETIC_METRIC]:
        print(f"\nMetric: {metric}")
        print_summary_statistics(df, metric)
        print()
    
    
    # Real Data (GSPC) wilcoxon test (Exculde Arima becaseu it only has 1 run, no variance)
    real_df = df[df["drift_type"] == "GSPC"]
    if not real_df.empty and REAL_METRIC in real_df.columns:
        print(f"ANALYSIS: Real Data (GSPC) - REAL METRIC: {REAL_METRIC}")
        
        # REport ARIMA Seperately because it has only 1 run 
        arima_df = real_df[real_df["model"].str.contains("ARIMA", case =False)]
        if not arima_df.empty:
            arima_val = arima_df[REAL_METRIC].iloc[0]
            print(f"ARIMA (single run, excluded from Wilcoxon): {REAL_METRIC} = {arima_val:.4f}\n")
            
        
        # Wilconxon on models with 30 Runs (Excluding ARIMA)
        real_df_no_arima = real_df[~real_df["model"].str.contains("ARIMA", case=False)]
        print("Note: 1 dataset with 30 runs per model — using Wilcoxon pairwise.\n")
        run_wilcoxon_real(real_df_no_arima, REAL_METRIC)
        print()
        
    # Synthetic Data Friedman + Nemenyi
    synthetic_df = df[df["drift_type"] != "GSPC"]
    if not synthetic_df.empty and SYNTHETIC_METRIC in synthetic_df.columns:
        print(f"ANALYSIS: Synthetic Data - SYNTHETIC METRIC: {SYNTHETIC_METRIC}")
        
        summary = synthetic_df.groupby(["model","dataset"])[SYNTHETIC_METRIC].mean().reset_index()
        # create pivot table change row to column for model and value to metric, drop any missing values
        
        pivot_raw = summary.pivot(index="dataset", columns="model", values=SYNTHETIC_METRIC)
        print(f"Before dropna: {pivot_raw.shape}")
        print("\nMissing counts per model:")
        print(pivot_raw.isna().sum())
        
        pivot = summary.pivot(index="dataset", columns="model", values=SYNTHETIC_METRIC).dropna()
        print(f"Pivot table Shape: {pivot.shape} | Models in pivot: {pivot.columns.tolist()}\n")
        print(pivot.round(4).to_string())
        # pivot.shape[1] = number of models, i need at least 2 to compare
        if not pivot.empty and pivot.shape[1] >=2:
            run_friedman_nemenyi(pivot,"All Synthetic",  SYNTHETIC_METRIC)
        print()
        
    # # Descriptive Table (median +- IQR per drrift type )
    if not synthetic_df.empty and SYNTHETIC_METRIC in synthetic_df.columns:
        print_descriptive_table(synthetic_df, SYNTHETIC_METRIC)
        print()

    print("Analysis Complete.\n")
             
           
            
            
        
    print("DONE")
    print("###################################################################\n")
    

if __name__ == "__main__":
    main()