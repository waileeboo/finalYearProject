"""
Removes all model "NAME" rows from rq2_phase2_synthetic_results.csv
so that the specified model can be cleanly rerun without duplicate entries.
"""

import pandas as pd
from src.utils.paths import RESULTS_FILE_RQ2_PHASE2_SYNTHETIC as RESULTS_FILE

MODEL_TO_REMOVE = "ELM_kswin"


def main(model_name: str = MODEL_TO_REMOVE):
    if not RESULTS_FILE.exists():
        print(f"File not found: {RESULTS_FILE}")
        return

    df = pd.read_csv(RESULTS_FILE)
    print(f"Rows before: {len(df)}")
    print(f"Models present: {sorted(df['model'].unique().tolist())}")

    removed = df[df["model"] == model_name]
    print(f"\nRows to remove ({model_name}): {len(removed)}")

    df_clean = df[df["model"] != model_name].reset_index(drop=True)
    print(f"Rows after:  {len(df_clean)}")

    df_clean.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved cleaned file to: {RESULTS_FILE}")
    print(f"Models remaining: {sorted(df_clean['model'].unique().tolist())}")


if __name__ == "__main__":
    main(model_name=MODEL_TO_REMOVE)