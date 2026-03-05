import pandas as pd
from datetime import datetime
from src.utils.paths import RESULTS_DIR, RESULTS_FILE_RQ1
from pathlib import Path

def log_results(model_name: str, dataset: str, metrics: dict, notes: str = "",path: str = RESULTS_FILE_RQ1) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "dataset": dataset,
        "notes": notes,
        **metrics,
    }

    row_df = pd.DataFrame([row])

    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(path, index=False)
    print(f"Results saved to {path}")