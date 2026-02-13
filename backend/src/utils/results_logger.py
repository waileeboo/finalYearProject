import pandas as pd
from datetime import datetime
from src.utils.paths import RESULTS_DIR, RESULTS_FILE


def log_results(model_name: str, dataset: str, metrics: dict, notes: str = "") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "dataset": dataset,
        "notes": notes,
        **metrics,
    }

    row_df = pd.DataFrame([row])

    if RESULTS_FILE.exists():
        existing = pd.read_csv(RESULTS_FILE)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")