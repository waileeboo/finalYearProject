from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_DIR = BACKEND_DIR.parent

RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"
SYNTHETIC_DATA_DIR = BACKEND_DIR / "data" / "synthetic"
RESULTS_DIR = BACKEND_DIR / "data" / "results"
RESULTS_FILE = RESULTS_DIR / "experiment_results.csv"
RESULTS_FILE_RQ2_PHASE1 = RESULTS_DIR / "rq2_phase1_results.csv"
RESULTS_FILE_RQ2_PHASE2 = RESULTS_DIR / "rq2_phase2_results.csv"