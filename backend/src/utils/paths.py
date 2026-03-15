from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_DIR = BACKEND_DIR.parent

RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"
SYNTHETIC_DATA_DIR = BACKEND_DIR / "data" / "synthetic"
RESULTS_DIR = BACKEND_DIR / "data" / "results"
RESULTS_FILE_RQ1 = RESULTS_DIR / "rq1_results.csv"
RESULTS_FILE_RQ2_PHASE1 = RESULTS_DIR / "rq2_phase1_results.csv"
RESULTS_FILE_RQ2_PHASE2_SYNTHETIC = RESULTS_DIR / "rq2_phase2_synthetic_results.csv"
RESULTS_FILE_RQ2_PHASE2_REAL = RESULTS_DIR / "rq2_phase2_real_results.csv"
RESULTS_FILE_RQ3_SYNTHETIC = RESULTS_DIR / "rq3_synthetic_results.csv"
RESULTS_FILE_RQ3_REAL = RESULTS_DIR / "rq3_real_results.csv"
TEST2 = RESULTS_DIR / "test2.csv"
RESULTS_FILE_RQ4_REAL = RESULTS_DIR / "rq4_real_results.csv"
RESULTS_FILE_RQ4_SYNTHETIC = RESULTS_DIR / "rq4_synthetic_results.csv"