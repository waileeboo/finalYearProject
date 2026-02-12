from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_DIR = BACKEND_DIR.parent

RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"
SYNTHETIC_DATA_DIR = BACKEND_DIR / "data" / "synthetic"