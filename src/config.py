from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Epileptic Seizure Recognition.csv"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TARGET_CLASSES = (2, 3)   # 2 = tumour area, 3 = healthy area (Kaggle labels)
N_FOLDS = 5

HOLDOUT_FRAC = 0.2