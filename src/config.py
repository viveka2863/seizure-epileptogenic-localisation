from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Epileptic Seizure Recognition.csv"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
SPLITS = ROOT / "splits"

RANDOM_STATE = 42
TARGET_CLASSES = (2, 3)   # 2 = tumour area, 3 = healthy area (Kaggle labels)
N_FOLDS = 10             # Outer folds (segments per fold: 16)
N_INNER_FOLDS = 5         # Inner folds for tuning
N_REPEATS = 5             # Repeats when comparing models

HOLDOUT_FRAC = 0.2
SAMPLING_RATE_HZ = 4097 / 23.6  # about 173.6 Hz, implied by Kaggle's description (4097 points in 23.6 s)

SIMILARITY_STRONG = 0.7   # any two segments more similar than this are always kept together
SIMILARITY_WEAK = 0.4     # segments in a tight cluster (every pair above this) are kept together too