"""
Project settings in one place, so every script uses the same paths, seed and class list.
If you move the data somewhere else, this is the only file you need to change.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # zip files + unzipped folders live here
INDEX_CSV = DATA_DIR / "index.csv"    # one row per patch: id, label, paths, split
FEATURES_CSV = DATA_DIR / "eurosat_ms_features.csv"   # our derived tabular dataset

OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"      # metrics as json
PRED_DIR = OUT_DIR / "predictions"     # per-sample test predictions (for McNemar etc.)

for d in (RAW_DIR, FIG_DIR, MODEL_DIR, RESULTS_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Official Zenodo record 7711810 (EuroSAT, Helber et al. 2019)
RGB_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
MS_URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Band order inside the EuroSAT_MS .tif files (note B8A is stored LAST, not after B08)
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B09", "B10", "B11", "B12", "B8A"]
BAND = {b: i for i, b in enumerate(BAND_NAMES)}

# The three visible bands. We use these to build an "RGB-only" version of the tabular data,
# so we can test H1 (does the extra spectral info help?) on exactly the same samples.
VISIBLE_BANDS = ["B02", "B03", "B04"]   # blue, green, red

# 70 / 15 / 15 stratified split, as proposed in the pitch
TEST_SIZE = 0.15
VAL_SIZE = 0.15
