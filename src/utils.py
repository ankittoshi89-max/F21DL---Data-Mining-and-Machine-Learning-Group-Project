
import json
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # so plots save fine on machines without a display
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

from config import CLASSES, FIG_DIR, RESULTS_DIR, PRED_DIR, BAND


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_class_root(base_dir):
    """The zips sometimes have an extra folder level, so we search for the folder
    that actually contains the 10 class sub-folders instead of hard-coding it."""
    base_dir = __import__("pathlib").Path(base_dir)
    candidates = [base_dir] + [p for p in base_dir.rglob("*") if p.is_dir()]
    for c in candidates:
        if all((c / cls).is_dir() for cls in CLASSES):
            return c
    raise FileNotFoundError(f"Could not find the 10 class folders under {base_dir}")


def load_rgb(path):
    """Returns a (64, 64, 3) uint8 array."""
    return np.array(Image.open(path).convert("RGB"))


def load_ms(path, tries=5):
    """Returns a (64, 64, 13) float32 array of raw Sentinel-2 values.
    We read the bytes ourselves and retry a few times, because on Windows (especially inside
    OneDrive) opening thousands of files quickly can fail with 'insufficient system resources'."""
    import io
    import time
    import tifffile
    for attempt in range(tries):
        try:
            with open(path, "rb") as f:
                data = f.read()
            arr = tifffile.imread(io.BytesIO(data))
            break
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
    if arr.ndim == 3 and arr.shape[0] == 13:      # some readers give bands first
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def safe_index(a, b, eps=1e-6):
    """Normalised difference (a - b) / (a + b) without dividing by zero."""
    return (a - b) / (a + b + eps)


def ms_patch_to_features(ms):
    """Turns one 13-band patch into one row of our tabular dataset.
    26 band statistics (mean + std per band) + mean/std of NDVI, NDWI, NDBI = 32 features."""
    feats = {}
    for name, i in BAND.items():
        band = ms[:, :, i]
        feats[f"{name}_mean"] = float(band.mean())
        feats[f"{name}_std"] = float(band.std())
    nir, red, green, swir = (ms[:, :, BAND["B08"]], ms[:, :, BAND["B04"]],
                             ms[:, :, BAND["B03"]], ms[:, :, BAND["B11"]])
    indices = {
        "NDVI": safe_index(nir, red),      # vegetation
        "NDWI": safe_index(green, nir),    # open water (McFeeters)
        "NDBI": safe_index(swir, nir),     # built-up areas
    }
    for name, img in indices.items():
        feats[f"{name}_mean"] = float(img.mean())
        feats[f"{name}_std"] = float(img.std())
    return feats


def feature_columns(df, which="all"):
    """'all' = every engineered feature, 'visible' = only blue/green/red stats (RGB baseline)."""
    from config import VISIBLE_BANDS
    cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    if which == "visible":
        cols = [c for c in cols if c.split("_")[0] in VISIBLE_BANDS]
    return cols


def evaluate(y_true, y_pred, name, extra=None, ids=None):
    """Works out the metrics we report everywhere, saves them to json,
    saves the confusion matrix plot and the per-sample predictions."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    per_class = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))),
                                      target_names=CLASSES, output_dict=True, zero_division=0)
    result = {
        "model": name,
        "accuracy": round(acc, 4),
        "macro_precision": round(p, 4),
        "macro_recall": round(r, 4),
        "macro_f1": round(f1, 4),
        "weighted_f1": round(f1_w, 4),
        "per_class_f1": {c: round(per_class[c]["f1-score"], 4) for c in CLASSES},
        "per_class_recall": {c: round(per_class[c]["recall"], 4) for c in CLASSES},
    }
    if extra:
        result.update(extra)

    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)

    plot_confusion(y_true, y_pred, f"Confusion matrix - {name}", FIG_DIR / f"cm_{name}.png")

    if ids is not None:
        pd.DataFrame({"id": ids, "y_true": y_true, "y_pred": y_pred}).to_csv(
            PRED_DIR / f"{name}.csv", index=False)

    print(f"[{name}] acc={acc:.4f}  macro-F1={f1:.4f}  macro-P={p:.4f}  macro-R={r:.4f}")
    return result


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)   # row-normalised = recall per class
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()