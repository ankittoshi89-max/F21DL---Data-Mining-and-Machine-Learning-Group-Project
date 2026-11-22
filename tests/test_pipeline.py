
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import BAND_NAMES, CLASSES, INDEX_CSV   # noqa: E402
from utils import ms_patch_to_features, safe_index, feature_columns   # noqa: E402


def test_feature_count_and_names():
    feats = ms_patch_to_features(np.random.rand(64, 64, 13).astype(np.float32) * 3000)
    assert len(feats) == 13 * 2 + 6
    assert all(f"{b}_mean" in feats for b in BAND_NAMES)


def test_indices_are_bounded():
    a, b = np.random.rand(64, 64) * 5000, np.random.rand(64, 64) * 5000
    idx = safe_index(a, b)
    assert np.all(idx >= -1) and np.all(idx <= 1)


def test_ndvi_high_for_vegetation():
    ms = np.full((64, 64, 13), 500, np.float32)
    ms[:, :, 7] = 4000          # strong NIR (B08)
    ms[:, :, 3] = 400           # low red (B04)
    assert ms_patch_to_features(ms)["NDVI_mean"] > 0.7


def test_visible_feature_set_only_rgb_bands():
    df = pd.DataFrame([ms_patch_to_features(np.random.rand(64, 64, 13))])
    cols = feature_columns(df, "visible")
    assert sorted({c.split("_")[0] for c in cols}) == ["B02", "B03", "B04"]
    assert len(feature_columns(df, "all")) == 32


def test_split_has_no_overlap():
    if not INDEX_CSV.exists():
        return
    df = pd.read_csv(INDEX_CSV)
    assert df["id"].is_unique
    assert set(df["split"]) == {"train", "val", "test"}
    assert set(df["label"]) <= set(CLASSES)
