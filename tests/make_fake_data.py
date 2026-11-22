
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import RAW_DIR, CLASSES   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-class", type=int, default=40)
n = ap.parse_args().per_class
rng = np.random.default_rng(0)

for k, cls in enumerate(CLASSES):
    rgb_dir = RAW_DIR / "EuroSAT_RGB" / "EuroSAT_RGB" / cls
    ms_dir = RAW_DIR / "EuroSAT_MS" / "EuroSAT_MS" / cls
    rgb_dir.mkdir(parents=True, exist_ok=True)
    ms_dir.mkdir(parents=True, exist_ok=True)
    base_rgb = rng.integers(40, 200, 3)
    base_ms = rng.integers(500, 3000, 13)
    for i in range(1, n + 1):
        img = np.clip(base_rgb + rng.normal(0, 25, (64, 64, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(rgb_dir / f"{cls}_{i}.jpg")
        ms = np.clip(base_ms + rng.normal(0, 300, (64, 64, 13)), 0, 10000).astype(np.uint16)
        tifffile.imwrite(ms_dir / f"{cls}_{i}.tif", ms)
print("fake data written to", RAW_DIR)
