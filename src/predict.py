
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from config import MODEL_DIR, OUT_DIR, CLASSES
from utils import ms_patch_to_features

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
TIF_EXT = {".tif", ".tiff"}
PATCH = 64
CLASS_COLOURS = plt.cm.tab10(np.arange(len(CLASSES)))


# ----------------------------------------------------------------------------- loading
def read_any(path, gsd=10.0):
    """Returns (H, W, C) float32 array and a flag saying whether it is 'rgb' or 'ms'."""
    path = Path(path)
    if path.suffix.lower() in TIF_EXT:
        import tifffile
        arr = tifffile.imread(str(path))
        if arr.ndim == 3 and arr.shape[0] in (3, 4, 13) and arr.shape[-1] not in (3, 4, 13):
            arr = np.transpose(arr, (1, 2, 0))
        arr = arr.astype(np.float32)
        kind = "ms" if arr.shape[-1] == 13 else "rgb"
        if kind == "rgb":
            arr = arr[..., :3]
            if arr.max() > 255:                       # 16-bit RGB -> stretch to 0..255
                arr = np.clip(arr / np.percentile(arr, 99.5) * 255, 0, 255)
    else:
        arr = np.array(Image.open(path).convert("RGB")).astype(np.float32)
        kind = "rgb"

    if abs(gsd - 10.0) > 1e-6:                        # resample to EuroSAT's 10 m/pixel
        scale = gsd / 10.0
        h, w = arr.shape[:2]
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        arr = np.stack([np.array(Image.fromarray(arr[..., c]).resize(new, Image.BILINEAR))
                        for c in range(arr.shape[-1])], axis=-1)
    return arr, kind


def to_tiles(arr):
    """Cut into a grid of 64x64 tiles. Images smaller than a tile are resized up to 64x64."""
    h, w = arr.shape[:2]
    if h < PATCH or w < PATCH or (h < 1.5 * PATCH and w < 1.5 * PATCH):
        img = np.stack([np.array(Image.fromarray(arr[..., c]).resize((PATCH, PATCH), Image.BILINEAR))
                        for c in range(arr.shape[-1])], axis=-1)
        return img[None], (1, 1)
    rows, cols = h // PATCH, w // PATCH
    tiles = [arr[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] for r in range(rows) for c in range(cols)]
    return np.stack(tiles), (rows, cols)


# ----------------------------------------------------------------------------- models
class CNNPredictor:
    def __init__(self, ckpt_path):
        import torch
        from torch_common import build_model, get_device
        self.torch = torch
        self.device = get_device()
        ck = torch.load(ckpt_path, map_location=self.device)
        self.bands = ck["bands"]
        self.model = build_model(ck["arch"], ck["in_ch"], len(CLASSES), pretrained=False).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.mean = torch.tensor(ck["mean"]).view(1, -1, 1, 1)
        self.std = torch.tensor(ck["std"]).view(1, -1, 1, 1)
        self.name = Path(ckpt_path).stem

    def predict_proba(self, tiles):
        t = self.torch
        x = t.from_numpy(tiles.astype(np.float32)).permute(0, 3, 1, 2)
        if self.bands == "rgb":
            x = x / 255.0
        x = (x - self.mean) / self.std
        out = []
        with t.no_grad():
            for i in range(0, len(x), 256):
                out.append(t.softmax(self.model(x[i:i + 256].to(self.device)), 1).cpu())
        return t.cat(out).numpy()


class TabularPredictor:
    def __init__(self, path):
        bundle = joblib.load(path)
        self.model, self.columns = bundle["model"], bundle["columns"]
        self.name = Path(path).stem

    def predict_proba(self, tiles):
        X = pd.DataFrame([ms_patch_to_features(t) for t in tiles])[self.columns]
        return self.model.predict_proba(X)


def load_predictors(kind, args):
    preds = []
    if kind == "rgb":
        p = Path(args.rgb_cnn)
        if p.exists():
            preds.append(CNNPredictor(p))
    else:
        p = Path(args.ms_cnn)
        if p.exists():
            preds.append(CNNPredictor(p))
        p = Path(args.tabular)
        if p.exists():
            preds.append(TabularPredictor(p))
    if not preds:
        raise SystemExit(f"No trained model found for {kind} input - train it first (see README)")
    return preds


# ----------------------------------------------------------------------------- output
def save_map(arr, kind, labels, grid_shape, out_path, title):
    rows, cols = grid_shape
    if kind == "ms":
        from config import BAND
        show = arr[..., [BAND["B04"], BAND["B03"], BAND["B02"]]]
        show = np.clip(show / np.percentile(show, 99), 0, 1)
    else:
        show = np.clip(arr / 255.0, 0, 1)
    show = show[:rows * PATCH, :cols * PATCH]
    overlay = CLASS_COLOURS[labels.reshape(rows, cols)][..., :3]
    overlay = np.kron(overlay, np.ones((PATCH, PATCH, 1)))

    fig, ax = plt.subplots(1, 2, figsize=(14, 7 * rows / max(cols, 1) + 1))
    ax[0].imshow(show); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(show); ax[1].imshow(overlay, alpha=0.5); ax[1].set_title(title); ax[1].axis("off")
    present = sorted(set(labels.tolist()))
    ax[1].legend(handles=[Patch(color=CLASS_COLOURS[i], label=CLASSES[i]) for i in present],
                 bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def predict_file(path, args, cache):
    arr, kind = read_any(path, args.gsd)
    tiles, grid_shape = to_tiles(arr)
    if kind not in cache:
        cache[kind] = load_predictors(kind, args)
    rows = []
    probs_all = []
    for predictor in cache[kind]:
        probs = predictor.predict_proba(tiles)
        probs_all.append(probs)
        mean_probs = probs.mean(0)
        top3 = np.argsort(mean_probs)[::-1][:3]
        print(f"\n{Path(path).name}  [{predictor.name}]  {len(tiles)} tile(s)")
        if len(tiles) == 1:
            for k in top3:
                print(f"   {CLASSES[k]:22s} {mean_probs[k] * 100:5.1f}%")
        else:
            shares = pd.Series(probs.argmax(1)).map(dict(enumerate(CLASSES))).value_counts(normalize=True)
            for c, s in shares.items():
                print(f"   {c:22s} {s * 100:5.1f}% of the area")
            out = OUT_DIR / f"live_map_{Path(path).stem}_{predictor.name}.png"
            save_map(arr, kind, probs.argmax(1), grid_shape, out, f"Predicted land cover ({predictor.name})")
            print(f"   map saved to {out}")
        rows.append({"file": str(path), "model": predictor.name, "tiles": len(tiles),
                     "prediction": CLASSES[top3[0]], "confidence": round(float(mean_probs[top3[0]]), 4),
                     "second": CLASSES[top3[1]], "third": CLASSES[top3[2]]})
        if mean_probs[top3[0]] < 0.5 and len(tiles) == 1:
            print("   (low confidence - the image may be unlike EuroSAT, check resolution / colours)")
    if len(probs_all) > 1:   # simple ensemble when we have both MS models
        ens = np.mean([p.mean(0) for p in probs_all], axis=0)
        print(f"   ensemble -> {CLASSES[int(ens.argmax())]} ({ens.max() * 100:.1f}%)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--gsd", type=float, default=10.0, help="metres per pixel of the input (EuroSAT = 10)")
    ap.add_argument("--rgb-cnn", default=str(MODEL_DIR / "cnn_simple__rgb.pt"))
    ap.add_argument("--ms-cnn", default=str(MODEL_DIR / "cnn_simple__ms.pt"))
    ap.add_argument("--tabular", default=str(MODEL_DIR / "random_forest__all.joblib"))
    args = ap.parse_args()

    files = []
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT | TIF_EXT)
        elif p.exists():
            files.append(p)
        else:
            print(f"skipping {p} (not found)")
    if not files:
        raise SystemExit("No images to predict")

    cache, all_rows = {}, []
    for f in files:
        try:
            all_rows += predict_file(f, args, cache)
        except Exception as e:
            print(f"could not process {f}: {e}")
    if all_rows:
        out = OUT_DIR / "live_predictions.csv"
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\nAll predictions saved to {out}")


if __name__ == "__main__":
    main()
