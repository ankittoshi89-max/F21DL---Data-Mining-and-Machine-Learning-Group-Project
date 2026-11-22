
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RAW_DIR, INDEX_CSV, CLASSES, CLASS_TO_IDX, SEED, TEST_SIZE, VAL_SIZE, DATA_DIR
from utils import find_class_root, load_rgb, load_ms


def scan(root, ext):
    rows = []
    for cls in CLASSES:
        for f in sorted((root / cls).glob(f"*{ext}")):
            rows.append({"id": f.stem, "label": cls, "path": str(f)})
    return pd.DataFrame(rows)


def main():
    rgb_root = find_class_root(RAW_DIR / "EuroSAT_RGB")
    rgb = scan(rgb_root, ".jpg").rename(columns={"path": "rgb_path"})
    print(f"RGB images found: {len(rgb)}")

    try:
        ms_root = find_class_root(RAW_DIR / "EuroSAT_MS")
        ms = scan(ms_root, ".tif").rename(columns={"path": "ms_path"})
        print(f"MS images found:  {len(ms)}")
        df = rgb.merge(ms, on=["id", "label"], how="outer", indicator=True)
        unpaired = df[df["_merge"] != "both"]
        if len(unpaired):
            print(f"WARNING: {len(unpaired)} patches exist in only one version -> dropped")
            unpaired.to_csv(DATA_DIR / "unpaired_files.csv", index=False)
        df = df[df["_merge"] == "both"].drop(columns="_merge")
    except FileNotFoundError:
        print("MS data not found, building an RGB-only index for now")
        df = rgb.copy()
        df["ms_path"] = None

    # --- data quality checks -------------------------------------------------
    bad, hashes = [], []
    for row in df.itertuples():
        try:
            img = load_rgb(row.rgb_path)
            if img.shape != (64, 64, 3):
                bad.append((row.id, f"rgb shape {img.shape}"))
            hashes.append(hashlib.md5(img.tobytes()).hexdigest())
        except Exception as e:
            bad.append((row.id, f"rgb unreadable: {e}"))
            hashes.append(None)
    df["rgb_hash"] = hashes

    # Opening all 27k tifs is slow, so we fully check a random sample of 1000.
    if df["ms_path"].notna().all():
        sample = df.sample(min(1000, len(df)), random_state=SEED)
        for row in sample.itertuples():
            try:
                m = load_ms(row.ms_path)
                if m.shape != (64, 64, 13):
                    bad.append((row.id, f"ms shape {m.shape}"))
                if not np.isfinite(m).all():
                    bad.append((row.id, "ms has NaN/inf"))
            except Exception as e:
                bad.append((row.id, f"ms unreadable: {e}"))

    dupes = df[df["rgb_hash"].notna() & df.duplicated("rgb_hash", keep="first")]
    print(f"Unreadable / wrong-shape files: {len(bad)}")
    print(f"Exact duplicate RGB images:    {len(dupes)}")
    bad_ids = {b[0] for b in bad} | set(dupes["id"])
    if bad:
        pd.DataFrame(bad, columns=["id", "problem"]).to_csv(DATA_DIR / "bad_files.csv", index=False)
    df = df[~df["id"].isin(bad_ids)].drop(columns="rgb_hash").reset_index(drop=True)

    # --- stratified split ----------------------------------------------------
    df["y"] = df["label"].map(CLASS_TO_IDX)
    train_val, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["y"], random_state=SEED)
    val_share = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(train_val, test_size=val_share, stratify=train_val["y"], random_state=SEED)
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    df.to_csv(INDEX_CSV, index=False)
    print(f"\nSaved {INDEX_CSV}  ({len(df)} clean patches)")
    print(pd.crosstab(df["label"], df["split"], margins=True))


if __name__ == "__main__":
    main()
