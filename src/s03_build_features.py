
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from config import INDEX_CSV, FEATURES_CSV
from utils import load_ms, ms_patch_to_features


def one_row(row):
    try:
        feats = ms_patch_to_features(load_ms(row["ms_path"]))
    except OSError as e:
        print(f"  could not read {row['id']} ({e.__class__.__name__}) - skipped")
        return None
    feats.update({"id": row["id"], "label": row["label"], "y": row["y"], "split": row["split"]})
    return feats


def main(workers):
    index = pd.read_csv(INDEX_CSV)
    if index["ms_path"].isna().any():
        raise SystemExit("MS paths missing - run s01 (without --rgb-only) and s02 first")

    records = index.to_dict("records")
    total = len(records)
    print(f"Extracting features from {total} multispectral patches using {workers} thread(s) ...")

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, feats in enumerate(pool.map(one_row, records), start=1):
            if feats is not None:
                rows.append(feats)
            if i % 1000 == 0 or i == total:
                elapsed = time.time() - t0
                left = elapsed / i * (total - i)
                print(f"  {i:5d}/{total}  ({elapsed / 60:.1f} min so far, ~{left / 60:.1f} min left)")

    df = pd.DataFrame(rows)
    front = ["id", "label", "y", "split"]
    df = df[front + [c for c in df.columns if c not in front]]

    # cleaning checks on the derived table
    feats = df.drop(columns=front)
    n_bad = (~np.isfinite(feats.values)).sum()
    constant = feats.columns[feats.std() == 0].tolist()
    print(f"Non-finite values: {n_bad}   Constant columns: {constant}")
    if n_bad:
        df = df[np.isfinite(feats.values).all(axis=1)]
    if constant:
        df = df.drop(columns=constant)

    df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {FEATURES_CSV}  shape={df.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    main(ap.parse_args().workers)