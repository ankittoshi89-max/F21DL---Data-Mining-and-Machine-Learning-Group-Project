
import json
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest

from config import RESULTS_DIR, PRED_DIR, FIG_DIR, CLASSES


def load_results():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        r = json.load(open(f))
        if "macro_f1" in r:          # skip clustering / vae summary files
            rows.append(r)
    return rows


def mcnemar(a, b):
    """Exact McNemar test on two models' test predictions (same samples)."""
    pa, pb = pd.read_csv(PRED_DIR / f"{a}.csv"), pd.read_csv(PRED_DIR / f"{b}.csv")
    m = pa.merge(pb, on="id", suffixes=("_a", "_b"))
    ca = m.y_pred_a == m.y_true_a
    cb = m.y_pred_b == m.y_true_b
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"model_a": a, "model_b": b, "a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": p}


def main():
    rows = load_results()
    if not rows:
        raise SystemExit("No results yet - run the model scripts first")
    df = pd.DataFrame([{"model": r["model"], "input": r.get("feature_set", "?"), "accuracy": r["accuracy"],
                        "macro_precision": r["macro_precision"], "macro_recall": r["macro_recall"],
                        "macro_f1": r["macro_f1"]} for r in rows]).sort_values("macro_f1", ascending=False)
    df.to_csv(RESULTS_DIR / "results_table.csv", index=False)
    try:
        md = df.to_markdown(index=False, floatfmt=".4f")   # needs the 'tabulate' package
    except ImportError:
        md = df.to_string(index=False)
    (RESULTS_DIR / "results_table.md").write_text(md)
    print(df.to_string(index=False))

    palette = {"all": "tab:blue", "visible": "tab:orange", "rgb": "tab:green", "ms": "tab:purple"}
    plt.figure(figsize=(10, max(4, 0.32 * len(df))))
    sns.barplot(data=df, y="model", x="macro_f1", hue="input", dodge=False, palette=palette)
    plt.xlim(max(0, df.macro_f1.min() - 0.1), 1)
    plt.title("Test macro-F1 of every model")
    plt.legend(title="input", loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "compare_macro_f1.png", dpi=150)
    plt.close()

    pc = pd.DataFrame({r["model"]: r["per_class_f1"] for r in rows}).T.loc[df.model, CLASSES]
    plt.figure(figsize=(12, max(4, 0.4 * len(pc))))
    sns.heatmap(pc, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.4, vmax=1, cbar=False)
    plt.title("Per-class F1 on the test set")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "compare_per_class_f1.png", dpi=150)
    plt.close()

    # H1: same model, all bands vs visible only
    tab = df[df.input.isin(["all", "visible"])].copy()
    if len(tab):
        tab["algo"] = tab.model.str.split("__").str[0]
        piv = tab.pivot_table(index="algo", columns="input", values="macro_f1")
        piv["gain_from_extra_bands"] = piv.get("all", np.nan) - piv.get("visible", np.nan)
        piv.to_csv(RESULTS_DIR / "h1_all_vs_visible.csv")
        piv[["visible", "all"]].plot(kind="bar", figsize=(9, 4.5), color=["tab:orange", "tab:blue"])
        plt.title("H1: visible-band features vs all 13 bands (test macro-F1)")
        plt.ylabel("macro-F1"); plt.xticks(rotation=30, ha="right"); plt.ylim(0, 1)
        plt.tight_layout(); plt.savefig(FIG_DIR / "h1_all_vs_visible.png", dpi=150); plt.close()
        print("\nH1 table:\n", piv.round(4))

    # significance between the top models (+ best RGB vs best MS model if both exist)
    have = {p.stem for p in PRED_DIR.glob("*.csv")}
    top = [m for m in df.model if m in have][:4]
    pairs = list(combinations(top, 2))
    best_by_input = df[df.model.isin(have)].groupby("input").first()["model"]
    if {"rgb", "all"} <= set(best_by_input.index):
        pairs.append((best_by_input["rgb"], best_by_input["all"]))
    tests = pd.DataFrame([mcnemar(a, b) for a, b in dict.fromkeys(pairs)])
    if len(tests):
        tests.to_csv(RESULTS_DIR / "mcnemar_tests.csv", index=False)
        print("\nMcNemar tests:\n", tests.to_string(index=False))


if __name__ == "__main__":
    main()
