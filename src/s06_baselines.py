
import argparse
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from config import FEATURES_CSV, FIG_DIR, MODEL_DIR, RESULTS_DIR, CLASSES, SEED
from utils import feature_columns, evaluate


MODELS = {
    "decision_tree": (DecisionTreeClassifier(random_state=SEED),
                      {"clf__max_depth": [5, 10, 15, 20, None], "clf__min_samples_leaf": [1, 5, 20],
                       "clf__criterion": ["gini", "entropy"]}),
    "naive_bayes": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}),
    "logistic_regression": (LogisticRegression(max_iter=3000),
                            {"clf__C": [0.01, 0.1, 1, 10, 100]}),
    "knn": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 5, 9, 15], "clf__weights": ["uniform", "distance"]}),
    "perceptron": (Perceptron(random_state=SEED, max_iter=2000),
                   {"clf__alpha": [1e-5, 1e-4, 1e-3], "clf__penalty": [None, "l2"]}),
    # forests get big in memory, so the grid is kept small and leaves are not allowed to be tiny
    "random_forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1, min_samples_leaf=2),
                      {"clf__n_estimators": [150, 300], "clf__max_depth": [20, 30],
                       "clf__max_features": ["sqrt"]}),
}


def main(only=None):
    feat = pd.read_csv(FEATURES_CSV)
    trval = feat[feat["split"].isin(["train", "val"])].reset_index(drop=True)
    test = feat[feat["split"] == "test"].reset_index(drop=True)
    fold = np.where(trval["split"] == "val", 0, -1)   # -1 = always train, 0 = validation fold
    ps = PredefinedSplit(fold)

    summary = []
    for fs in ["all", "visible"]:
        cols = feature_columns(feat, fs)
        print(f"\n===== feature set: {fs} ({len(cols)} features) =====")
        for name, (clf, grid) in MODELS.items():
            if only and name not in only:
                continue
            pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
            # one model at a time (n_jobs=1): copying models into many processes ran out of RAM
            # on a laptop. Random forest still uses all cores inside itself.
            search = GridSearchCV(pipe, grid, cv=ps, scoring="f1_macro", n_jobs=1, refit=False)
            t0 = time.time()
            search.fit(trval[cols], trval["y"])

            # refit the best settings on TRAIN only, so val stays a true validation set
            best = pipe.set_params(**search.best_params_)
            tr = trval[trval["split"] == "train"]
            best.fit(tr[cols], tr["y"])
            fit_time = time.time() - t0

            tag = f"{name}__{fs}"
            res = evaluate(test["y"].values, best.predict(test[cols]), tag, ids=test["id"].values,
                           extra={"feature_set": fs, "n_features": len(cols),
                                  "best_params": {k.replace("clf__", ""): str(v) for k, v in search.best_params_.items()},
                                  "val_macro_f1": round(search.best_score_, 4),
                                  "tuning_time_s": round(fit_time, 1)})
            summary.append(res)
            joblib.dump({"model": best, "columns": cols}, MODEL_DIR / f"{tag}.joblib")

            if name == "decision_tree" and fs == "all":
                explain_tree(best, cols)
            if name == "random_forest" and fs == "all":
                importance(best, cols)

    # build the summary from every saved result, so re-running only some models (--only)
    # still gives the full table
    import json
    keep = ["model", "feature_set", "accuracy", "macro_f1", "macro_precision", "macro_recall", "val_macro_f1"]
    algos = tuple(MODELS)
    rows = []
    for f in RESULTS_DIR.glob("*.json"):
        r = json.load(open(f))
        if str(r.get("model", "")).startswith(algos) and "macro_f1" in r:
            rows.append({k: r.get(k) for k in keep})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "baselines_summary.csv", index=False)
    print("\n", table.sort_values("macro_f1", ascending=False).to_string(index=False))


def explain_tree(pipe, cols):
    """Decision trees are readable, so we plot the top of the tree and save the rules
    as text - this is what we discuss as the 'practical use' of trees in the report."""
    tree = pipe.named_steps["clf"]
    plt.figure(figsize=(22, 9))
    plot_tree(tree, max_depth=3, feature_names=cols, class_names=CLASSES, filled=True, fontsize=7, impurity=False)
    plt.title("Decision tree - top 3 levels (features are standardised)")
    plt.savefig(FIG_DIR / "decision_tree_top.png", dpi=150, bbox_inches="tight")
    plt.close()
    with open(RESULTS_DIR / "decision_tree_rules.txt", "w") as f:
        f.write(export_text(tree, feature_names=cols, max_depth=4))
    print(f"   tree depth={tree.get_depth()}  leaves={tree.get_n_leaves()}")


def importance(pipe, cols):
    rf = pipe.named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    imp.to_csv(RESULTS_DIR / "rf_feature_importance.csv")
    top = imp.head(15)[::-1]
    colours = ["tab:orange" if c.split("_")[0] in ("B02", "B03", "B04") else "tab:blue" for c in top.index]
    plt.figure(figsize=(8, 5.5))
    plt.barh(top.index, top.values, color=colours)
    plt.title("Random forest feature importance (orange = visible band)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rf_feature_importance.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run only some models, e.g. --only decision_tree naive_bayes")
    main(ap.parse_args().only)