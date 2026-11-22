
import itertools
import time

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from config import FEATURES_CSV, FIG_DIR, MODEL_DIR, RESULTS_DIR, SEED
from utils import feature_columns, evaluate

GRID = {
    "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
    "alpha": [1e-4, 1e-3, 1e-2],            # L2 penalty
    "learning_rate_init": [1e-3, 3e-3],
}


def make(params):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", MLPClassifier(max_iter=400, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=15, batch_size=256, random_state=SEED, **params))])


def main():
    feat = pd.read_csv(FEATURES_CSV)
    tr, va, te = (feat[feat["split"] == s] for s in ("train", "val", "test"))

    for fs in ["all", "visible"]:
        cols = feature_columns(feat, fs)
        print(f"\n===== MLP, feature set: {fs} =====")
        tuning = []
        for values in itertools.product(*GRID.values()):
            params = dict(zip(GRID.keys(), values))
            t0 = time.time()
            m = make(params).fit(tr[cols], tr["y"])
            f1 = f1_score(va["y"], m.predict(va[cols]), average="macro")
            tuning.append({"_params": params, **{k: str(v) for k, v in params.items()}, "val_macro_f1": round(f1, 4),
                           "epochs": m.named_steps["clf"].n_iter_, "time_s": round(time.time() - t0, 1)})
            print(f"  {params} -> val macro-F1 {f1:.4f}")
        tuning = pd.DataFrame(tuning).sort_values("val_macro_f1", ascending=False)
        tuning.drop(columns="_params").to_csv(RESULTS_DIR / f"mlp_tuning__{fs}.csv", index=False)

        best_row = tuning.iloc[0]
        best_params = best_row["_params"]
        best = make(best_params).fit(tr[cols], tr["y"])
        tag = f"mlp__{fs}"
        evaluate(te["y"].values, best.predict(te[cols]), tag, ids=te["id"].values,
                 extra={"feature_set": fs, "best_params": {k: str(v) for k, v in best_params.items()},
                        "val_macro_f1": float(best_row["val_macro_f1"])})
        joblib.dump({"model": best, "columns": cols}, MODEL_DIR / f"{tag}.joblib")

        clf = best.named_steps["clf"]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(clf.loss_curve_, label="training loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        ax2.plot(clf.validation_scores_, c="tab:orange", label="internal val accuracy")
        ax2.set_ylabel("accuracy")
        fig.legend(loc="center right")
        plt.title(f"MLP learning curve ({fs} features) - {best_params['hidden_layer_sizes']}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"mlp_curve__{fs}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
