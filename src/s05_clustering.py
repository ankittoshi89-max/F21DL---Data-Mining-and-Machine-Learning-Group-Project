
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score, homogeneity_score, completeness_score)

from config import FEATURES_CSV, FIG_DIR, RESULTS_DIR, CLASSES, SEED
from utils import feature_columns


def cluster_scores(X, labels_true, labels_pred, sil_sample=5000):
    return {
        "silhouette": round(float(silhouette_score(X, labels_pred, sample_size=min(sil_sample, len(X)), random_state=SEED)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X, labels_pred)), 4),
        "ARI": round(float(adjusted_rand_score(labels_true, labels_pred)), 4),
        "NMI": round(float(normalized_mutual_info_score(labels_true, labels_pred)), 4),
        "homogeneity": round(float(homogeneity_score(labels_true, labels_pred)), 4),
        "completeness": round(float(completeness_score(labels_true, labels_pred)), 4),
    }


def main():
    feat = pd.read_csv(FEATURES_CSV)
    train = feat[feat["split"] == "train"].reset_index(drop=True)
    X = StandardScaler().fit_transform(train[feature_columns(feat)])
    y = train["y"].values

    # ---- choose k -----------------------------------------------------------
    ks = list(range(2, 16))
    inertia, sil, bic = [], [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_, sample_size=min(5000, len(X)), random_state=SEED))
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED, max_iter=300).fit(X)
        bic.append(gmm.bic(X))
        print(f"k={k:2d}  inertia={inertia[-1]:.0f}  silhouette={sil[-1]:.3f}  GMM BIC={bic[-1]:.0f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(ks, inertia, marker="o"); axes[0].set_title("K-means elbow (inertia)")
    axes[1].plot(ks, sil, marker="o", c="tab:green"); axes[1].set_title("K-means silhouette")
    axes[2].plot(ks, bic, marker="o", c="tab:red"); axes[2].set_title("GMM (EM) BIC - lower is better")
    for ax in axes:
        ax.axvline(10, ls="--", c="grey"); ax.set_xlabel("k")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_choose_k.png", dpi=150)
    plt.close()

    # ---- compare to the true classes at k = 10 ------------------------------
    km = KMeans(n_clusters=10, n_init=20, random_state=SEED).fit(X)
    gmm = GaussianMixture(n_components=10, covariance_type="full", random_state=SEED, max_iter=300).fit(X)
    results = {
        "kmeans_k10": cluster_scores(X, y, km.labels_),
        "gmm_em_k10": cluster_scores(X, y, gmm.predict(X)),
        "best_k_by_silhouette": int(ks[int(np.argmax(sil))]),
        "best_k_by_bic": int(ks[int(np.argmin(bic))]),
    }
    print(json.dumps(results, indent=2))
    with open(RESULTS_DIR / "clustering.json", "w") as f:
        json.dump(results, f, indent=2)

    for name, labels in [("kmeans", km.labels_), ("gmm", gmm.predict(X))]:
        ct = pd.crosstab(train["label"], labels, normalize="index").reindex(CLASSES)
        plt.figure(figsize=(9, 5.5))
        sns.heatmap(ct, annot=True, fmt=".2f", cmap="viridis", cbar=False)
        plt.title(f"{name.upper()} (k=10): share of each class in each cluster")
        plt.xlabel("cluster"); plt.ylabel("true class")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"cluster_vs_class_{name}.png", dpi=150)
        plt.close()

    Z = PCA(n_components=2, random_state=SEED).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=train["label"], hue_order=CLASSES, s=3, linewidth=0, ax=axes[0])
    axes[0].set_title("True classes (PCA view)")
    axes[0].legend(fontsize=7, markerscale=3)
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=km.labels_, palette="tab10", s=3, linewidth=0, ax=axes[1], legend=False)
    axes[1].set_title("K-means clusters (k=10, PCA view)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_pca.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
