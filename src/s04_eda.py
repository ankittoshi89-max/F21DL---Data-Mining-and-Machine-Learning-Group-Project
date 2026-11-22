
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif, f_classif

from config import INDEX_CSV, FEATURES_CSV, FIG_DIR, RESULTS_DIR, CLASSES, BAND_NAMES, SEED
from utils import load_rgb, feature_columns

sns.set_theme(style="whitegrid")


def class_balance(index):
    ct = pd.crosstab(index["label"], index["split"])[["train", "val", "test"]]
    ct.plot(kind="bar", stacked=True, figsize=(9, 4.5))
    plt.title("Patches per class and split")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_class_balance.png", dpi=150)
    plt.close()
    ct.to_csv(RESULTS_DIR / "class_counts.csv")
    print(ct)


def sample_grid(index, n=6):
    fig, axes = plt.subplots(len(CLASSES), n, figsize=(n * 1.3, len(CLASSES) * 1.35))
    for i, cls in enumerate(CLASSES):
        picks = index[index["label"] == cls].sample(n, random_state=SEED)
        for j, p in enumerate(picks["rgb_path"]):
            axes[i, j].imshow(load_rgb(p))
            axes[i, j].axis("off")
        axes[i, 0].text(-8, 32, cls, ha="right", va="center", fontsize=8)
    plt.suptitle("Random EuroSAT RGB patches per class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close()


def rgb_histograms(index, per_class=150):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    for ax, cls in zip(axes.ravel(), CLASSES):
        paths = index[(index["label"] == cls) & (index["split"] == "train")]["rgb_path"]
        imgs = np.stack([load_rgb(p) for p in paths.sample(min(per_class, len(paths)), random_state=SEED)])
        for c, colour in enumerate(["red", "green", "blue"]):
            ax.hist(imgs[..., c].ravel(), bins=64, range=(0, 255), color=colour, alpha=0.45, density=True)
        ax.set_title(cls, fontsize=9)
    plt.suptitle("RGB pixel intensity distribution per class (train sample)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_rgb_histograms.png", dpi=150)
    plt.close()


def spectral_signature(feat):
    train = feat[feat["split"] == "train"]
    mean_cols = [f"{b}_mean" for b in BAND_NAMES if f"{b}_mean" in feat.columns]
    sig = train.groupby("label")[mean_cols].mean()
    # put the bands in wavelength order for the plot (B8A sits between B08 and B09)
    order = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    order = [f"{b}_mean" for b in order if f"{b}_mean" in sig.columns]
    plt.figure(figsize=(10, 5))
    for cls in CLASSES:
        plt.plot([o.split("_")[0] for o in order], sig.loc[cls, order], marker="o", label=cls)
    for b in ["B02", "B03", "B04"]:
        plt.axvspan(order.index(f"{b}_mean") - 0.4, order.index(f"{b}_mean") + 0.4, color="grey", alpha=0.12)
    plt.title("Mean spectral signature per class (shaded = visible RGB bands)")
    plt.ylabel("mean reflectance (raw DN)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_spectral_signature.png", dpi=150)
    plt.close()
    sig.to_csv(RESULTS_DIR / "spectral_signature.csv")


def index_boxplots(feat):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, idx in zip(axes, ["NDVI_mean", "NDWI_mean", "NDBI_mean"]):
        sns.boxplot(data=feat, x="label", y=idx, order=CLASSES, ax=ax, fliersize=1)
        ax.set_title(idx.replace("_mean", ""))
        ax.tick_params(axis="x", rotation=60)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_spectral_indices.png", dpi=150)
    plt.close()


def correlation(feat):
    cols = feature_columns(feat)
    corr = feat[cols].corr()
    plt.figure(figsize=(13, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, xticklabels=True, yticklabels=True)
    plt.title("Feature correlation (Pearson)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_correlation.png", dpi=150)
    plt.close()
    # list very correlated pairs - useful to talk about redundancy / Naive Bayes assumption
    pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
             .rename("r").reset_index().query("abs(r) > 0.95").sort_values("r", ascending=False))
    pairs.to_csv(RESULTS_DIR / "highly_correlated_pairs.csv", index=False)
    print(f"Feature pairs with |r| > 0.95: {len(pairs)}")


def projections(feat, n_tsne=4000):
    train = feat[feat["split"] == "train"]
    X = StandardScaler().fit_transform(train[feature_columns(feat)])
    pca = PCA(random_state=SEED).fit(X)
    Z = pca.transform(X)[:, :2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(np.arange(1, len(pca.explained_variance_ratio_) + 1),
                 np.cumsum(pca.explained_variance_ratio_), marker=".")
    axes[0].axhline(0.95, ls="--", c="grey")
    axes[0].set_title("PCA cumulative explained variance")
    axes[0].set_xlabel("components")

    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=train["label"], hue_order=CLASSES, s=4, linewidth=0, ax=axes[1], legend=False)
    axes[1].set_title("PCA (first 2 components)")

    sub = train.sample(min(n_tsne, len(train)), random_state=SEED)
    Xs = StandardScaler().fit_transform(sub[feature_columns(feat)])
    T = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(Xs)
    sns.scatterplot(x=T[:, 0], y=T[:, 1], hue=sub["label"], hue_order=CLASSES, s=5, linewidth=0, ax=axes[2])
    axes[2].set_title(f"t-SNE ({len(sub)} training patches)")
    axes[2].legend(fontsize=7, markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_pca_tsne.png", dpi=150)
    plt.close()
    n95 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
    print(f"Components needed for 95% variance: {n95}")


def feature_ranking(feat):
    train = feat[feat["split"] == "train"]
    cols = feature_columns(feat)
    mi = mutual_info_classif(train[cols], train["y"], random_state=SEED)
    f, _ = f_classif(train[cols], train["y"])
    rank = pd.DataFrame({"feature": cols, "mutual_info": mi, "anova_F": f}).sort_values("mutual_info", ascending=False)
    rank.to_csv(RESULTS_DIR / "feature_ranking.csv", index=False)

    top = rank.head(20)
    colours = ["tab:orange" if t.split("_")[0] in ("B02", "B03", "B04") else "tab:blue" for t in top["feature"]]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["mutual_info"][::-1], color=colours[::-1])
    plt.title("Top 20 features by mutual information (orange = visible band)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_feature_ranking.png", dpi=150)
    plt.close()
    print("Top 10 features:\n", rank.head(10).to_string(index=False))


def main():
    index = pd.read_csv(INDEX_CSV)
    class_balance(index)
    sample_grid(index)
    rgb_histograms(index)
    if FEATURES_CSV.exists():
        feat = pd.read_csv(FEATURES_CSV)
        print("\nDescriptive stats saved to outputs/results/feature_describe.csv")
        feat[feature_columns(feat)].describe().T.to_csv(RESULTS_DIR / "feature_describe.csv")
        print(f"Missing values in features: {feat.isna().sum().sum()}")
        spectral_signature(feat)
        index_boxplots(feat)
        correlation(feat)
        projections(feat)
        feature_ranking(feat)
    else:
        print("No feature CSV yet - run s03 to get the spectral plots")
    print("EDA figures saved in", FIG_DIR)


if __name__ == "__main__":
    main()
