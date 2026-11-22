
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from config import INDEX_CSV, FIG_DIR, MODEL_DIR, RESULTS_DIR, CLASSES, SEED
from utils import set_seed, evaluate
from torch_common import get_device, EuroSATDataset, ConvVAE


def vae_loss(recon, x, mu, logvar, beta):
    rec = F.mse_loss(recon, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
    return rec + beta * kl, rec, kl


def grid(imgs, title, path, ncol=8, labels=None):
    n = len(imgs)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.4, nrow * 1.5))
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if k < n:
            ax.imshow(np.clip(imgs[k].transpose(1, 2, 0), 0, 1))
            if labels is not None:
                ax.set_title(labels[k], fontsize=6)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@torch.no_grad()
def encode_all(model, loader, device):
    model.eval()
    mus, ys, errs = [], [], []
    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encode(x)
        recon = model.decode(mu)
        errs.append(((recon - x) ** 2).mean(dim=(1, 2, 3)).cpu())
        mus.append(mu.cpu()); ys.append(y)
    return torch.cat(mus).numpy(), torch.cat(ys).numpy(), torch.cat(errs).numpy()


def main(a):
    set_seed(SEED)
    device = get_device()
    index = pd.read_csv(INDEX_CSV)
    tr, te = index[index["split"] == "train"], index[index["split"] == "test"]

    # VAE works on [0, 1] pixels, so mean 0 / std 1 = no extra normalisation
    ds_tr = EuroSATDataset(tr, "rgb", [0, 0, 0], [1, 1, 1], train=True)
    ds_tr_plain = EuroSATDataset(tr, "rgb", [0, 0, 0], [1, 1, 1])
    ds_te = EuroSATDataset(te, "rgb", [0, 0, 0], [1, 1, 1])
    dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=256)

    model = ConvVAE(a.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    hist = []
    for epoch in range(1, a.epochs + 1):
        model.train()
        beta = a.beta * min(1.0, epoch / max(1, a.warmup))   # KL warm-up avoids posterior collapse
        tot = rec_t = kl_t = 0
        for x, _ in dl_tr:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar, beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x); rec_t += rec.item() * len(x); kl_t += kl.item() * len(x)
        n = len(ds_tr)
        hist.append({"epoch": epoch, "loss": tot / n, "recon": rec_t / n, "kl": kl_t / n, "beta": beta})
        print(f"epoch {epoch:2d}  loss {tot / n:.1f}  recon {rec_t / n:.1f}  KL {kl_t / n:.1f}")

    torch.save({"state_dict": model.state_dict(), "latent": a.latent}, MODEL_DIR / "vae_rgb.pt")
    hist = pd.DataFrame(hist)
    hist.to_csv(RESULTS_DIR / "history_vae.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(hist.epoch, hist.recon); ax[0].set_title("Reconstruction loss (MSE sum / image)")
    ax[1].plot(hist.epoch, hist.kl, c="tab:red"); ax[1].set_title("KL divergence")
    for x_ in ax:
        x_.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(FIG_DIR / "vae_loss.png", dpi=150); plt.close()

    model.eval()
    with torch.no_grad():
        # reconstructions: 2 test images per class, original row above reconstruction row
        picks = te.groupby("label").sample(2, random_state=SEED).index
        pos = [te.index.get_loc(i) for i in picks]
        x = torch.stack([ds_te[i][0] for i in pos]).to(device)
        recon = model.decode(model.encode(x)[0]).cpu().numpy()
        both = np.concatenate([x.cpu().numpy(), recon])
        labels = [te.loc[i, "label"][:10] for i in picks] + ["recon"] * len(picks)
        grid(both, "Top: real test patches  |  Bottom: VAE reconstructions", FIG_DIR / "vae_reconstructions.png",
             ncol=len(picks), labels=labels)

        z = torch.randn(32, a.latent, device=device)
        grid(model.decode(z).cpu().numpy(), "New patches sampled from the VAE prior N(0, I)", FIG_DIR / "vae_samples.png")

        a_img = ds_te[te.index.get_loc(te[te.label == "Forest"].index[0])][0]
        b_img = ds_te[te.index.get_loc(te[te.label == "SeaLake"].index[0])][0]
        za, zb = model.encode(torch.stack([a_img, b_img]).to(device))[0]
        steps = torch.stack([za + t * (zb - za) for t in torch.linspace(0, 1, 8, device=device)])
        grid(model.decode(steps).cpu().numpy(), "Latent interpolation: Forest -> SeaLake", FIG_DIR / "vae_interpolation.png")

    # latent codes -> per-class error, PCA plot and a simple classifier
    z_tr, y_tr, _ = encode_all(model, DataLoader(ds_tr_plain, batch_size=256), device)
    z_te, y_te, err_te = encode_all(model, dl_te, device)
    per_class = pd.Series(err_te).groupby(pd.Series(y_te).map(dict(enumerate(CLASSES)))).mean().reindex(CLASSES)
    per_class.to_csv(RESULTS_DIR / "vae_recon_error_per_class.csv", header=["mse"])
    plt.figure(figsize=(8, 4)); per_class.sort_values().plot(kind="barh")
    plt.title("VAE test reconstruction error per class (MSE)"); plt.tight_layout()
    plt.savefig(FIG_DIR / "vae_error_per_class.png", dpi=150); plt.close()

    P = PCA(2, random_state=SEED).fit_transform(z_te)
    plt.figure(figsize=(7, 5.5))
    sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=[CLASSES[i] for i in y_te], hue_order=CLASSES, s=5, linewidth=0)
    plt.legend(fontsize=7, markerscale=3); plt.title("VAE latent space (test set, PCA)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "vae_latent_pca.png", dpi=150); plt.close()

    clf = LogisticRegression(max_iter=3000).fit(z_tr, y_tr)
    evaluate(y_te, clf.predict(z_te), "vae_latent_logreg__rgb", ids=te["id"].values,
             extra={"feature_set": "rgb", "note": "logistic regression on unsupervised VAE latent codes"})

    summary = {"test_recon_mse_mean": float(err_te.mean()), "latent_dim": a.latent, "beta": a.beta,
               "epochs": a.epochs, "final_train_recon": float(hist.recon.iloc[-1]), "final_train_kl": float(hist.kl.iloc[-1])}
    with open(RESULTS_DIR / "vae_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    main(ap.parse_args())
