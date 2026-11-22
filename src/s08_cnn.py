
import argparse
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from config import INDEX_CSV, FIG_DIR, MODEL_DIR, RESULTS_DIR, CLASSES, SEED, BAND
from utils import set_seed, evaluate
from torch_common import get_device, rgb_stats, ms_stats, EuroSATDataset, build_model


def run_epoch(model, loader, device, opt=None, loss_fn=None, scaler=None):
    train = opt is not None
    model.train(train)
    total_loss, preds, trues = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
            total_loss += loss.item() * len(y)
            preds.append(out.argmax(1).cpu()); trues.append(y.cpu())
    preds, trues = torch.cat(preds).numpy(), torch.cat(trues).numpy()
    return total_loss / len(trues), (preds == trues).mean(), f1_score(trues, preds, average="macro"), preds


def show_mistakes(ds, y_true, y_pred, bands, name, n=16):
    wrong = np.where(y_true != y_pred)[0][:n]
    if len(wrong) == 0:
        return
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.2))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        x = ds.load(i).numpy()
        if bands == "ms":   # show the MS patch as natural colour (B04, B03, B02)
            x = x[[BAND["B04"], BAND["B03"], BAND["B02"]]]
            x = np.clip(x / np.percentile(x, 99), 0, 1)
        ax.imshow(np.transpose(x, (1, 2, 0)))
        ax.set_title(f"T:{CLASSES[y_true[i]][:9]}\nP:{CLASSES[y_pred[i]][:9]}", fontsize=7)
    plt.suptitle(f"Misclassified test patches - {name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"mistakes_{name}.png", dpi=150)
    plt.close()


def main(a):
    set_seed(SEED)
    device = get_device()
    print("Using device:", device)

    index = pd.read_csv(INDEX_CSV)
    tr, va, te = (index[index["split"] == s] for s in ("train", "val", "test"))
    if a.bands == "rgb":
        mean, std = rgb_stats(tr)
        in_ch = 3
    else:
        mean, std = ms_stats(tr["id"])
        in_ch = 13

    cache = a.bands == "rgb" or a.cache_ms
    ds_tr = EuroSATDataset(tr, a.bands, mean, std, train=True, cache=cache)
    ds_va = EuroSATDataset(va, a.bands, mean, std, cache=cache)
    ds_te = EuroSATDataset(te, a.bands, mean, std, cache=cache)
    kw = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=device.type == "cuda")
    dl_tr = DataLoader(ds_tr, shuffle=True, **kw)
    dl_va = DataLoader(ds_va, **kw)
    dl_te = DataLoader(ds_te, **kw)

    model = build_model(a.arch, in_ch, len(CLASSES), pretrained=not a.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model {a.arch}: {n_params / 1e6:.2f}M parameters")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    name = f"cnn_{a.arch}__{a.bands}" + (f"__{a.tag}" if a.tag else "")
    ckpt_path = MODEL_DIR / f"{name}.pt"
    history, best_f1, bad_epochs = [], -1, 0
    t_start = time.time()
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        tl, ta, tf, _ = run_epoch(model, dl_tr, device, opt, loss_fn, scaler)
        vl, vacc, vf, _ = run_epoch(model, dl_va, device, loss_fn=loss_fn)
        sched.step()
        history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta, "val_loss": vl,
                        "val_acc": vacc, "val_macro_f1": vf, "lr": opt.param_groups[0]["lr"]})
        print(f"epoch {epoch:2d}  train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {vacc:.3f} "
              f"F1 {vf:.3f}  ({time.time() - t0:.0f}s)")
        if vf > best_f1:
            best_f1, bad_epochs = vf, 0
            torch.save({"state_dict": model.state_dict(), "arch": a.arch, "bands": a.bands,
                        "in_ch": in_ch, "mean": mean, "std": std, "classes": CLASSES}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                print(f"No improvement for {a.patience} epochs - stopping early")
                break

    hist = pd.DataFrame(history)
    hist.to_csv(RESULTS_DIR / f"history_{name}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.epoch, hist.train_loss, label="train"); axes[0].plot(hist.epoch, hist.val_loss, label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(hist.epoch, hist.train_acc, label="train acc"); axes[1].plot(hist.epoch, hist.val_acc, label="val acc")
    axes[1].plot(hist.epoch, hist.val_macro_f1, label="val macro-F1", ls="--")
    axes[1].set_title("Accuracy / F1"); axes[1].legend()
    for ax in axes:
        ax.set_xlabel("epoch")
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"curves_{name}.png", dpi=150)
    plt.close()

    # final test with the best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    _, _, _, preds = run_epoch(model, dl_te, device, loss_fn=loss_fn)
    y_true = te["y"].values
    evaluate(y_true, preds, name, ids=te["id"].values,
             extra={"feature_set": a.bands, "arch": a.arch, "epochs_run": len(hist), "best_val_macro_f1": round(best_f1, 4),
                    "params_millions": round(n_params / 1e6, 2), "lr": a.lr, "batch_size": a.batch_size,
                    "weight_decay": a.weight_decay, "train_time_min": round((time.time() - t_start) / 60, 1)})
    show_mistakes(ds_te, y_true, preds, a.bands, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", choices=["rgb", "ms"], default="rgb")
    ap.add_argument("--arch", choices=["simple", "resnet18"], default="simple")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    ap.add_argument("--cache-ms", action="store_true", help="load all MS patches into RAM (~3 GB)")
    ap.add_argument("--tag", default="", help="extra name for fine-tuning runs")
    main(ap.parse_args())
