
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config import FEATURES_CSV, BAND_NAMES
from utils import load_rgb, load_ms


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")      # Apple silicon
    return torch.device("cpu")


# ----------------------------------------------------------------------------- normalisation
def rgb_stats(train_df, n=3000, seed=42):
    """Per-channel mean/std from TRAINING images only (no test leakage)."""
    paths = train_df["rgb_path"].sample(min(n, len(train_df)), random_state=seed)
    imgs = np.stack([load_rgb(p) for p in paths]).astype(np.float32) / 255.0
    return imgs.mean(axis=(0, 1, 2)).tolist(), imgs.std(axis=(0, 1, 2)).tolist()


def ms_stats(train_ids):
    """Per-band pixel mean/std worked out from our feature CSV, so we don't need another pass
    over 2 GB of tifs. Every patch has the same size, so:
        pixel mean = average of patch means
        pixel var  = average(std^2 + mean^2) - pixel mean^2"""
    feat = pd.read_csv(FEATURES_CSV)
    feat = feat[feat["id"].isin(set(train_ids))]
    means, stds = [], []
    for b in BAND_NAMES:
        m, s = feat[f"{b}_mean"].values, feat[f"{b}_std"].values
        mu = m.mean()
        var = (s ** 2 + m ** 2).mean() - mu ** 2
        means.append(float(mu))
        stds.append(float(np.sqrt(max(var, 1e-6))))
    return means, stds


# ----------------------------------------------------------------------------- data
def augment(x):
    """Satellite tiles have no 'up', so flips and 90-degree turns are safe augmentations.
    Works for 3 or 13 channels. x: (C, H, W) tensor."""
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[2])
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[1])
    k = int(torch.randint(0, 4, (1,)))
    return torch.rot90(x, k, dims=[1, 2])


class EuroSATDataset(Dataset):
    """bands='rgb' reads the jpgs, bands='ms' reads the 13-band tifs.
    RGB is small (~330 MB for all patches) so we keep it in memory for speed."""

    def __init__(self, df, bands, mean, std, train=False, cache=True):
        self.df = df.reset_index(drop=True)
        self.bands = bands
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.train = train
        self.cache = None
        if cache and bands == "rgb":
            self.cache = np.stack([load_rgb(p) for p in self.df["rgb_path"]])
        elif cache and bands == "ms":
            # ~3 GB for the whole dataset in float16 - only use if you have the RAM
            self.cache = np.stack([load_ms(p).astype(np.float16) for p in self.df["ms_path"]])

    def __len__(self):
        return len(self.df)

    def load(self, i):
        if self.cache is not None:
            arr = self.cache[i]
        elif self.bands == "rgb":
            arr = load_rgb(self.df.loc[i, "rgb_path"])
        else:
            arr = load_ms(self.df.loc[i, "ms_path"])
        x = torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32)).permute(2, 0, 1)
        if self.bands == "rgb":
            x = x / 255.0
        return x

    def __getitem__(self, i):
        x = (self.load(i) - self.mean) / self.std
        if self.train:
            x = augment(x)
        return x, int(self.df.loc[i, "y"])


# ----------------------------------------------------------------------------- models
def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Our own CNN: 4 conv blocks (64->32->16->8->4), global average pooling, dropout, linear."""

    def __init__(self, in_ch=3, n_classes=10, width=32, dropout=0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(conv_block(in_ch, w), conv_block(w, 2 * w),
                                      conv_block(2 * w, 4 * w), conv_block(4 * w, 8 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(8 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18(in_ch=3, n_classes=10, pretrained=True):
    """Transfer-learning option. For 13 bands we rebuild the first conv layer and start it
    from the average of the pretrained RGB filters."""
    import torchvision
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    if in_ch != 3:
        old = m.conv1
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            m.conv1.weight[:] = old.weight.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def build_model(arch, in_ch, n_classes=10, pretrained=True, **kw):
    if arch == "simple":
        return SimpleCNN(in_ch, n_classes, **kw)
    if arch == "resnet18":
        return resnet18(in_ch, n_classes, pretrained)
    raise ValueError(arch)


class ConvVAE(nn.Module):
    """Variational autoencoder for 64x64 RGB patches (generative part of the project)."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),     # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),    # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2), # 4x4
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 256, 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparameterisation trick
        return self.decode(z), mu, logvar
