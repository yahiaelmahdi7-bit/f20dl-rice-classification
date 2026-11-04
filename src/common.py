"""Shared paths, seeds, feature sets, split gate, image cache, nets and helpers."""
import csv, hashlib, json, os, re, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SMOKE = os.environ.get("RICE_SMOKE") == "1"
_s = "smoke" if SMOKE else ""
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived" / _s
RES = ROOT / "results" / _s
FIG = ROOT / "figures" / _s
MODELS = Path(os.environ["RICE_MODELS_DIR"]) if os.environ.get("RICE_MODELS_DIR") else ROOT / "models" / _s
SPLITS = DERIVED / "splits.csv" if SMOKE else ROOT / "data" / "splits.csv"
FROZEN = ROOT / "results" / "frozen.json"
TEST_READS = ROOT / "results" / "test_reads.csv"
SEED = 0
SEEDS = (0, 1, 2)
CLASSES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
LATE_FROM = 11718          # Jasmine file number where the late lot starts
IMG = 128
READERS = {"07_compare.py", "08_routing.py", "09_robustness.py"}
PAIRS = [("hgb:all106", "cnn:image"), ("hgb:all106", "mlp:all106"), ("mlp:all106", "cnn:image")]

ALL106 = ("AREA PERIMETER MAJOR_AXIS MINOR_AXIS ECCENTRICITY EQDIASQ SOLIDITY CONVEX_AREA EXTENT ASPECT_RATIO "
          "ROUNDNESS COMPACTNESS SHAPEFACTOR_1 SHAPEFACTOR_2 SHAPEFACTOR_3 SHAPEFACTOR_4 meanRR meanRG meanRB "
          "StdDevRR StdDevRG StdDevRB skewRR skewRG skewRB kurtosisRR kurtosisRG kurtosisRB entropyRR entropyRG "
          "entropyRB meanH meanS meanV StdDevH StdDevS StdDevV skewH skewS skewV kurtosisH kurtosisS kurtosisV "
          "entropyH entropyS entropyV meanL meanA meanB StdDevL StdDevA StdDevB skewL skewA skewB kurtosisL "
          "kurtosisA kurtosisB entropyL entropyA entropyB meanY meanCb meanCr StdDevY StdDevCb StdDevCr skewY "
          "skewCb skewCr kurtosisY kurtosisCb kurtosisCr entropyY entropyCb entropyCr meanXX meanYY meanZZ "
          "StdDevXX StdDevYY StdDevZZ skewXX skewYY skewZZ kurtosisXX kurtosisYY kurtosisZZ entropyXX entropyYY "
          "entropyZZ ALLdaub4RR ALLdaub4RG ALLdaub4RB ALLdaub4H ALLdaub4S ALLdaub4V ALLdaub4L ALLdaub4a "
          "ALLdaub4b ALLdaub4Y ALLdaub4Cb ALLdaub4Cr ALLdaub4XX ALLdaub4YY ALLdaub4ZZ").split()
assert len(ALL106) == 106
_SF7 = "ECCENTRICITY SOLIDITY ASPECT_RATIO ROUNDNESS COMPACTNESS SHAPEFACTOR_3 SHAPEFACTOR_4".split()
_SF22 = _SF7 + ("meanRR meanRG meanRB meanH meanS meanV meanL meanA meanB meanY meanCb meanCr "
                "meanXX meanYY meanZZ").split()


def features(fs):
    """Column list for a feature-set name."""
    if fs == "all106": return list(ALL106)
    if fs == "distinct91": return [c for c in ALL106 if not c.startswith("ALLdaub4")]
    if fs == "shape16": return ALL106[:16]
    if fs == "colour90": return ALL106[16:]
    if fs == "sizefree7": return list(_SF7)
    if fs == "sizefree22": return list(_SF22)
    if fs == "selected": return json.loads((RES / "features" / "selected.json").read_text())["selected"]
    raise KeyError(fs)


def stage_dir(name):
    d = RES / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def savefig(fig, name):
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p, flush=True)


def wrote(p):
    print("wrote", p, flush=True)


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---- grains and the test gate -------------------------------------------------
def load_grains():
    """Train and validation grains only. Test rows are reachable only through load_split('test')."""
    g = pd.read_parquet(DERIVED / "grains.parquet")
    return g[g["split"] != "test"].reset_index(drop=True)


def _stack_files():
    f, out = sys._getframe(2), []
    while f is not None:
        out.append(Path(f.f_code.co_filename).name)
        f = f.f_back
    return out


def _gate():
    """Raise unless a permitted stage is on the call stack and the freeze file exists. Returns frozen hash."""
    names = _stack_files()
    ok = [n for n in names if n in READERS and (n != "08_routing.py" or "--score" in sys.argv)]
    if not ok:
        raise PermissionError("test split read refused: caller is not stage 7, 8 --score or 9")
    if not FROZEN.exists():
        raise PermissionError(f"test split read refused: {FROZEN} missing (run make freeze)")
    return ok[0], json.loads(FROZEN.read_text())["hash"]


def load_split(name):
    if name != "test":
        g = pd.read_parquet(DERIVED / "grains.parquet")
        return g[g["split"] == name].reset_index(drop=True)
    g = pd.read_parquet(DERIVED / "grains.parquet")
    if SMOKE:                      # validation stands in for test: no hash, no log
        return g[g["split"] == "val"].reset_index(drop=True)
    caller, h = _gate()
    TEST_READS.parent.mkdir(parents=True, exist_ok=True)
    new = not TEST_READS.exists()
    with open(TEST_READS, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["caller", "time", "frozen_hash"])
        w.writerow([caller, time.strftime("%Y-%m-%dT%H:%M:%S"), h])
    return g[g["split"] == "test"].reset_index(drop=True)


def xy(df, fs):
    return df[features(fs)].to_numpy(dtype=np.float64), df["y"].to_numpy()


# ---- image cache ----------------------------------------------------------------
def cache_path():
    return DERIVED / "images128.u8"


def open_cache():
    p = cache_path()
    n = p.stat().st_size // (IMG * IMG * 3)
    return np.memmap(p, dtype=np.uint8, mode="r", shape=(n, IMG, IMG, 3))


_mm = None


def read_images(df):
    """uint8 (n,128,128,3) for the rows of df, in df order. Test rows need a permitted caller."""
    global _mm
    if not SMOKE and (df["split"] == "test").any():
        _gate()
    if _mm is None: _mm = open_cache()
    idx = df["row"].to_numpy()
    o = np.argsort(idx, kind="stable")
    out = np.empty((len(idx), IMG, IMG, 3), np.uint8)
    out[o] = _mm[idx[o]]
    return out


def resize128(a):
    from PIL import Image
    return np.asarray(Image.fromarray(a).resize((IMG, IMG), Image.BILINEAR))


def grain_mask(a):
    """Largest Otsu-thresholded component of a uint8 RGB image (bool mask), or None."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    g = 0.2989 * a[..., 0] + 0.5870 * a[..., 1] + 0.1140 * a[..., 2]
    m = g > threshold_otsu(g)
    lab, n = ndi.label(m)
    if n == 0: return None
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


# ---- predictions ----------------------------------------------------------------
def pred_path(model, fs, seed, split):
    return RES / "preds" / f"{model}__{fs}__seed{seed}__{split}.parquet"


def save_preds(model, fs, seed, split, ids, y, P):
    p = pred_path(model, fs, seed, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame({"grain_id": np.asarray(ids), "y": np.asarray(y)})
    for k in range(5): d[f"p{k}"] = np.asarray(P)[:, k]
    d.to_parquet(p, index=False)
    wrote(p)


def load_preds(model, fs, seed, split):
    p = pred_path(model, fs, seed, split)
    return pd.read_parquet(p) if p.exists() else None


def probs(d):
    return d[[f"p{k}" for k in range(5)]].to_numpy()


# ---- statistics -----------------------------------------------------------------
def macro_f1(y, pred):
    from sklearn.metrics import f1_score
    return f1_score(y, pred, average="macro", labels=range(5))


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def ece(conf, correct, bins=15):
    e, n = 0.0, len(conf)
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return e


def temperature(P, y):
    """Temperature T minimising NLL of softmax(log P / T)."""
    from scipy.optimize import minimize_scalar
    L = np.log(np.clip(P, 1e-12, 1))
    def nll(t):
        z = L / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def apply_temp(P, t):
    z = np.log(np.clip(P, 1e-12, 1)) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def floor_for(conf, correct, target):
    """Lowest confidence t whose answers (conf >= t) reach the target accuracy; 1.01 if none."""
    o = np.argsort(conf)
    c, k = conf[o], correct[o].astype(float)
    tail = np.cumsum(k[::-1])[::-1] / np.arange(len(k), 0, -1)   # accuracy of everything from i upward
    ok = np.where(tail >= target)[0]
    return float(c[ok[0]]) if len(ok) else 1.01


# ---- generative models ------------------------------------------------------------
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


class ClassGMM(BaseEstimator, ClassifierMixin):
    """Per-class Gaussian mixture on PCA-k. Generative classifier and familiarity score."""
    def __init__(self, n_components=2, pca=20, seed=0):
        self.n_components, self.pca, self.seed = n_components, pca, seed

    def fit(self, X, y):
        self.pca_ = PCA(min(self.pca, X.shape[1]), random_state=self.seed).fit(X)
        Z = self.pca_.transform(X)
        self.classes_ = np.unique(y)
        self.gm_, lp = [], []
        for c in self.classes_:
            Zc = Z[y == c]
            k = int(max(1, min(self.n_components, len(Zc) // 20)))
            self.gm_.append(GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=self.seed).fit(Zc))
            lp.append(len(Zc))
        self.logprior_ = np.log(np.array(lp) / sum(lp))
        return self

    def loglik(self, X):
        Z = self.pca_.transform(X)
        return np.column_stack([g.score_samples(Z) for g in self.gm_])

    def predict_proba(self, X):
        L = self.loglik(X) + self.logprior_
        L = L - L.max(1, keepdims=True)
        e = np.exp(L)
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class Gate:
    """Familiarity check: best per-class log-likelihood under a PCA-20 mixture, cut at the 1st validation percentile."""
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, Xtr, ytr, Xval, pct=1.0):
        self.pre = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit(Xtr)
        self.gmm = ClassGMM(self.n_components).fit(self.pre.transform(Xtr), ytr)
        self.cut = float(np.percentile(self.score(Xval), pct))
        return self

    def score(self, X):
        return self.gmm.loglik(self.pre.transform(X)).max(1)

    def unfamiliar(self, X):
        return self.score(X) < self.cut


# ---- neural nets ------------------------------------------------------------------
def _nn():
    import torch.nn as nn
    return nn


def make_mlp(hidden, dropout, d_in=106):
    nn = _nn()
    L, c = [], d_in
    for h in hidden:
        L += [nn.Linear(c, h), nn.ReLU(), nn.Dropout(dropout)]
        c = h
    return nn.Sequential(*L, nn.Linear(c, 5))


def make_cnn(width=32, dropout=0.3):
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(s):
            super().__init__()
            L, c = [], 3
            for i in range(4):
                o = width * 2 ** i
                L += [nn.Conv2d(c, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
                c = o
            s.body, s.drop, s.fc, s.emb_dim = nn.Sequential(*L), nn.Dropout(dropout), nn.Linear(c, 5), c

        def embed(s, x):
            return s.body(x).mean((2, 3))

        def forward(s, x):
            return s.fc(s.drop(s.embed(x)))
    return SmallCNN()


def to_tensor(b, mean, std, dev):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(b)).to(dev).permute(0, 3, 1, 2).float() / 255.0
    m = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    return (x - m) / s


def load_cnn(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    dev = device()
    m = make_cnn(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["mean"], ck["std"]


def run_cnn(model, mean, std, getter, n, bs=256):
    """Probabilities and embeddings for n images; getter(i, j) returns uint8 images i..j."""
    import torch
    dev, P, E = device(), [], []
    with torch.no_grad():
        for i in range(0, n, bs):
            x = to_tensor(getter(i, min(n, i + bs)), mean, std, dev)
            e = model.embed(x)
            P.append(torch.softmax(model.fc(e), 1).cpu().numpy())
            E.append(e.cpu().numpy())
    return np.concatenate(P), np.concatenate(E)


def cnn_on_df(path, df):
    m, mean, std = load_cnn(path)
    return run_cnn(m, mean, std, lambda a, b: read_images(df.iloc[a:b]), len(df))


def load_mlp(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = make_mlp(ck["hidden"], ck["dropout"])
    m.load_state_dict(ck["state"])
    return m.eval(), joblib.load(Path(path).with_name("mlp_prep.joblib"))


def mlp_probs(path, X):
    import torch
    m, prep = load_mlp(path)
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(prep.transform(X), dtype=torch.float32)), 1).numpy()
