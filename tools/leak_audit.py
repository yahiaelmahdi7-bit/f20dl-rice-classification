"""Leak audit: do any grains, table rows, files or pixels sit in more than one split?

Takes no argument. Reads data/splits.csv without its class column and grains.parquet without class or y.
Writes results/quality/leak_audit.csv with columns `check,value`, one row per flattened key.
"""
import hashlib, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C  # noqa: E402


def audit(sp, g, mm, n_probe=8000):
    """sp: grain_id, msc_row, split. g: grain_id, split, path, row and the 106 columns. mm: image cache array. Returns nested dict."""
    R = {}
    R["counts"] = sp.split.value_counts().to_dict()
    R["dup_grain_id"], R["dup_msc_row"] = int(sp.grain_id.duplicated().sum()), int(sp.msc_row.duplicated().sum())
    S = {k: set(v) for k, v in sp.groupby("split").grain_id}
    R["id_overlap"] = {f"{a}&{b}": len(S[a] & S[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    m = g.merge(sp, on="grain_id", suffixes=("", "_csv"))
    R["parquet_vs_csv_split_mismatch"], R["parquet_rows"] = int((m.split != m.split_csv).sum()), len(g)

    def cross(key, name):
        d = pd.DataFrame({"k": np.asarray(key), "s": g.split.values})
        c = d.groupby("k").s.agg(["size", "nunique"])
        x = d[d.k.isin(c.index[c["nunique"] > 1])]
        R[name] = dict(repeat_groups=int((c["size"] > 1).sum()), groups_across_splits=int((c["nunique"] > 1).sum()),
                       grains_across_splits=len(x), by_split=x.s.value_counts().to_dict())
        return x

    F = C.ALL106
    H = lambda df: pd.util.hash_pandas_object(df, index=False)
    cross(H(g[F]), "table_exact"); cross(H(g[F].round(6)), "table_6dp"); cross(H(g[F].round(3)), "table_3dp")
    cross(H(g[F[:16]].round(6)), "shape16_6dp"); cross(H(g[F[16:]].round(6)), "colour90_6dp")
    cross([hashlib.sha1(Path(p).read_bytes()).hexdigest() for p in g.path], "jpg_bytes")
    k1, k2, k3 = [], [], []
    for r in g.row.to_numpy():
        a = mm[r]; k1.append(hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest())
        q = (a[..., 1].reshape(16, 8, 16, 8).mean((1, 3)).astype(np.uint8) >> 3); 
        v = [np.rot90(x, i)[::2, ::2, 1] for x in (a, a[:, ::-1]) for i in range(4)]; w = [np.rot90(x, i) for x in (q, q[:, ::-1]) for i in range(4)]
        k2.append(min(hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=16).hexdigest() for x in v))
        k3.append(min(hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=16).hexdigest() for x in w))
    cross(k1, "pixels128"); cross(k2, "pixels_any_rotation_flip"); cross(k3, "thumb16_any_rotation_flip")

    c = C.features("distinct91"); tr = (g.split == "train").values
    imp = SimpleImputer(strategy="median").fit(g.loc[tr, c]); sc = StandardScaler().fit(imp.transform(g.loc[tr, c]))
    Z = sc.transform(imp.transform(g[c])); nn = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(Z[tr])
    rng = np.random.default_rng(0); out = {}
    n_tr = int(tr.sum())
    for name, idx, col in [("train_to_train", rng.choice(np.where(tr)[0], min(n_probe, n_tr), replace=False), 1),
                           ("val_to_train", np.where(g.split == "val")[0], 0), ("test_to_train", np.where(g.split == "test")[0], 0)]:
        d = nn.kneighbors(Z[idx])[0][:, col]
        out[name] = dict(n=len(d), min=float(d.min()), **{f"p{p}": float(np.percentile(d, p)) for p in (0.1, 1, 10, 50)}, under_0_5=int((d < .5).sum()))
    R["nearest_train_neighbour_distinct91"] = out
    return R


def flatten(R, prefix=""):
    """Nested dict -> [(dotted_name, value)], one per leaf. Empty dicts give no row."""
    rows = []
    for k, v in R.items():
        name = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            rows += flatten(v, name)
        else:
            rows.append((name, v.item() if hasattr(v, "item") else v))
    return rows


def write_rows(rows, path):
    """Write check,value rows to a temp name beside path, then rename."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        pd.DataFrame(rows, columns=["check", "value"]).to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main():
    assert not C.SMOKE
    sp = pd.read_csv(C.SPLITS, usecols=["grain_id", "msc_row", "split"])   # no class column
    g = pd.read_parquet(C.DERIVED / "grains.parquet", columns=["grain_id", "split", "path", "row", *C.ALL106])  # no class, no y
    rows = flatten(audit(sp, g, C.open_cache()))
    out = C.stage_dir("quality") / "leak_audit.csv"
    write_rows(rows, out)
    print(pd.DataFrame(rows, columns=["check", "value"]).to_string(index=False)); C.wrote(out)


if __name__ == "__main__":
    main()
