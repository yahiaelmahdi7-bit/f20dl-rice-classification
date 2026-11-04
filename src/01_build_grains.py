"""Stage 1: measure images, link them one-to-one to table rows, drop duplicates, split, cache images."""
import argparse, glob, os, re, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching as mwm
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split
import common as C

# table column <- image measure, used for linking
F = [("AREA", "area"), ("MAJOR_AXIS", "major"), ("MINOR_AXIS", "minor"), ("meanRR", "mR"), ("meanRG", "mG"), ("meanRB", "mB")]


def measure(p):
    a = np.asarray(Image.open(p).convert("RGB"))
    c = C.grain_mask(a)
    rp = regionprops(c.astype(np.uint8))[0]
    return dict(path=p, area=int(c.sum()), major=rp.axis_major_length, minor=rp.axis_minor_length,
                mR=a[..., 0][c].mean(), mG=a[..., 1][c].mean(), mB=a[..., 2][c].mean())


def small(p):
    return C.resize128(np.asarray(Image.open(p).convert("RGB")))


def read_arff(path):
    lines = open(path).read().splitlines()
    names = [l.split()[1] for l in lines if l.lower().startswith("@attribute")]
    start = next(i for i, l in enumerate(lines) if l.lower().startswith("@data")) + 1
    return pd.read_csv(path, skiprows=start, names=names, na_values="?")


def list_images():
    ps = sorted(glob.glob(str(C.RAW / "Rice_Image_Dataset" / "*" / "*.jpg")))
    d = pd.DataFrame({"path": ps})
    d["cls"] = d.path.map(lambda p: os.path.basename(os.path.dirname(p)))
    d["file_no"] = d.path.str.extract(r"\((\d+)\)\.jpg$")[0].astype(int)
    d["lot"] = np.where((d.cls == "Jasmine") & (d.file_no >= C.LATE_FROM), "late", "main")
    return d


def measure_all(paths, cache):
    if cache and cache.exists():
        m = pd.read_parquet(cache)
        if list(m.path) == list(paths):
            print("reusing measures", cache); return m
    with Pool(os.cpu_count()) as pool:
        rows = []
        for i, r in enumerate(pool.imap(measure, paths, chunksize=256)):
            rows.append(r)
            if i % 10000 == 0: print(f"  measured {i}/{len(paths)}", flush=True)
    m = pd.DataFrame(rows)
    if cache: m.to_parquet(cache, index=False)
    return m


def link(msc, im):
    """One-to-one image -> table row per class; returns msc_row per image."""
    sd = np.array([msc[m].std() for m, _ in F])
    out = pd.Series(-1, index=im.index)
    for c in C.CLASSES:
        ai, mi = im[im.cls == c], msc[msc.CLASS == c]
        I, R = ai[[i for _, i in F]].values / sd, mi[[m for m, _ in F]].values / sd
        d, j = cKDTree(R).query(I, k=8)
        n = len(I)
        G = csr_matrix(((d + 1e-6).ravel(), (np.repeat(np.arange(n), 8), j.ravel())), shape=(n, len(R)))
        r, col = mwm(G)
        out[ai.index.values[r]] = mi.index.values[col]
        print(f"  linked {c}: {n} images, unique rows {len(set(col))}", flush=True)
    assert (out >= 0).all()
    return out.values


def make_splits(g):
    key = g["class"] + "|" + g["lot"]
    tr, rest = train_test_split(g.grain_id, test_size=0.30, stratify=key, random_state=C.SEED)
    k2 = key[g.grain_id.isin(rest)]
    va, te = train_test_split(g.grain_id[g.grain_id.isin(rest)], test_size=0.5, stratify=k2, random_state=C.SEED)
    s = pd.Series("train", index=g.grain_id)
    s[va.values], s[te.values] = "val", "test"
    return s.reindex(g.grain_id).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=2000 if C.SMOKE else 0)
    a = ap.parse_args()
    C.DERIVED.mkdir(parents=True, exist_ok=True)
    full = a.subset == 0
    msc = read_arff(C.RAW / "Rice_MSC_Dataset" / "Rice_MSC_Dataset.arff")
    imgs = list_images()
    print(f"table {msc.shape}, images {len(imgs)}")
    if not full:
        imgs = imgs.groupby(["cls", "lot"], group_keys=False).sample(frac=a.subset / len(imgs), random_state=C.SEED)
        imgs = imgs.sort_values("path").reset_index(drop=True)
    else:
        assert len(imgs) == 75000 and len(msc) == 75000
    m = measure_all(imgs.path.tolist(), C.DERIVED / "measures.parquet" if full else None)
    imgs = imgs.reset_index(drop=True).merge(m, on="path")
    imgs["msc_row"] = link(msc, imgs)
    if full: assert imgs.msc_row.nunique() == 75000, "links not one-to-one"
    n_links = len(imgs)
    g = imgs.merge(msc.reset_index().rename(columns={"index": "msc_row"}), on="msc_row")
    assert (g["CLASS"] == g["cls"]).all()
    g = g.copy()
    drops = 0
    if full:
        key = g[C.ALL106].round(6)
        dup = key.duplicated(keep=False)
        grp = key[dup].apply(tuple, axis=1)
        order = g[dup].assign(k=grp).sort_values(["k", "file_no"])
        drop_idx = order[order.k.duplicated(keep="first")].index
        drops = len(drop_idx)
        assert set(g.loc[drop_idx, "cls"]) == {"Ipsala"}
        g = g.drop(index=drop_idx)
    g = g.rename(columns={"cls": "class"}).sort_values(["class", "file_no"]).reset_index(drop=True)
    g["grain_id"] = g["class"] + "-" + g.file_no.astype(str)
    g["y"] = g["class"].map({c: i for i, c in enumerate(C.CLASSES)})
    g["row"] = np.arange(len(g))
    n_late = int((g.lot == "late").sum())
    print(f"links {n_links}, drops {drops}, grains {len(g)}, late {n_late}", flush=True)
    if full:
        assert (n_links, drops, len(g), n_late) == (75000, 297, 74703, 3283), "stage 1 asserts failed"
    g["split"] = make_splits(g)
    sp = g[["grain_id", "class", "lot", "file_no", "msc_row", "split"]]
    if C.SPLITS.exists() and full:
        old = pd.read_csv(C.SPLITS)
        assert old.equals(sp.reset_index(drop=True)), "splits.csv disagrees with committed copy"
        print("splits.csv agrees with committed copy")
    else:
        sp.to_csv(C.SPLITS, index=False); C.wrote(C.SPLITS)
    cols = ["grain_id", "class", "y", "lot", "file_no", "msc_row", "path", "row", "split"] + C.ALL106
    g[cols].to_parquet(C.DERIVED / "grains.parquet", index=False); C.wrote(C.DERIVED / "grains.parquet")
    mm = np.memmap(C.cache_path(), dtype=np.uint8, mode="w+", shape=(len(g), C.IMG, C.IMG, 3))
    with Pool(os.cpu_count()) as pool:
        for i, arr in enumerate(pool.imap(small, g.path.tolist(), chunksize=64)):
            mm[i] = arr
            if i % 10000 == 0: print(f"  cached {i}/{len(g)}", flush=True)
    mm.flush()
    C.wrote(C.cache_path())
    t = g.groupby("split").agg(n=("y", "size"), late=("lot", lambda s: (s == "late").mean()))
    print(t.assign(share=t.n / t.n.sum()).round(4).to_string())
    print(pd.crosstab(g["class"], g["split"], normalize="columns").round(4).to_string())


if __name__ == "__main__":
    main()
