"""Stage 7: compare every model on one split with intervals and paired tests. --split test is one gated test read."""
import argparse, json
import joblib, numpy as np, pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

BASE = ["hgb", "nb", "gmm", "tree", "logreg"]
BASE_FS = ["selected", "shape16", "colour90", "all106"]
NBOOT = 2000


def model_list():
    """(model, featureset, seeds) for everything that can be scored."""
    L = [(m, fs, (0,)) for fs in BASE_FS for m in BASE] + [("nb", "distinct91", (0,))]
    return L + [("mlp", "all106", C.SEEDS), ("cnn", "image", C.SEEDS)]


def test_probs(df, m, fs, k):
    """Test-split probabilities from a saved model."""
    if m == "cnn": return C.cnn_on_df(C.MODELS / f"cnn_seed{k}.pt", df)[0]
    if m == "mlp": return C.mlp_probs(C.MODELS / f"mlp_seed{k}.pt", df[C.ALL106].to_numpy(np.float64))
    obj = joblib.load(C.MODELS / "baselines" / f"{m}__{fs}.joblib")
    return obj["pipe"].predict_proba(df[obj["cols"]])


def boot_f1(y, pred, idx):
    out = np.empty(len(idx))
    for i, ix in enumerate(idx):
        yt, yp = y[ix], pred[ix]
        f = []
        for c in range(5):
            tp = np.sum((yt == c) & (yp == c)); fp = np.sum((yt != c) & (yp == c)); fn = np.sum((yt == c) & (yp != c))
            f.append(2 * tp / max(2 * tp + fp + fn, 1))
        out[i] = np.mean(f)
    return out


def holm(p):
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    a = ap.parse_args()
    out = C.stage_dir("compare")
    sp = a.split
    preds = {}
    if sp == "test":
        df = C.load_split("test")          # the single gated test read of this stage
        print(f"test read: {len(df)} grains", flush=True)
        for m, fs, seeds in model_list():
            for k in seeds:
                try:
                    P = test_probs(df, m, fs, k)
                except FileNotFoundError:
                    print("missing model", m, fs, k); continue
                C.save_preds(m, fs, k, "test", df.grain_id, df.y, P)
                preds[(m, fs, k)] = C.load_preds(m, fs, k, "test")
    else:
        for m, fs, seeds in model_list():
            for k in seeds:
                d = C.load_preds(m, fs, k, "val")
                if d is not None: preds[(m, fs, k)] = d
    assert preds, "no predictions found"
    ref = next(iter(preds.values()))
    ids = ref.grain_id.to_numpy()
    y = ref.y.to_numpy().astype(int)
    for key, d in preds.items():
        assert (d.grain_id.to_numpy() == ids).all(), f"grain order differs for {key}"
    n = len(y)
    rng = np.random.default_rng(C.SEED)
    idx = rng.integers(0, n, (NBOOT, n))       # one set of resamples shared by every model (paired)
    pred = {k: C.probs(d).argmax(1) for k, d in preds.items()}
    ok = {k: (p == y) for k, p in pred.items()}
    rows, rec, cm = [], [], []
    for k, p in pred.items():
        m, fs, s = k
        acc_b = ok[k][idx].mean(1)
        f1_b = boot_f1(y, p, idx)
        rows.append(dict(model=m, featureset=fs, seed=s, split=sp, n=n, accuracy=ok[k].mean(), errors=int((~ok[k]).sum()),
                         acc_lo=np.percentile(acc_b, 2.5), acc_hi=np.percentile(acc_b, 97.5), macro_f1=C.macro_f1(y, p),
                         f1_lo=np.percentile(f1_b, 2.5), f1_hi=np.percentile(f1_b, 97.5)))
        for c in range(5):
            rec.append(dict(model=m, featureset=fs, seed=s, split=sp, **{"class": C.CLASSES[c]}, n=int((y == c).sum()),
                            recall=float((p[y == c] == c).mean()), errors=int((p[y == c] != c).sum())))
        for t in range(5):
            for q in range(5):
                cm.append(dict(model=m, featureset=fs, seed=s, true=C.CLASSES[t], predicted=C.CLASSES[q], count=int(((y == t) & (p == q)).sum())))
    M = pd.DataFrame(rows)
    M.to_csv(out / f"{sp}_metrics.csv", index=False); C.wrote(out / f"{sp}_metrics.csv")
    pd.DataFrame(rec).to_csv(out / f"{sp}_recall.csv", index=False); C.wrote(out / f"{sp}_recall.csv")
    pd.DataFrame(cm).to_csv(out / f"{sp}_confusion.csv", index=False); C.wrote(out / f"{sp}_confusion.csv")
    sd = M.groupby(["model", "featureset"]).agg(seeds=("seed", "size"), acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                                                  f1_mean=("macro_f1", "mean"), f1_sd=("macro_f1", "std"), err_min=("errors", "min"), err_max=("errors", "max")).reset_index()
    sd.to_csv(out / f"{sp}_seed_spread.csv", index=False); C.wrote(out / f"{sp}_seed_spread.csv")

    # paired tests: seed 0 is the primary family, seeds 1 and 2 are a check
    def get(spec, s):
        m, fs = spec.split(":")
        return (m, fs, s if m in ("cnn", "mlp") else 0)
    tests = []
    for s in C.SEEDS:
        for A, B in C.PAIRS:
            ka, kb = get(A, s), get(B, s)
            if ka not in ok or kb not in ok: continue
            b = int((ok[ka] & ~ok[kb]).sum()); c = int((~ok[ka] & ok[kb]).sum())
            p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
            diff = ok[ka][idx].mean(1) - ok[kb][idx].mean(1)
            tests.append(dict(seed=s, model_a=A, model_b=B, b_a_right_b_wrong=b, c_a_wrong_b_right=c, disagreements=b + c, p_exact=p,
                              errors_a=int((~ok[ka]).sum()), errors_b=int((~ok[kb]).sum()), acc_diff=float(ok[ka].mean() - ok[kb].mean()),
                              diff_lo=np.percentile(diff, 2.5), diff_hi=np.percentile(diff, 97.5), family="primary" if s == 0 else "seed_check"))
    T = pd.DataFrame(tests)
    if len(T):
        T["p_holm"] = np.nan
        for s in T.seed.unique():
            i = T.index[T.seed == s]
            T.loc[i, "p_holm"] = holm(T.loc[i, "p_exact"].to_numpy())
    T.to_csv(out / f"{sp}_mcnemar.csv", index=False); C.wrote(out / f"{sp}_mcnemar.csv")
    print(M[M.seed == 0][["model", "featureset", "accuracy", "errors", "macro_f1"]].round(5).to_string(index=False))
    print(T.round(5).to_string(index=False))

    prim = [k for k in [("hgb", "all106", 0), ("cnn", "image", 0), ("mlp", "all106", 0)] if k in ok]
    fig, axs = plt.subplots(1, len(prim), figsize=(4.2 * len(prim), 4), squeeze=False)
    for ax, k in zip(axs[0], prim):
        c = pd.DataFrame(cm)
        c = c[(c.model == k[0]) & (c.featureset == k[1]) & (c.seed == k[2])].pivot(index="true", columns="predicted", values="count").reindex(index=C.CLASSES, columns=C.CLASSES)
        ax.imshow(np.log1p(c.values), cmap="Blues")
        for i in range(5):
            for j in range(5): ax.text(j, i, int(c.values[i, j]), ha="center", va="center", fontsize=7)
        ax.set_xticks(range(5), C.CLASSES, rotation=45, fontsize=7); ax.set_yticks(range(5), C.CLASSES, fontsize=7)
        ax.set(title=f"{k[0]}:{k[1]} seed {k[2]} ({sp}, n={n})", xlabel="Predicted", ylabel="True")
    C.savefig(fig, f"compare_confusion_{sp}")


if __name__ == "__main__":
    main()
