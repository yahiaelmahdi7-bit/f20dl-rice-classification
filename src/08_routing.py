"""Stage 8: temperature, floors, familiarity gates and the confidence chain. --fit on validation, --score on test."""
import argparse, json
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

TARGET = {"hgb": 0.9995, "cnn": 0.99}
COVERAGE = 0.95


def risk_coverage(key, correct):
    """Error rate among the top-c most confident answers, for c = 1/n .. 1."""
    o = np.argsort(-key, kind="stable")
    err = 1 - correct[o].astype(float)
    return np.arange(1, len(o) + 1) / len(o), np.cumsum(err) / np.arange(1, len(o) + 1)


def err_at(key, correct, cov=COVERAGE):
    c, r = risk_coverage(key, correct)
    return float(r[int(np.ceil(cov * len(c))) - 1])


def chain(Ph, Pc, fh, fc, unf):
    """Route per grain: hgb if above its floor, else cnn if above its floor, else unsure; unfamiliar overrides."""
    ch, cc = Ph.max(1), Pc.max(1)
    route = np.where(ch >= fh, "hgb", np.where(cc >= fc, "cnn", "unsure")).astype(object)
    route[unf] = "unfamiliar"
    ans = np.where(route == "hgb", Ph.argmax(1), Pc.argmax(1))
    key = np.where(ch >= fh, 2 + ch, np.where(cc >= fc, 1 + cc, cc))
    return route, ans, key


def report(tag, y, Ph, Pcs, R, unf_t, unf_i, out):
    """Tables and figures for one split. Ph: hgb probs; Pcs: {seed: cnn probs}; R: routing dict."""
    rows = []
    cal = {}
    for name, P, T in [("hgb", Ph, R["hgb"]["temperature"])] + [(f"cnn_seed{s}", p, R["cnn"]["temperature"] if s == 0 else C.temperature(p, y) if tag == "val" else R["cnn"]["temperature"]) for s, p in Pcs.items()]:
        Pk = C.apply_temp(P, T)
        cal[name] = Pk
        ok = Pk.argmax(1) == y
        rows.append(dict(split=tag, model=name, n=len(y), accuracy=float(ok.mean()), ece_raw=C.ece(P.max(1), P.argmax(1) == y), ece_calibrated=C.ece(Pk.max(1), ok),
                         temperature=T, err_at_95_coverage=err_at(Pk.max(1), ok)))
    M = pd.DataFrame(rows)
    M.to_csv(out / f"{tag}_calibration.csv", index=False); C.wrote(out / f"{tag}_calibration.csv")
    Pc0 = cal["cnn_seed0"]
    unf = unf_t | unf_i
    route, ans, key = chain(cal["hgb"], Pc0, R["hgb"]["floor"], R["cnn"]["floor"], unf)
    right = ans == y
    rt = []
    for r in ["hgb", "cnn", "unsure", "unfamiliar"]:
        m = route == r
        rt.append(dict(split=tag, route=r, grains=int(m.sum()), share=float(m.mean()), errors=int((~right[m]).sum()) if r in ("hgb", "cnn") else np.nan,
                       accuracy=float(right[m].mean()) if r in ("hgb", "cnn") and m.any() else np.nan))
    ans_m = np.isin(route, ["hgb", "cnn"])
    rt.append(dict(split=tag, route="answered_total", grains=int(ans_m.sum()), share=float(ans_m.mean()), errors=int((~right[ans_m]).sum()),
                   accuracy=float(right[ans_m].mean()) if ans_m.any() else np.nan))
    rt.append(dict(split=tag, route="unfamiliar_table_gate", grains=int(unf_t.sum()), share=float(unf_t.mean()), errors=np.nan, accuracy=np.nan))
    rt.append(dict(split=tag, route="unfamiliar_image_gate", grains=int(unf_i.sum()), share=float(unf_i.mean()), errors=np.nan, accuracy=np.nan))
    rt.append(dict(split=tag, route="chain_err_at_95_coverage", grains=len(y), share=COVERAGE, errors=np.nan, accuracy=1 - err_at(key, right)))
    pd.DataFrame(rt).to_csv(out / f"{tag}_routes.csv", index=False); C.wrote(out / f"{tag}_routes.csv")
    print(M.round(5).to_string(index=False)); print(pd.DataFrame(rt).round(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, k, ok in [("hgb", cal["hgb"].max(1), cal["hgb"].argmax(1) == y), ("cnn seed0", Pc0.max(1), Pc0.argmax(1) == y), ("chain", key, right)]:
        c, r = risk_coverage(k, ok)
        ax.plot(c, 100 * r, label=name)
    ax.axvline(COVERAGE, color="grey", ls="--")
    ax.set(title=f"Risk-coverage ({tag}, n={len(y)})", xlabel="Coverage (fraction answered)", ylabel="Error rate among answered (%)", yscale="symlog", ylim=(0, None))
    ax.legend()
    C.savefig(fig, f"routing_risk_coverage_{tag}")
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (nm, raw) in zip(axs, [("hgb", Ph), ("cnn seed0", Pcs[0])]):
        for lab, P in [("raw", raw), ("calibrated", cal["hgb"] if nm == "hgb" else Pc0)]:
            cf, ok = P.max(1), P.argmax(1) == y
            b = np.clip((cf * 10).astype(int), 0, 9) if cf.min() < 0.9 else np.clip(((cf - cf.min()) / max(1 - cf.min(), 1e-9) * 10).astype(int), 0, 9)
            xs = [cf[b == i].mean() for i in range(10) if (b == i).any()]
            ys = [ok[b == i].mean() for i in range(10) if (b == i).any()]
            ax.plot(xs, ys, "o-", label=lab)
        ax.plot([0, 1], [0, 1], "k:")
        ax.set(title=f"Reliability, {nm} ({tag}, n={len(y)})", xlabel="Mean confidence", ylabel="Accuracy in bin")
        ax.legend()
    C.savefig(fig, f"routing_reliability_{tag}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit", action="store_true")
    g.add_argument("--score", action="store_true")
    a = ap.parse_args()
    out = C.stage_dir("routing")
    C.MODELS.mkdir(parents=True, exist_ok=True)
    emb = C.DERIVED / "embeddings"
    if a.fit:
        tr, va = C.load_split("train"), C.load_split("val")
        y = va.y.to_numpy()
        Ph = C.probs(C.load_preds("hgb", "all106", 0, "val"))
        Pcs = {s: C.probs(C.load_preds("cnn", "image", s, "val")) for s in C.SEEDS if C.load_preds("cnn", "image", s, "val") is not None}
        R = {}
        for nm, P in [("hgb", Ph), ("cnn", Pcs[0])]:
            T = C.temperature(P, y)
            Pk = C.apply_temp(P, T)
            R[nm] = {"temperature": T, "target_accuracy": TARGET[nm], "floor": C.floor_for(Pk.max(1), Pk.argmax(1) == y, TARGET[nm])}
        gt = C.Gate().fit(tr[C.ALL106].to_numpy(np.float64), tr.y.to_numpy(), va[C.ALL106].to_numpy(np.float64))
        gi = C.Gate().fit(np.load(emb / "cnn_seed0_train.npy"), tr.y.to_numpy(), np.load(emb / "cnn_seed0_val.npy"))
        joblib.dump(gt, C.MODELS / "gate_table.joblib"); joblib.dump(gi, C.MODELS / "gate_image.joblib")
        R["gate_table"] = {"cut": gt.cut, "percentile": 1.0}
        R["gate_image"] = {"cut": gi.cut, "percentile": 1.0}
        (C.MODELS / "routing.json").write_text(json.dumps(R, indent=1)); C.wrote(C.MODELS / "routing.json")
        unf_t = gt.unfamiliar(va[C.ALL106].to_numpy(np.float64)); unf_i = gi.unfamiliar(np.load(emb / "cnn_seed0_val.npy"))
        report("val", y, Ph, Pcs, R, unf_t, unf_i, out)
    else:
        R = json.loads((C.MODELS / "routing.json").read_text())
        gt, gi = joblib.load(C.MODELS / "gate_table.joblib"), joblib.load(C.MODELS / "gate_image.joblib")
        df = C.load_split("test")       # the single gated test read of this stage
        _, E = C.cnn_on_df(C.MODELS / "cnn_seed0.pt", df)
        y = df.y.to_numpy()
        Ph = C.load_preds("hgb", "all106", 0, "test")
        assert (Ph.grain_id.to_numpy() == df.grain_id.to_numpy()).all(), "run stage 7 on test first"
        Pcs = {}
        for s in C.SEEDS:
            d = C.load_preds("cnn", "image", s, "test")
            if d is not None: Pcs[s] = C.probs(d)
        unf_t = gt.unfamiliar(df[C.ALL106].to_numpy(np.float64)); unf_i = gi.unfamiliar(E)
        report("test", y, C.probs(Ph), Pcs, R, unf_t, unf_i, out)


if __name__ == "__main__":
    main()
