"""Stage 6: MLP on all106 and SmallCNN on 128 px images; tuning, seeds, curves, embeddings."""
import argparse, itertools, json, time
import joblib, numpy as np, pandas as pd
import torch, torch.nn as nn
import common as C

DEFAULT_CNN = {"width": 32, "lr": 1e-3, "dropout": 0.3}
DEFAULT_MLP = {"hidden": [256, 128], "lr": 1e-3, "dropout": 0.2}
MLP_GRID = [{"hidden": h, "lr": lr, "dropout": d} for h in ([100], [50, 50], [256, 128]) for lr in (1e-3, 3e-4) for d in (0.0, 0.2)]
CNN_GRID = [{"width": w, "lr": lr, "dropout": d} for w in (16, 32) for lr in (1e-3, 3e-4) for d in (0.0, 0.3)]
if C.SMOKE:
    MLP_GRID, CNN_GRID = MLP_GRID[:2], CNN_GRID[:1]
EP_MLP, PAT_MLP = (1, 10) if C.SMOKE else (100, 10)
EP_CNN, PAT_CNN = (1, 4) if C.SMOKE else (20, 4)
EP_TUNE = 1 if C.SMOKE else 8


def log_epoch(tag, ep, secs, cfg, n, extra=""):
    p = C.stage_dir("neural") / "epoch_times.csv"
    new = not p.exists()
    with open(p, "a") as f:
        if new: f.write("time,tag,epoch,seconds,device,n_train,config\n")
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{tag},{ep},{secs:.2f},{C.device()},{n},\"{json.dumps(cfg)}\"\n")
    print(f"[{tag}] epoch {ep}: {secs:.1f} s/epoch on {C.device()} {extra}", flush=True)


def sync():
    if C.device().type == "mps": torch.mps.synchronize()


def chan_stats(tr):
    s = tr.sample(min(len(tr), 5000), random_state=C.SEED)
    x = C.read_images(s).astype(np.float32) / 255
    return x.mean((0, 1, 2)).tolist(), x.std((0, 1, 2)).tolist()


def train_cnn(cfg, seed, tr, va, epochs, patience, tag, shuffle=False, aug=True, stats=None):
    """Returns model (best-val-loss weights), history, mean, std."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = C.device()
    mean, std = stats or chan_stats(tr)
    model = C.make_cnn(cfg["width"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    ytr = tr.y.to_numpy().copy()
    if shuffle: ytr = rng.permutation(ytr)
    best, bad, hist, state = 1e9, 0, [], None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(len(tr))
        tl = 0.0
        for i in range(0, len(perm), 128):
            idx = np.sort(perm[i:i + 128])
            x = C.to_tensor(C.read_images(tr.iloc[idx]), mean, std, dev)
            if aug:
                fh = torch.rand(len(idx), device=dev) < 0.5
                x = torch.where(fh.view(-1, 1, 1, 1), x.flip(3), x)
                fv = torch.rand(len(idx), device=dev) < 0.5
                x = torch.where(fv.view(-1, 1, 1, 1), x.flip(2), x)
            y = torch.tensor(ytr[idx], device=dev)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(idx)
        sched.step()
        sync()
        secs = time.time() - t0
        row = {"tag": tag, "epoch": ep, "train_loss": tl / len(tr), "seconds": secs}
        if va is not None:
            P, _ = C.run_cnn(model.eval(), mean, std, lambda a, b: C.read_images(va.iloc[a:b]), len(va))
            yv = va.y.to_numpy()
            row.update(val_loss=float(-np.log(np.clip(P[np.arange(len(yv)), yv], 1e-12, 1)).mean()), val_acc=float((P.argmax(1) == yv).mean()))
            if row["val_loss"] < best - 1e-6:
                best, bad, state = row["val_loss"], 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        hist.append(row)
        log_epoch(tag, ep, secs, cfg, len(tr), f"train_loss {row['train_loss']:.4f} val_acc {row.get('val_acc', float('nan')):.4f}")
        if va is not None and bad >= patience: break
    if state is not None: model.load_state_dict(state)
    return model.eval(), hist, mean, std


def cnn_eval(model, mean, std, df):
    return C.run_cnn(model, mean, std, lambda a, b: C.read_images(df.iloc[a:b]), len(df))


def train_mlp(cfg, seed, Xtr, ytr, Xva, yva, epochs, patience, tag):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = C.make_mlp(cfg["hidden"], cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva)
    best, bad, hist, state = 1e9, 0, [], None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.tensor(rng.permutation(len(Xt)))
        tl = 0.0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = nn.functional.cross_entropy(model(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            lo = model(Xv)
            vl = nn.functional.cross_entropy(lo, yv).item()
            va_acc = (lo.argmax(1) == yv).float().mean().item()
        hist.append({"tag": tag, "epoch": ep, "train_loss": tl / len(Xt), "val_loss": vl, "val_acc": va_acc, "seconds": time.time() - t0})
        if vl < best - 1e-6: best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else: bad += 1
        if bad >= patience: break
    model.load_state_dict(state)
    print(f"[{tag}] {len(hist)} epochs, best val loss {best:.4f}", flush=True)
    return model.eval(), hist


def mlp_proba(model, X):
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X, dtype=torch.float32)), 1).numpy()


def save_curves(rows):
    p = C.stage_dir("neural") / "curves.csv"
    d = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        d = pd.concat([old[~old.tag.isin(d.tag.unique())], d])
    d.to_csv(p, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["mlp", "cnn"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    ap.add_argument("--overfit-test", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--time-epoch", action="store_true")
    a = ap.parse_args()
    out = C.stage_dir("neural")
    C.MODELS.mkdir(parents=True, exist_ok=True)
    tr, va = C.load_split("train"), C.load_split("val")
    bpath = out / "best_config.json"
    cfgs = json.loads(bpath.read_text()) if bpath.exists() else {}

    if a.overfit_test:
        sub = tr.sample(256, random_state=C.SEED).reset_index(drop=True)
        cfg = {**DEFAULT_CNN, "dropout": 0.0}
        model, hist, mean, std = train_cnn(cfg, 0, sub, None, 60 if not C.SMOKE else 2, 99, "overfit", aug=False)
        P, _ = cnn_eval(model, mean, std, sub)
        acc = float((P.argmax(1) == sub.y.to_numpy()).mean())
        print(f"overfit test: train accuracy on 256 grains = {acc:.4f} -> {'PASS' if acc >= 0.99 else 'FAIL'}")
        raise SystemExit(0 if acc >= 0.99 or C.SMOKE else 1)
    if a.time_epoch:
        cfg = DEFAULT_CNN
        train_cnn(cfg, 0, tr, None, 1, 1, "time_epoch")
        return
    if a.control:
        rows = []
        if a.only in (None, "cnn"):
            sub = tr.sample(min(len(tr), 10000), random_state=C.SEED).reset_index(drop=True)
            m, h, mean, std = train_cnn(cfgs.get("cnn", DEFAULT_CNN), 0, sub, va, 6 if not C.SMOKE else 1, 99, "control_cnn", shuffle=True)
            rows.append({"model": "cnn", "val_accuracy": h[-1]["val_acc"], "train_n": len(sub)})
        if a.only in (None, "mlp"):
            prep = C.Pipeline([("imp", C.SimpleImputer(strategy="median")), ("sc", C.StandardScaler())]).fit(tr[C.ALL106])
            ys = np.random.default_rng(0).permutation(tr.y.to_numpy())
            m, h = train_mlp(cfgs.get("mlp", DEFAULT_MLP), 0, prep.transform(tr[C.ALL106]), ys, prep.transform(va[C.ALL106]), va.y.to_numpy(), EP_MLP, PAT_MLP, "control_mlp")
            P = mlp_proba(m, prep.transform(va[C.ALL106]))
            rows.append({"model": "mlp", "val_accuracy": float((P.argmax(1) == va.y.to_numpy()).mean()), "train_n": len(tr)})
        d = pd.DataFrame(rows)
        d["in_18_22_band"] = d.val_accuracy.between(0.18, 0.22)
        d.to_csv(out / "control.csv", index=False); C.wrote(out / "control.csv"); print(d.to_string())
        return

    if a.only in (None, "mlp"):
        prep = C.Pipeline([("imp", C.SimpleImputer(strategy="median")), ("sc", C.StandardScaler())]).fit(tr[C.ALL106])
        Xtr, Xva = prep.transform(tr[C.ALL106]), prep.transform(va[C.ALL106])
        ytr, yva = tr.y.to_numpy(), va.y.to_numpy()
        joblib.dump(prep, C.MODELS / "mlp_prep.joblib")
        rows, curves = [], []
        for cfg in MLP_GRID:
            m, h = train_mlp(cfg, 0, Xtr, ytr, Xva, yva, EP_MLP, PAT_MLP, "mlp_tune")
            rows.append({**cfg, "hidden": str(cfg["hidden"]), "val_macro_f1": C.macro_f1(yva, mlp_proba(m, Xva).argmax(1))})
        pd.DataFrame(rows).to_csv(out / "tuning_mlp.csv", index=False); C.wrote(out / "tuning_mlp.csv")
        cfgs["mlp"] = MLP_GRID[int(np.argmax([r["val_macro_f1"] for r in rows]))]
        for s in a.seeds:
            m, h = train_mlp(cfgs["mlp"], s, Xtr, ytr, Xva, yva, EP_MLP, PAT_MLP, f"mlp_seed{s}")
            curves += h
            torch.save({"state": m.state_dict(), "hidden": cfgs["mlp"]["hidden"], "dropout": cfgs["mlp"]["dropout"]}, C.MODELS / f"mlp_seed{s}.pt")
            C.save_preds("mlp", "all106", s, "val", va.grain_id, yva, mlp_proba(m, Xva))
        save_curves(curves)
        bpath.write_text(json.dumps(cfgs, indent=1))

    if a.only in (None, "cnn"):
        sub = tr.sample(min(len(tr), 10000), random_state=C.SEED).reset_index(drop=True)
        stats = chan_stats(tr)
        rows = []
        for cfg in CNN_GRID:
            m, h, mean, std = train_cnn(cfg, 0, sub, va, EP_TUNE, 99, "cnn_tune", stats=stats)
            P, _ = cnn_eval(m, mean, std, va)
            rows.append({**cfg, "val_macro_f1": C.macro_f1(va.y.to_numpy(), P.argmax(1))})
        pd.DataFrame(rows).to_csv(out / "tuning_cnn.csv", index=False); C.wrote(out / "tuning_cnn.csv")
        cfgs["cnn"] = CNN_GRID[int(np.argmax([r["val_macro_f1"] for r in rows]))]
        bpath.write_text(json.dumps(cfgs, indent=1))
        curves = []
        emb = C.DERIVED / "embeddings"
        emb.mkdir(parents=True, exist_ok=True)
        for s in a.seeds:
            m, h, mean, std = train_cnn(cfgs["cnn"], s, tr, va, EP_CNN, PAT_CNN, f"cnn_seed{s}", stats=stats)
            curves += h
            torch.save({"state": {k: v.cpu() for k, v in m.state_dict().items()}, "cfg": {"width": cfgs["cnn"]["width"], "dropout": cfgs["cnn"]["dropout"]},
                        "mean": mean, "std": std}, C.MODELS / f"cnn_seed{s}.pt")
            Pv, Ev = cnn_eval(m, mean, std, va)
            C.save_preds("cnn", "image", s, "val", va.grain_id, va.y, Pv)
            _, Et = cnn_eval(m, mean, std, tr)
            np.save(emb / f"cnn_seed{s}_train.npy", Et.astype(np.float32)); np.save(emb / f"cnn_seed{s}_val.npy", Ev.astype(np.float32))
        save_curves(curves)


if __name__ == "__main__":
    main()
