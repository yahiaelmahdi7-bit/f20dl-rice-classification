# Generated with AI assistance (Claude Sonnet, 2026-11-10).
# Prompt: "Write report/make_numbers.py: read committed CSVs under results/ (SUMMARY.csv first,
# raw CSVs only for facts it lacks) and write report/numbers.tex (one LaTeX macro per reported
# number, letters-only names, val/test prefixes, per-seed vs seed-mean names) and report/FACTS.md
# (macro, value, source, row/column). Read-only; writes nothing outside report/."
"""Build report/numbers.tex and report/FACTS.md from committed results/ CSVs.

Deterministic and idempotent. Reads results/ only; writes only report/.
"""
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "report"

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
ROWS = []   # (macro, tex_value, display_value, source, rowcol)
SEEN = set()


def w(s):
    """Letters-only CamelCase: digits spelled as words, separators dropped."""
    out, new = [], True
    for ch in str(s):
        if ch.isdigit():
            out.append(DIGITS[int(ch)])
            new = True
        elif ch.isalpha():
            out.append(ch.upper() if new else ch)
            new = False
        else:
            new = True
    return "".join(out)


def esc(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def add(name, tex, disp, src, rowcol):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in SEEN, f"duplicate macro {name}"
    SEEN.add(name)
    ROWS.append((name, tex, disp, src, rowcol))


def add_int(name, v, src, rc):
    add(name, str(int(v)), str(int(v)), src, rc)


def add_text(name, v, src, rc):
    add(name, esc(v), str(v), src, rc)


def add_prop(name, v, src, rc):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return
    add(name, f"{v:.4f}", f"{v:.4f}", src, rc)


def add_acc(name, v, src, rc):
    """Proportion (4 dp) plus percentage (2 dp, no % sign) under suffix Pct."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return
    add(name, f"{v:.4f}", f"{v:.4f}", src, rc)
    add(name + "Pct", f"{100 * v:.2f}", f"{100 * v:.2f}", src, rc + " (x100)")


def add_p(name, p, src, rc):
    if p >= 1e-3:
        s = f"{p:.4f}"
        add(name, s, s, src, rc)
    else:
        m, e = f"{p:.2e}".split("e")
        s = rf"{m}\times10^{{{int(e)}}}"
        add(name, rf"\ensuremath{{{s}}}", s, src, rc)


def rd(rel):
    return pd.read_csv(RES / rel)


def seedword(i):
    return "Seed" + DIGITS[int(i)]


# ---------------------------------------------------------------- 1 dataset
src = "results/eda/class_balance.csv"
cb = rd("eda/class_balance.csv")
for _, r in cb.iterrows():
    for sp in ("train", "val", "test"):
        add_int(f"Data{sp.capitalize()}{w(r['class'])}", r[sp], src, f"class={r['class']}, {sp}")
    add_int(f"DataTotal{w(r['class'])}", r[["train", "val", "test"]].sum(), src, f"class={r['class']}, train+val+test")
tot = {sp: int(cb[sp].sum()) for sp in ("train", "val", "test")}
for sp in tot:
    add_int(f"Data{sp.capitalize()}Total", tot[sp], src, f"sum of {sp}")
grains = sum(tot.values())
add_int("DataGrainsTotal", grains, src, "sum of train+val+test")
add_int("DataClassCount", len(cb), src, "number of class rows")

src = "results/quality/cleaning_log.csv"
cl = rd("quality/cleaning_log.csv")
raw_images = int(re.match(r"(\d+)", cl.loc[cl.check == "image_decode", "scope"].iloc[0]).group(1))
add_int("DataImagesRaw", raw_images, src, "check=image_decode, scope (leading number)")
for _, r in cl.iterrows():
    add_int(f"Clean{w(r['check'])}Rows", r["rows_affected"], src, f"check={r['check']}, rows_affected")
for chk in ("exact_duplicate_pairs", "missing_cells"):
    r = cl[cl.check == chk].iloc[0]
    add_text(f"Clean{w(chk)}Action", r["action"], src, f"check={chk}, action")
    add_text(f"Clean{w(chk)}Reason", r["reason"], src, f"check={chk}, reason")
dups = int(cl.loc[cl.check == "exact_duplicate_pairs", "rows_affected"].iloc[0])
add_int("DataRowsRemoved", dups, src, "check=exact_duplicate_pairs, rows_affected (only check that drops rows)")
add_int("DataRemovedCheck", raw_images - dups - grains, "derived", "raw images - duplicates dropped - split total (0 means consistent)")

# ---------------------------------------------------------------- 2 features
src = "results/features/selected.json"
sel = json.loads((RES / "features/selected.json").read_text())
add_int("FeatSelectedK", sel["k"], src, "k")
add_text("FeatSelectedNames", ", ".join(sel["selected"]), src, "selected (joined)")
add_acc("FeatBestValAcc", sel["best_val_accuracy"], src, "best_val_accuracy")
for m, k in sel["k_by_model"].items():
    add_int(f"FeatK{w(m)}", k, src, f"k_by_model.{m}")

# ---------------------------------------------------------------- 3 clustering
src = "results/cluster/heldout.csv"
ho = rd("cluster/heldout.csv")
add_int("ClusterNFit", ho.n_fit.iloc[0], src, "n_fit")
add_int("ClusterNVal", ho.n_val.iloc[0], src, "n_val")
for _, r in ho.iterrows():
    base = f"Cluster{w(r.method)}{w(r.featureset)}K{w(int(r.k))}"
    rc = f"featureset={r.featureset}, method={r.method}, k={int(r.k)}"
    add_prop(base + "ARIVal", r.ARI_val, src, rc + ", ARI_val")
    add_prop(base + "NMIVal", r.NMI_val, src, rc + ", NMI_val")
for (fs, me), g in ho.groupby(["featureset", "method"]):
    b = g.loc[g.ARI_val.idxmax()]
    base = f"ClusterBest{w(me)}{w(fs)}"
    rc = f"featureset={fs}, method={me}, argmax ARI_val"
    add_int(base + "K", b.k, src, rc + ", k")
    add_prop(base + "ARIVal", b.ARI_val, src, rc + ", ARI_val")
    add_prop(base + "NMIVal", b.NMI_val, src, rc + ", NMI_val")
src = "results/cluster/cluster_metrics.csv"
cm = rd("cluster/cluster_metrics.csv")
for _, r in cm[cm.featureset == "all106"].iterrows():
    add_prop(f"Cluster{w(r.method)}AllOneZeroSixK{w(int(r.k))}Silhouette", r.silhouette, src,
             f"featureset=all106, method={r.method}, k={int(r.k)}, silhouette")

# ---------------------------------------------------------------- 4/5 model rows
src = "results/SUMMARY.csv"
sm = rd("SUMMARY.csv")
for _, r in sm.iterrows():
    sp = r.split.capitalize()
    tag = f"{w(r.model)}{w(r.featureset)}" + ("SeedMean" if r.seeds > 1 else "")
    rc = f"split={r.split}, model={r.model}, featureset={r.featureset}"
    add_acc(f"{sp}Acc{tag}", r.accuracy, src, rc + ", accuracy")
    add_acc(f"{sp}FOne{tag}", r.macro_f1, src, rc + ", macro_f1")
    add_acc(f"{sp}Acc{tag}Lo", r.acc_lo, src, rc + ", acc_lo (Wilson)")
    add_acc(f"{sp}Acc{tag}Hi", r.acc_hi, src, rc + ", acc_hi (Wilson)")
    if r.seeds > 1:
        add_acc(f"{sp}Acc{tag}Sd", r.acc_sd, src, rc + ", acc_sd")
        add_prop(f"{sp}FOne{tag}Sd", r.f1_sd, src, rc + ", f1_sd")
    if sp == "Test":
        add_int(f"TestN", r.n, src, rc + ", n") if "TestN" not in SEEN else None
    else:
        add_int(f"ValN", r.n, src, rc + ", n") if "ValN" not in SEEN else None

for sp in ("val", "test"):
    src = f"results/compare/{sp}_metrics.csv"
    mt = rd(f"compare/{sp}_metrics.csv")
    for _, r in mt[mt.model.isin(["mlp", "cnn"])].iterrows():
        tag = f"{w(r.model)}{w(r.featureset)}{seedword(r.seed)}"
        rc = f"model={r.model}, featureset={r.featureset}, seed={int(r.seed)}"
        S = sp.capitalize()
        add_acc(f"{S}Acc{tag}", r.accuracy, src, rc + ", accuracy")
        add_acc(f"{S}FOne{tag}", r.macro_f1, src, rc + ", macro_f1")
        add_int(f"{S}Errors{tag}", r.errors, src, rc + ", errors")
        add_prop(f"{S}Acc{tag}Lo", r.acc_lo, src, rc + ", acc_lo (Wilson)")
        add_prop(f"{S}Acc{tag}Hi", r.acc_hi, src, rc + ", acc_hi (Wilson)")

src = "results/compare/test_seed_spread.csv"
ss = rd("compare/test_seed_spread.csv")
for _, r in ss[ss.seeds > 1].iterrows():
    tag = f"{w(r.model)}{w(r.featureset)}"
    rc = f"model={r.model}, featureset={r.featureset}"
    add_int(f"TestErrMin{tag}", r.err_min, src, rc + ", err_min")
    add_int(f"TestErrMax{tag}", r.err_max, src, rc + ", err_max")

# ---------------------------------------------------------------- 6 McNemar
src = "results/compare/test_mcnemar.csv"
mc = rd("compare/test_mcnemar.csv")
for _, r in mc.iterrows():
    a, b = r.model_a, r.model_b
    base = f"McNemar{w(a)}Vs{w(b)}{seedword(r.seed)}"
    rc = f"seed={int(r.seed)}, model_a={a}, model_b={b}"
    add_int(base + "AOnly", r.b_a_right_b_wrong, src, rc + ", b_a_right_b_wrong")
    add_int(base + "BOnly", r.c_a_wrong_b_right, src, rc + ", c_a_wrong_b_right")
    add_int(base + "Discordant", r.disagreements, src, rc + ", disagreements")
    add_int(base + "ErrorsA", r.errors_a, src, rc + ", errors_a")
    add_int(base + "ErrorsB", r.errors_b, src, rc + ", errors_b")
    add_p(base + "P", r.p_exact, src, rc + ", p_exact")
    add_p(base + "PHolm", r.p_holm, src, rc + ", p_holm")
    add_text(base + "Family", r.family, src, rc + ", family")

# ---------------------------------------------------------------- 7 routing
src = "results/routing/test_routes.csv"
rt = rd("routing/test_routes.csv")
for _, r in rt.iterrows():
    base = f"Route{seedword(r.seed)}{w(r.route)}"
    rc = f"seed={int(r.seed)}, route={r.route}"
    add_int(base + "Grains", r.grains, src, rc + ", grains")
    add_prop(base + "Share", r.share, src, rc + ", share")
    if not pd.isna(r.errors):
        add_int(base + "Errors", r.errors, src, rc + ", errors")
    add_acc(base + "Acc", r.accuracy, src, rc + ", accuracy")

# ---------------------------------------------------------------- 8 late lot
src = "results/robustness/late_recall.csv"
lr = rd("robustness/late_recall.csv")
for _, r in lr.iterrows():
    base = f"Late{w(r.model)}{w(r.subset)}"
    rc = f"model={r.model}, subset={r.subset}"
    add_int(base + "N", r.n, src, rc + ", n")
    add_acc(base + "Recall", r.recall, src, rc + ", recall")
    add_prop(base + "Lo", r.ci_low, src, rc + ", ci_low")
    add_prop(base + "Hi", r.ci_high, src, rc + ", ci_high")
    add_prop(base + "MeanConf", r.mean_confidence, src, rc + ", mean_confidence")
    add_int(base + "ConfWrong", r.confidently_wrong, src, rc + ", confidently_wrong")

# ---------------------------------------------------------------- 9 grey-world
src = "results/robustness/greyworld_results.csv"
gw = rd("robustness/greyworld_results.csv")
for _, r in gw.iterrows():
    add_acc(f"Grey{w(r.arm)}{w(r.variant)}", r.accuracy, src, f"arm={r.arm}, variant={r.variant}, accuracy")
add_int("GreyN", gw.n.iloc[0], src, "n")
cnn = gw[gw.arm.str.startswith("cnn")]
for var, nm in (("raw_same_path", "Raw"), ("grey_world", "Grey")):
    add_acc(f"GreyCnn{nm}SeedMean", cnn.loc[cnn.variant == var, "accuracy"].mean(), src,
            f"mean of arm=cnn_seed0..2, variant={var} (derived)")
src = "results/robustness/greyworld_gain_vs_area.csv"
ga = rd("robustness/greyworld_gain_vs_area.csv")
for _, r in ga.iterrows():
    for ch in "RGB":
        add_prop(f"GreyGainArea{w(r['class'])}R{ch}", r[f"r_{ch}"], src, f"class={r['class']}, r_{ch}")
    add_prop(f"GreyGainArea{w(r['class'])}MaxAbs", r.max_abs_r, src, f"class={r['class']}, max_abs_r")
    add_text(f"GreyGainArea{w(r['class'])}Passes", str(r.passes_abs_lt_0_1), src, f"class={r['class']}, passes_abs_lt_0_1")
    add_int(f"GreyGainArea{w(r['class'])}N", r.n, src, f"class={r['class']}, n")

# ---------------------------------------------------------------- 10 disclosures / test reads
add_int("DiscCount", len(rd("DISCLOSURES.csv")), "results/DISCLOSURES.csv", "number of data rows")
add_int("TestReadsCount", len(rd("test_reads.csv")), "results/test_reads.csv", "number of data rows")
add_int("TestReadsDisclosedCount", len(rd("test_reads_disclosed.csv")), "results/test_reads_disclosed.csv", "number of data rows")

# ---------------------------------------------------------------- 11 round-2 additions (report text support)
# Appended section: new macros only; no existing macro is changed.
def _val(name):
    return next(t for n, t, d, s_, r_ in ROWS if n == name)

def add_fix(name, v, nd, src, rc):
    add(name, f"{v:.{nd}f}", f"{v:.{nd}f}", src, rc)

# link check and validity / outlier reasons
src = "results/quality/cleaning_log.csv"
cl = rd("quality/cleaning_log.csv").set_index("check")
m = re.match(r"(\d+) links", cl.loc["link_distance", "scope"])
add_int("LinkRowsRebuilt", m.group(1), src, "check=link_distance, scope")
m = re.search(r"median true ([\d.]+) vs shuffled ([\d.]+)", cl.loc["link_distance", "reason"])
add_text("LinkMedianTrue", m.group(1), src, "check=link_distance, reason (median true)")
add_text("LinkMedianShuffled", m.group(2), src, "check=link_distance, reason (median shuffled)")
m = re.match(r"(\d+) columns flagged \((\d+) cells\)", cl.loc["validity_out_of_range", "reason"])
add_int("CleanValidityColumns", m.group(1), src, "check=validity_out_of_range, reason (columns flagged)")
add_int("CleanValidityCells", m.group(2), src, "check=validity_out_of_range, reason (cells flagged)")
m = re.match(r"(\d+) cells with \|robust z\| > (\d+)", cl.loc["outliers_robust_z6", "reason"])
add_int("CleanOutlierCells", m.group(1), src, "check=outliers_robust_z6, reason (cells)")

# Jasmine late lot vs early Jasmine (median ratios, 2 dp)
src = "results/eda/jasmine_lots.csv"
jl = rd("eda/jasmine_lots.csv").set_index("feature")
for nm, feat in (("Length", "MAJOR_AXIS"), ("Width", "MINOR_AXIS"), ("Area", "AREA"), ("Extent", "EXTENT"), ("Tint", "tint")):
    add_fix(f"JasmineLateRatio{nm}", jl.loc[feat, "ratio"], 2, src, f"feature={feat}, ratio (late/early)")

# best baseline hyperparameters on validation
src = "results/baselines/best_params.json"
bp = json.loads((RES / "baselines/best_params.json").read_text())
for key, prm in bp.items():
    mdl, fs = key.split(":")
    for pn, pv in prm.items():
        nm = f"Param{w(mdl)}{w(fs)}{w(pn)}"
        if isinstance(pv, float) and pv < 1e-3:
            e = int(round(-math.log10(pv)))
            add(nm, rf"\ensuremath{{10^{{-{e}}}}}", f"1e-{e}", src, f"{key}, {pn}")
        elif isinstance(pv, float) and pv != int(pv):
            add(nm, f"{pv:g}", f"{pv:g}", src, f"{key}, {pn}")
        else:
            add_int(nm, pv, src, f"{key}, {pn}")

# neural best config
src = "results/neural/best_config.json"
bc = json.loads((RES / "neural/best_config.json").read_text())
for i, hdn in enumerate(bc["mlp"]["hidden"]):
    add_int(f"NeuralMlpHidden{DIGITS[i + 1]}", hdn, src, f"mlp.hidden[{i}]")
add(f"NeuralMlpLr", f"{bc['mlp']['lr']:g}", f"{bc['mlp']['lr']:g}", src, "mlp.lr")
add("NeuralMlpDropout", f"{bc['mlp']['dropout']:g}", f"{bc['mlp']['dropout']:g}", src, "mlp.dropout")
add_int("NeuralCnnWidth", bc["cnn"]["width"], src, "cnn.width")
add("NeuralCnnLr", f"{bc['cnn']['lr']:g}", f"{bc['cnn']['lr']:g}", src, "cnn.lr")
add("NeuralCnnDropout", f"{bc['cnn']['dropout']:g}", f"{bc['cnn']['dropout']:g}", src, "cnn.dropout")

# shuffled-label control
src = "results/baselines/control.csv"
ct = rd("baselines/control.csv")
add_int("ControlBaselineRows", len(ct), src, "number of data rows")
add_acc("ControlBaselineMaxValAcc", ct.val_accuracy.max(), src, "max val_accuracy")
src = "results/neural/control.csv"
nc = rd("neural/control.csv").set_index("model")
add_acc("ControlMlpValAcc", nc.loc["mlp", "val_accuracy"], src, "model=mlp, val_accuracy")
add_acc("ControlCnnValAcc", nc.loc["cnn", "val_accuracy"], src, "model=cnn, val_accuracy")

# clustering: PCA variance and late-lot cluster at k=6
src = "results/cluster/pca_variance.csv"
pv = rd("cluster/pca_variance.csv").set_index("featureset")
add_acc("ClusterPcaCumVarAllOneZeroSix", pv.loc["all106", "cumulative_explained_variance"], src, "featureset=all106, cumulative_explained_variance")
src = "results/cluster/late_lot_check.csv"
lc = rd("cluster/late_lot_check.csv")
lc = lc[(lc.featureset == "all106") & (lc.method == "kmeans") & lc.passes]
add_acc("ClusterLateKmeansAllOneZeroSixShare", lc.share_of_late.iloc[0], src, "all106, kmeans, passing cluster, share_of_late")
add_acc("ClusterLateKmeansAllOneZeroSixPurity", lc.purity_late.iloc[0], src, "all106, kmeans, passing cluster, purity_late")

# test errors for HGB and tree4, and errors per ten thousand test grains
src = "results/compare/test_metrics.csv"
tm = rd("compare/test_metrics.csv")
def _err(model, fs):
    return int(tm[(tm.model == model) & (tm.featureset == fs) & (tm.split == "test")].errors.iloc[0])
add_int("TestErrorsHgbAllOneZeroSix", _err("hgb", "all106"), src, "model=hgb, featureset=all106, errors")
add_int("TestErrorsTreeFourAllOneZeroSix", _err("tree4", "all106"), src, "model=tree4, featureset=all106, errors")
tn = int(_val("TestN"))
for nm, e in (("Hgb", _err("hgb", "all106")), ("TreeFour", _err("tree4", "all106")),
              ("MlpMin", int(_val("TestErrMinMlpAllOneZeroSix"))), ("MlpMax", int(_val("TestErrMaxMlpAllOneZeroSix"))),
              ("CnnMin", int(_val("TestErrMinCnnImage"))), ("CnnMax", int(_val("TestErrMaxCnnImage")))):
    add_fix(f"ErrPerTenThousand{nm}", 10000 * e / tn, 1, src, f"errors / TestN x 10000 (derived), {nm}")

# ---------------------------------------------------------------- write
OUT.mkdir(exist_ok=True)
tex = ["% Generated by report/make_numbers.py from results/. Do not edit by hand."]
for name, t, d, s, rc in ROWS:
    tex.append(f"% {s}: {rc}")
    tex.append(f"\\newcommand{{\\{name}}}{{{t}}}")
(OUT / "numbers.tex").write_text("\n".join(tex) + "\n")

md = ["Generated by report/make_numbers.py from results/. Do not edit by hand.", "",
      "| macro | value | source file | row/column |", "|---|---|---|---|"]
for name, t, d, s, rc in ROWS:
    cell = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    md.append(f"| `\\{name}` | {cell(d)} | {s} | {cell(rc)} |")
(OUT / "FACTS.md").write_text("\n".join(md) + "\n")
print(f"{len(ROWS)} macros written")
