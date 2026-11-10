# Generated with AI assistance (Claude Sonnet, 2026-11-10).
# Prompt: "Write report/build.sh (tectonic build of main.tex to F20DL-Group16-report.pdf) and report/check_report.py (page limit via bodyend label, no hand-typed result numbers, every DISCLOSURES.csv item present, no undefined refs, figures exist, forbidden-claim grep)."
"""Checks for report/main.tex. Stdlib only; exit 1 on any failure."""
import csv, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
fails = []

def result(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (": " + detail if detail else ""))
    if not ok:
        fails.append(name)

def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()

tex = read(os.path.join(HERE, "main.tex"))

# (a) body ends on page <= 6
aux = read(os.path.join(HERE, "main.aux")) if os.path.exists(os.path.join(HERE, "main.aux")) else ""
m = re.search(r"\\newlabel\{bodyend\}\{\{[^}]*\}\{(\d+)\}", aux)
page = int(m.group(1)) if m else None
# floats are flushed at the \\clearpage before the appendix and can land after the label's page,
# so the body ends no earlier than the last body figure/table label
fl = [int(x) for x in re.findall(r"\\newlabel\{(?:fig|tab):[^}]*\}\{\{[^}]*\}\{(\d+)\}", aux)]
end = max([page] + fl) if page is not None else None
result("(a) bodyend page <= 6", end is not None and end <= 6, "bodyend label on page %s, last body float on page %s, body ends page %s" % (page, max(fl) if fl else None, end))

# (b) no hand-typed result numbers
def strip(src):
    out = []
    for line in src.split("\n"):
        line = re.sub(r"(?<!\\)%.*", "", line)
        if re.match(r"\s*\\input\b", line) or re.match(r"\s*\\bibliography", line):
            line = ""
        out.append(line)
    t = "\n".join(out)
    t = re.sub(r"\\(cite|ref|label|includegraphics|graphicspath|path|url)(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?", r"\\\1", t)
    t = re.sub(r"\\begin\{(tabular|tabularx)\}(\{[^}]*\})?(\{(?:[^{}]|\{[^}]*\})*\})", "", t)
    return t
clean = strip(tex)
pats = [r"\d\s*\\?%", r"\\?%\s*\d", r"0\.\d", r"p\s*[=<]\s*\d", r"\d+\.\d+"]
hits = []
for i, line in enumerate(clean.split("\n"), 1):
    if any(re.search(p, line) for p in pats):
        hits.append((i, line.strip()))
for i, l in hits:
    print("   hit line %d: %s" % (i, l[:100]))
result("(b) no hand-typed result numbers", not hits, "%d hits" % len(hits))

# (c) disclosure items present verbatim
with open(os.path.join(ROOT, "results", "DISCLOSURES.csv"), newline="", encoding="utf-8") as f:
    items = [r["item"] for r in csv.DictReader(f)]
missing = [i for i in items if i not in tex]
result("(c) all DISCLOSURES.csv items present", not missing, "%d items, missing: %s" % (len(items), missing))

# (d) build.log clean
log = read(os.path.join(HERE, "build.log")) if os.path.exists(os.path.join(HERE, "build.log")) else None
bad = []
if log is not None:
    for l in log.split("\n"):
        if re.search(r"undefined", l, re.I) or re.search(r"Citation|Reference", l) and re.search(r"undefined", l, re.I):
            bad.append(l.strip())
result("(d) build.log has no undefined refs/citations", log is not None and not bad, "; ".join(bad[:3]) if bad else ("no build.log" if log is None else ""))

# (e) figures exist
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
nofig = [g for g in figs if not os.path.exists(os.path.join(ROOT, "figures", g))]
result("(e) every includegraphics exists in figures/", bool(figs) and not nofig, "%d figures, missing: %s" % (len(figs), nofig))

# (f) forbidden claims
prose = re.sub(r"(?<!\\)%.*", "", tex)
fb = []
for ph in ["significantly better", "statistically superior", "contamination-free", "no leakage"]:
    if re.search(re.escape(ph), prose, re.I):
        fb.append(ph)
if re.search(r"(causes|caused by).{0,60}(colou?r|CNN)|(colou?r|CNN).{0,60}(causes|caused by)", prose, re.I | re.S):
    fb.append("causes/caused by near colour/CNN")
result("(f) no forbidden claims", not fb, ", ".join(fb))

sys.exit(1 if fails else 0)
