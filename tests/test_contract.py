"""The contract between stages. Reads code and CSV files only, fits no model.

Each check builds a set of finding lines and compares it with tests/contract_known.txt: a finding not
listed fails, and a listed line that is no longer found fails too. The reader scan follows only frames
bound by name inside one file: a frame that arrives as a function argument under another name is not seen.
"""
import ast, csv, re
from pathlib import Path
import pandas as pd
import pytest
import common as C
from conftest import ROOT

FILES = C.ARTEFACTS["files"]
KNOWN = Path(__file__).with_name("contract_known.txt")
GRAINS, SPLITS = "data/derived/grains.parquet", "data/splits.csv"
PREDS = next(k for k in FILES if k.startswith("results/preds/"))
TABLES = [k for k in FILES if k.endswith(".csv")]
FRAME_ATTRS = set(dir(pd.DataFrame)) | set(dir(pd.Series))    # x[c] with a variable c may be a column
KEEP = {"copy", "reset_index", "sort_values", "dropna", "query", "drop_duplicates", "head", "tail", "sample", "assign"}
COL_ARGS = {"groupby", "sort_values", "drop_duplicates", "merge"}
NO_RESULTS = "results/ is gitignored and absent from this worktree: the integrator runs this check on main"


def stage_files():
    return [p for p in sorted((ROOT / "src").glob("*.py")) if p.name != "common.py"] + [ROOT / "tools" / "freeze.py"]


def known(*kinds, path=KNOWN):
    lines = [l.split(" # ")[0].strip() for l in path.read_text().splitlines()]
    return {l for l in lines if l and not l.startswith("#") and l.split()[0] in kinds}


def settle(found, listed):
    new, stale = sorted(found - listed), sorted(listed - found)
    assert not new and not stale, f"new mismatches: {new}; listed but no longer true: {stale}"


def expand(rel):
    return [rel.format(sp=s) for s in ("val", "test")] if "{sp}" in rel else [rel]


# ---- reading the code ---------------------------------------------------------------
def literals(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, str): return [n.value]
    if isinstance(n, (ast.List, ast.Tuple)): return [s for e in n.elts for s in literals(e)]
    return []


def frame_of(n, bound):
    """Artefact behind an expression, if it is still that frame with all its columns."""
    if isinstance(n, ast.Name): return bound.get(n.id)
    if isinstance(n, ast.Subscript): return None if literals(n.slice) else frame_of(n.value, bound)
    if isinstance(n, ast.Attribute) and n.attr in ("loc", "iloc"): return frame_of(n.value, bound)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in KEEP: return frame_of(n.func.value, bound)
    return None


def source_of(value, last):
    """Artefact a right-hand side loads, if any."""
    text = ast.unparse(value)
    if re.search(r"\bC\.load_(grains|split)\(", text) or ("read_parquet(" in text and "grains.parquet" in text): return GRAINS
    if "load_preds(" in text: return PREDS
    m = re.search(r"\bread(?:_csv)?\(([^()]*)\)", text)
    if not m: return None
    if m[1] == "C.SPLITS": return SPLITS
    arg = ast.unparse(last[m[1]]) if m[1] in last else m[1]
    for rel in TABLES:
        if re.search(re.escape(rel.rsplit("/", 1)[1]).replace(r"\{sp\}", r"\{\w+\}") + "['\"]", arg): return rel
    return None


def pairs(n):
    """(target name, value) for a plain or tuple assignment; an unpacked call binds its names to nothing."""
    t, v = n.targets[0], n.value
    if isinstance(t, ast.Name): return [(t.id, v)]
    if isinstance(t, ast.Tuple):
        vals = v.elts if isinstance(v, ast.Tuple) and len(t.elts) == len(v.elts) else [ast.Constant(None)] * len(t.elts)
        return [(a.id, b) for a, b in zip(t.elts, vals) if isinstance(a, ast.Name)]
    return []


def note_asks(n, bound, asked, made):
    if isinstance(n, ast.Subscript) and (f := frame_of(n.value, bound)):
        (made if isinstance(n.ctx, ast.Store) else asked).update((f, c) for c in literals(n.slice))
    elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and n.attr not in FRAME_ATTRS and (f := frame_of(n.value, bound)):
        asked.add((f, n.attr))
    elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
        base = n.func.value
        if (f := frame_of(base, bound)):
            if n.func.attr in COL_ARGS:
                asked.update((f, c) for a in n.args + [k.value for k in n.keywords if k.arg in ("by", "on", "subset")] for c in literals(a))
            if n.func.attr == "assign":
                made.update((f, k.arg) for k in n.keywords)
        if n.func.attr == "agg" and isinstance(base, ast.Call) and isinstance(base.func, ast.Attribute) and base.func.attr == "groupby" and (f := frame_of(base.func.value, bound)):
            asked.update((f, literals(k.value)[0]) for k in n.keywords if literals(k.value))


def walk_scope(nodes, outer, loaded, asked, made):
    """Follow one scope in line order, so a name rebound to something else stops counting as the frame."""
    bound, last = dict(outer), {}
    def when(x): return (x.end_lineno, 1) if isinstance(x, ast.Assign) else (x.lineno, 0)
    for n in sorted((x for x in nodes if hasattr(x, "lineno")), key=when):
        if isinstance(n, ast.For):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name): bound.pop(t.id, None)
        if not isinstance(n, ast.Assign):
            note_asks(n, bound, asked, made)
            continue
        for name, v in pairs(n):
            last[name] = v
            src = source_of(v, last)
            art = src or frame_of(v, bound)
            if src: loaded.add(src)
            if art: bound[name] = art
            else: bound.pop(name, None)
    return bound


def scan(text):
    """(artefacts loaded, (artefact, column) pairs asked for) in one file's source."""
    tree = ast.parse(text)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    inside = {id(m) for f in funcs for m in ast.walk(f)}
    loaded, asked, made = set(), set(), set()
    outer = walk_scope([n for n in ast.walk(tree) if id(n) not in inside], {}, loaded, asked, made)
    for f in funcs:
        walk_scope(list(ast.walk(f)), outer, loaded, asked, made)
    return loaded, asked - made


def reader_findings(files, table=FILES):
    out = set()
    for p in files:
        loaded, asked = scan(Path(p).read_text())
        for art in loaded:
            if p.name not in table[art]["readers"] + table[art]["writers"]:
                out.add(f"reader {art} {p.name}")
        out |= {f"column {art} {p.name} {c}" for art, c in asked if c not in table[art]["columns"]}
    return out


def writer_findings(table=FILES, root=ROOT):
    """Every listed column of a table is named in the source of one of its writers."""
    out = set()
    for rel, e in table.items():
        if not rel.endswith(".csv"): continue
        src = "\n".join((root / d / w).read_text() for w in e["writers"] for d in ("src", "tools") if (root / d / w).exists())
        out |= {f"writer {rel} {c}" for c in e["columns"] if not re.search(rf"\b{re.escape(c)}\b", src)}
    return out


def alias_findings(table=FILES):
    bad = {a: name for name, aliases in C.ARTEFACTS["names"]["columns"].items() for a in aliases}
    return {f"alias {rel} {c} {bad[c]}" for rel, e in table.items() for c in e["columns"] if c in bad}


def function_findings(files):
    def tops(p): return {n.name for n in ast.parse(Path(p).read_text()).body if isinstance(n, ast.FunctionDef)}
    shared = tops(ROOT / "src" / "common.py")
    return {f"function {p.name} {f}" for p in files for f in sorted(tops(p) & shared)}


def csv_findings(path, rel, want):
    """Header against the entry; for files under results/ every row must also parse to the header's width."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = next(r, [])
        out = set() if head == want else {f"header {rel} {','.join(head)}"}
        if rel.startswith("results/") and any(len(row) != len(head) for row in r):
            out.add(f"rows {rel}")
    return out


# ---- the three checks ---------------------------------------------------------------
def test_a_readers_ask_only_for_written_columns():
    settle(reader_findings(stage_files()) | writer_findings(), known("reader", "column", "writer"))


def test_a_one_name_per_column():
    settle(alias_findings(), known("alias"))


@pytest.mark.parametrize("rel", [r for k in TABLES for r in expand(k)])
def test_b_csv_header_matches_entry(rel):
    if not (ROOT / "results").is_dir():
        pytest.xfail(NO_RESULTS)
    if not (ROOT / rel).exists():
        pytest.xfail(f"{rel} is not built yet")
    key = next(k for k in TABLES if rel in expand(k))
    listed = {l for l in known("header", "rows") if l.split()[1] == rel}
    settle(csv_findings(ROOT / rel, rel, FILES[key]["columns"]), listed)


def test_c_no_stage_redefines_common():
    settle(function_findings(stage_files()), known("function"))


def test_names_match_the_code():
    n = C.ARTEFACTS["names"]
    assert n["classes"] == C.CLASSES
    for fs in n["feature_sets"][:-1]:
        assert C.features(fs)
    with pytest.raises(KeyError):
        C.features("nope")


# ---- the checks themselves, on made-up code and files ---------------------------------
def test_settle_passes_listed_and_fails_new_or_stale():
    settle({"column a b c"}, {"column a b c"})
    with pytest.raises(AssertionError, match="new mismatches"):
        settle({"column a b c", "column a b d"}, {"column a b c"})
    with pytest.raises(AssertionError, match="no longer true"):
        settle(set(), {"column a b c"})


def test_known_list_parsing(tmp_path):
    p = tmp_path / "k.txt"
    p.write_text("# heading\n\nalias f cls class # src/x.py:1, Evaluation\nrows g\n")
    assert known("alias", path=p) == {"alias f cls class"} and known("rows", path=p) == {"rows g"}


def test_scan_finds_a_column_the_writer_lacks(tmp_path):
    src = ("import common as C, pandas as pd\n"
           "M = read(cmp / f'{sp}_metrics.csv')\n"
           "p = q / 'cleaning_log.csv'\n"
           "old = pd.read_csv(p) if p.exists() else None\n"
           "g = C.load_grains()\n"
           "J = g[g['class'] == 'Jasmine'].sort_values('file_no')\n"
           "J['tint'] = 1\n"
           "print(J.tint, J.nope, g.iloc[0], M.groupby(['model', 'missing']).agg(n=('n', 'first'), z=('zzz', 'mean')), old.check, old.bad)\n")
    _, asked = scan(src)
    m, cl = "results/compare/{sp}_metrics.csv", "results/quality/cleaning_log.csv"
    assert {(GRAINS, "class"), (GRAINS, "file_no"), (GRAINS, "nope"), (m, "missing"), (m, "zzz"), (cl, "bad")} <= asked
    assert (GRAINS, "tint") not in asked and (GRAINS, "iloc") not in asked
    for rebind in ("g = g.groupby('y').size()", "for _, g in x.items(): pass"):     # no longer the frame
        assert (GRAINS, "nope") not in scan(f"import common as C\ng = C.load_grains()\n{rebind}\nprint(g.nope)\n")[1]
    p = tmp_path / "99_x.py"
    p.write_text(src)
    assert "column data/derived/grains.parquet 99_x.py nope" in reader_findings([p])
    assert f"reader {GRAINS} 99_x.py" in reader_findings([p])


def test_writer_check_finds_a_column_the_code_never_names(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "w.py").write_text("d = {'a': 1}\n")
    t = {"x.csv": {"writers": ["w.py"], "readers": [], "columns": ["a", "b"]}}
    assert writer_findings(t, tmp_path) == {"writer x.csv b"}


def test_alias_check_flags_a_banned_name():
    assert alias_findings({"x.csv": {"columns": ["cls", "n"]}}) == {"alias x.csv cls class"}


def test_function_check_ignores_methods(tmp_path):
    p = tmp_path / "s.py"
    p.write_text("def wilson():\n    pass\nclass K:\n    def ece(self):\n        pass\n")
    assert function_findings([p]) == {"function s.py wilson"}


def test_header_check_on_a_made_up_file(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("a,b\n1,2\n")
    assert csv_findings(p, "results/h.csv", ["a", "b"]) == set()
    assert csv_findings(p, "results/h.csv", ["a", "c"]) == {"header results/h.csv a,b"}
    p.write_text('t,config\n1,"{"a": 1, "b": 2}"\n')          # the unescaped quotes of epoch_times.csv
    assert csv_findings(p, "results/h.csv", ["t", "config"]) == {"rows results/h.csv"}
    assert csv_findings(p, "data/h.csv", ["t", "config"]) == set()    # outside results/ the rows stay unread
