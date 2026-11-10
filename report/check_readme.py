# Generated with AI assistance (Claude Sonnet, 2026-11-10).
# Prompt: "Write report/check_readme.py: extract shell lines from README.md code blocks and check each make target exists in the Makefile and each script/path exists; exit 1 on failure."
"""Check the shell commands in README.md: make targets exist, referenced paths exist.
Paths under data/raw/ and data/derived/ are skipped: they are git-ignored and created by `make get-data` / `make data`."""
import re, shlex, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = {"", "bash", "sh", "console"}
PREFIXES = ("src/", "tools/", "report/", "data/", "results/")
IGNORED = ("data/raw/", "data/derived/")


def code_lines(text):
    inside, lang = False, ""
    for ln in text.splitlines():
        if ln.startswith("```"):
            inside, lang = (not inside), ln[3:].strip()
            continue
        if inside and lang in LANGS and ln.strip():
            yield ln.strip().removeprefix("$ ").strip()


def targets(makefile):
    return set(re.findall(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", makefile, re.M))


def check(line, tgts):
    errs = []
    try:
        toks = shlex.split(line)
    except ValueError as e:
        return [f"cannot parse: {e}"]
    while toks and (re.match(r"^[A-Za-z_]\w*=", toks[0]) or toks[0] == "env"):
        toks = toks[1:]
    if toks and toks[0] == "make":
        for t in toks[1:]:
            if re.match(r"^\w+=", t) or t.startswith("-"):
                continue
            if t not in tgts:
                errs.append(f"make target not in Makefile: {t}")
    for t in toks:
        if "<" in t and ">" in t:
            continue
        if t.endswith((".py", ".sh")) or t.startswith(PREFIXES):
            if t.startswith(IGNORED):
                continue
            if not (ROOT / t).exists():
                errs.append(f"path not found: {t}")
    return errs


def main():
    tgts = targets((ROOT / "Makefile").read_text())
    lines = list(code_lines((ROOT / "README.md").read_text()))
    bad = 0
    for ln in lines:
        errs = check(ln, tgts)
        print("FAIL" if errs else "PASS", ln)
        for e in errs:
            print("    ", e)
        bad += bool(errs)
    print(f"{len(lines) - bad}/{len(lines)} lines pass")
    return 1 if bad or not lines else 0


if __name__ == "__main__":
    sys.exit(main())
