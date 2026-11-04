"""Fail if any generated file's prompt record is missing or differs from the prompt files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stamp_prompts import ROOT, block, split, targets  # noqa: E402


def main():
    bad = 0
    for p in targets():
        head, _ = split(p.read_text())
        if head != block(p):
            print("MISMATCH" if head else "MISSING", p.relative_to(ROOT))
            bad += 1
    print(f"{len(targets())} files checked, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
