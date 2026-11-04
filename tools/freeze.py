"""Hash every configuration, seed, floor, familiarity cut and test pair after all validation work."""
import hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C  # noqa: E402

PARTS = [C.RES / "baselines" / "best_params.json", C.RES / "neural" / "best_config.json", C.MODELS / "routing.json", C.SPLITS]


def main():
    missing = [str(p) for p in PARTS if not p.exists()]
    if missing:
        print("cannot freeze, missing:", *missing, sep="\n  ")
        return 1
    parts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in PARTS}
    body = {"seeds": list(C.SEEDS), "pairs": C.PAIRS, "parts": parts}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    out = (C.RES / "frozen.json") if C.SMOKE else C.FROZEN
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"hash": h, **body}, indent=1))
    print("frozen", h, "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
