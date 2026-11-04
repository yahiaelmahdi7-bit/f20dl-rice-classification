"""Print paths of N real smoke-slice images, one per variety, for the predict.py smoke check."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C  # noqa: E402

g = C.load_split("val")
print(*[g[g.y == k].path.iloc[0] for k in range(int(sys.argv[1]))])
