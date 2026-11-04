"""Stage 0: find or fetch the two Kaggle zips, verify checksums, unpack, make raw read-only."""
import hashlib, os, sys, urllib.request, zipfile
from pathlib import Path
from common import RAW, ROOT

SLUGS = {"rice-msc.zip": "rice-msc-dataset", "rice-image.zip": "rice-image-dataset"}
DIRS = {"rice-msc.zip": "Rice_MSC_Dataset", "rice-image.zip": "Rice_Image_Dataset"}
URL = "https://www.kaggle.com/api/v1/datasets/download/muratkokludataset/{}"
MANUAL = """Manual download:
  1. open https://www.kaggle.com/datasets/muratkokludataset/rice-msc-dataset and rice-image-dataset
  2. download both zips, name them rice-msc.zip and rice-image.zip
  3. put them in data/raw/ (or point RICE_ZIP_DIR at their folder) and rerun: make data"""


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def expected():
    return {l.split()[1]: l.split()[0] for l in (ROOT / "data" / "SHA256SUMS").read_text().splitlines() if l.strip()}


def download(name, dest):
    url = URL.format(SLUGS[name])
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "rice-study"})) as r, open(dest, "wb") as f:
        total, got = int(r.headers.get("Content-Length") or 0), 0
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b); got += len(b)
            print(f"\r  {name}: {got/1e6:.0f} MB" + (f" / {total/1e6:.0f} MB" if total else ""), end="", flush=True)
    print()


def main():
    exp, RAW_ZIPS = expected(), Path(os.environ.get("RICE_ZIP_DIR", RAW))
    RAW.mkdir(parents=True, exist_ok=True)
    for name, want in exp.items():
        z = RAW_ZIPS / name
        if not z.exists() and (RAW / name).exists(): z = RAW / name
        if not z.exists():
            try:
                z = RAW / name
                download(name, z)
            except Exception as e:
                print(f"download failed: {e}\n{MANUAL}"); return 1
        got = sha(z)
        if got != want:
            print(f"checksum mismatch for {z}: {got} != {want}\n{MANUAL}"); return 1
        print(f"verified {z.name} sha256 {got[:16]}...", flush=True)
        d = RAW / DIRS[name]
        if d.exists() and any(d.iterdir()):
            print(f"already unpacked: {d}, no download, no unpack")
        else:
            with zipfile.ZipFile(z) as zf: zf.extractall(RAW)
            print(f"unpacked {d}")
    for name in DIRS.values():                    # read-only: remove write bits only
        os.system(f"chmod -R a-w '{RAW / name}'")
    print("raw data verified and read-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
