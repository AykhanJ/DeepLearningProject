from __future__ import annotations
import argparse
from pathlib import Path
import requests

from src.utils import ensure_dir

REPO_RAW = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master"

FILES = [
    "KDDTrain%2B.txt",  # URL-encoded '+' sign
    "KDDTest%2B.txt",
]

def download(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    for fn in FILES:
        url = f"{REPO_RAW}/{fn}"
        local_name = fn.replace("%2B", "+")
        out_path = out_dir / local_name
        if out_path.exists():
            print(f"Exists, skipping: {out_path}")
            continue
        print(f"Downloading {url} -> {out_path}")
        download(url, out_path)

    print("Done.")

if __name__ == "__main__":
    main()
