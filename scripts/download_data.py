#!/usr/bin/env python3
"""
Download the data for this repository from Zenodo (DOI 10.5281/zenodo.17650136)
and unpack it where the notebooks expect it.

    python scripts/download_data.py analysis      # data/ and input_data/  (~12.9 GB download)
    python scripts/download_data.py cfd           # data/cfd/              (~3.7 GB, needs `pip install py7zr`)
    python scripts/download_data.py all

Archives are cached in .zenodo_cache/ and checked against their MD5 sums.
Only the data folders are extracted; the code in the archives is the same as in this repository.
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/records/17650136/files"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".zenodo_cache"

ARCHIVES = {
    "analysis": {
        "file": "Analysis_transition.zip",
        "md5": "6cb19bbfd6bd9979553dad2bf60d92a4",
        # archive prefix -> destination in this repo
        "extract": {"Analysis_transition/data/": ROOT / "data",
                    "Analysis_transition/input_data/": ROOT / "input_data"},
    },
    "cfd": {
        "file": "CFD_simulations.7z",
        "md5": "500055339353fc231166f81f87656440",
        "extract": {"CFD_simulations/data/": ROOT / "data" / "cfd"},
    },
}


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(name):
    spec = ARCHIVES[name]
    CACHE.mkdir(exist_ok=True)
    dest = CACHE / spec["file"]
    if dest.exists() and md5sum(dest) == spec["md5"]:
        print(f"✓ {spec['file']} already downloaded")
        return dest
    url = f"{RECORD}/{spec['file']}?download=1"
    print(f"↓ {url}")

    def progress(blocks, block_size, total):
        done = blocks * block_size
        if total > 0:
            sys.stdout.write(f"\r  {done / 1e9:6.2f} / {total / 1e9:.2f} GB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()
    if md5sum(dest) != spec["md5"]:
        raise RuntimeError(f"MD5 mismatch for {dest}; delete it and retry.")
    return dest


def extract(name, archive):
    mapping = ARCHIVES[name]["extract"]

    def target_for(member):
        if "__MACOSX" in member or member.endswith(".DS_Store"):
            return None
        for prefix, out in mapping.items():
            if member.startswith(prefix) and len(member) > len(prefix):
                return out / member[len(prefix):]
        return None

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                target = target_for(info.filename)
                if target is None or info.is_dir():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    else:
        try:
            import py7zr
        except ImportError:
            sys.exit("The CFD archive is .7z: run `pip install py7zr` first.")
        with py7zr.SevenZipFile(archive) as z:
            names = [n for n in z.getnames() if target_for(n)]
            tmp = CACHE / "cfd_tmp"
            z.extract(path=tmp, targets=names)
            for n in names:
                src = tmp / n
                if src.is_file():
                    target = target_for(n)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), target)
            shutil.rmtree(tmp, ignore_errors=True)
    for out in mapping.values():
        print(f"✓ extracted → {out.relative_to(ROOT)}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["analysis", "cfd", "all"])
    args = parser.parse_args()
    for name in (["analysis", "cfd"] if args.what == "all" else [args.what]):
        extract(name, download(name))


if __name__ == "__main__":
    main()
