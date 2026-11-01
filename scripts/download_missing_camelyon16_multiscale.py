"""Download only missing DSMIL CAMELYON16 concatenated multiscale files.

The file IDs come from the author-provided Google Drive folder listing saved in
dataset_audits/camelyon16_multiscale_source_inventory.json. The `fuse` branch is
intentionally excluded because the planned ablation uses separable `cat` features.
"""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "dataset_audits/camelyon16_multiscale_source_inventory.json"
DESTINATION = REPO / "datasets/Camelyon16-multiscale"


def entries():
    inventory = json.loads(INVENTORY.read_text())
    result = []
    for item in inventory["source_files"]:
        relative = Path(item["path"])
        if relative.parts[0] != "cat":
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe path in inventory: {relative}")
        result.append((item["id"], relative, DESTINATION / relative))
    return result


def valid_csv(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle))
        expected_columns = 1024 if path.parent.name in ("0-normal", "1-tumor") else 2
        return len(header) == expected_columns
    except (OSError, StopIteration, UnicodeError, csv.Error):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing files; default is a dry run")
    parser.add_argument("--list", action="store_true", help="Print every missing relative path")
    parser.add_argument("--limit", type=int, help="Download at most this many missing files in one run")
    parser.add_argument("--min-free-gib", type=float, default=2.0,
                        help="Stop before a transfer if free space is below this amount [2]")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    source = entries()
    missing = [(file_id, rel, path) for file_id, rel, path in source if not valid_csv(path)]
    counts = Counter("/".join(rel.parts[:3]) if len(rel.parts) == 4 else "manifests"
                     for _, rel, _ in missing)
    print(f"Concatenated source files: {len(source)}; missing or invalid: {len(missing)}")
    for group, count in sorted(counts.items()):
        print(f"  {group}: {count}")
    if args.list:
        for _, rel, _ in missing:
            print(rel)
    if not args.download:
        print("Dry run only. Add --download to fetch these files.")
        return

    try:
        import gdown
    except ImportError as error:
        raise SystemExit("Install gdown with: python3 -m pip install gdown") from error

    selected = missing[:args.limit] if args.limit else missing
    for index, (file_id, rel, path) in enumerate(selected, 1):
        free = shutil.disk_usage(DESTINATION).free / (1024 ** 3)
        if free < args.min_free_gib:
            raise SystemExit(f"Stopping with {free:.1f} GiB free; upload or free space, then rerun.")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(selected)}] {rel}", flush=True)
        result = gdown.download(id=file_id, output=str(path), quiet=False,
                                use_cookies=False, resume=True, retries=3)
        if result is None or not valid_csv(path):
            raise SystemExit(f"Download failed or CSV header is invalid: {rel}. Rerun to resume.")
    print("Selected transfers finished. Rerun without --download to check remaining files.")


if __name__ == "__main__":
    main()
