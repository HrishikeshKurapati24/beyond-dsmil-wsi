"""Build a no-copy view of a partial multiscale Kaggle input plus a supplement."""

from __future__ import annotations

import csv
from pathlib import Path


def overlay_multiscale(primary: Path, input_base: Path, reference: Path, overlay: Path) -> tuple[Path, list[str]]:
    with reference.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {r["slide_id"]: int(r["label"]) for r in rows}
    if len(rows) != 399 or len(expected) != 399:
        raise ValueError("Expected 399 unique reference slide IDs")

    original = {}
    for folder in ("0-normal", "1-tumor"):
        for path in (primary / folder).glob("*.csv"):
            if path.stem in original:
                raise ValueError(f"Duplicate primary slide: {path.stem}")
            original[path.stem] = path
    if set(original) - set(expected):
        raise ValueError(f"Unexpected primary slides: {sorted(set(original)-set(expected))}")
    missing = sorted(set(expected) - set(original))
    if not missing:
        return primary, []

    supplemental = {}
    for path in input_base.rglob("*.csv"):
        if path.stem not in missing:
            continue
        if path.is_relative_to(primary):
            continue
        if path.stem in supplemental:
            raise ValueError(f"Duplicate supplemental slide: {path.stem}")
        supplemental[path.stem] = path
    if set(supplemental) != set(missing):
        raise ValueError(f"Supplement incomplete; missing slide IDs: {sorted(set(missing)-set(supplemental))}")

    for sid, path in supplemental.items():
        expected_folder = "1-tumor" if expected[sid] else "0-normal"
        if (path.parent.name in ("0-normal", "1-tumor") and
                path.parent.name != expected_folder) or path.stat().st_size == 0:
            raise ValueError(f"Wrong class folder or empty supplement: {path}")
        with path.open(newline="") as handle:
            if len(next(csv.reader(handle))) != 1024:
                raise ValueError(f"Wrong feature width in supplement: {path}")

    combined = {**original, **supplemental}
    for sid, path in combined.items():
        target = overlay / ("1-tumor" if expected[sid] else "0-normal") / f"{sid}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != path.resolve():
                raise ValueError(f"Stale overlay link: {target}")
        elif target.exists():
            raise ValueError(f"Overlay path already occupied: {target}")
        else:
            target.symlink_to(path.resolve())
    return overlay, missing
