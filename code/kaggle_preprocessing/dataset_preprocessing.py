# Designated code author: Hrishikesh Kurapati
# Scope/provenance: CAMELYON16 feature preparation; repository commit records Hrishikesh as author.
# Attribution header added to this review copy; original source remains unchanged.

"""Prepare one CAMELYON16 feature source per Kaggle notebook run.

Input datasets are read from /kaggle/input. Outputs are written to
/kaggle/working and must be saved as a Kaggle notebook version/dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


SOURCE_SLUGS = {
    "uniscale": "camelyon16-precomputedfeatures-uniscale",
    "multiscale": "camelyon16-precomputedfeatures-multiscale",
    "uni": "camelyon16-uni-uniscale-preprocessed-features",
}
SOURCE_DIMS = {"uniscale": 512, "multiscale": 1024, "uni": 1024}
ROOT_SUFFIXES = {
    "uniscale": ["Camelyon16", ""],
    "multiscale": [
        "Camelyon16-multiscale/cat/Camelyon16-20x_v2-5x",
        "cat/Camelyon16-20x_v2-5x",
        "Camelyon16-20x_v2-5x",
        "",
    ],
    "uni": ["Camelyon16-UNI", ""],
}
KNOWN_LABEL_CONFLICTS = {"uniscale": {"test_114": (0, 1)}}


def read_reference(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 399 or len({r["slide_id"] for r in rows}) != 399:
        raise ValueError("Reference must have exactly 399 unique slide IDs")
    for row in rows:
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1):
            raise ValueError(f"Invalid label: {row}")
        expected_split = "test" if row["slide_id"].startswith("test_") else "train"
        if row["split"] != expected_split:
            raise ValueError(f"Invalid official split: {row}")
        for source in SOURCE_SLUGS:
            row[f"{source}_patches"] = int(row[f"{source}_patches"])
    if sum(r["split"] == "train" for r in rows) != 270:
        raise ValueError("Expected 270 official training slides")
    if sum(r["split"] == "test" for r in rows) != 129:
        raise ValueError("Expected 129 held-out test slides")
    if next(r["label"] for r in rows if r["slide_id"] == "test_114") != 0:
        raise ValueError("test_114 must use the user-specified normal label")
    return sorted(rows, key=lambda row: row["slide_id"])


def folds_for(rows: list[dict], seed: int) -> dict[str, int | None]:
    train = [r for r in rows if r["split"] == "train"]
    labels = [r["label"] for r in train]
    result: dict[str, int | None] = {r["slide_id"]: None for r in rows}
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, valid_indices) in enumerate(splitter.split(np.zeros(len(train)), labels)):
        for index in valid_indices:
            result[train[index]["slide_id"]] = fold
    assert all(result[r["slide_id"]] is not None for r in train)
    assert all(result[r["slide_id"]] is None for r in rows if r["split"] == "test")
    return result


def resolve_root(source: str, base: Path) -> Path:
    for suffix in ROOT_SUFFIXES[source]:
        root = base / suffix
        marker = root / ("embeddings" if source == "uni" else "0-normal")
        if marker.is_dir():
            return root
    raise FileNotFoundError(f"Cannot find {source} feature layout under {base}")


def inventory(source: str, root: Path, rows: list[dict]) -> tuple[dict, list[dict]]:
    expected_ids = {r["slide_id"] for r in rows}
    paths = {}
    conflicts = []
    if source == "uni":
        embedding_dir = root / "embeddings"
        coordinate_dir = root / "patches" / "patches"
        embeddings = {p.stem: p for p in embedding_dir.glob("*.pt")}
        coordinates = {p.stem: p for p in coordinate_dir.glob("*.h5")}
        if set(embeddings) != expected_ids or set(coordinates) != expected_ids:
            raise ValueError(f"UNI coverage mismatch: embeddings missing={sorted(expected_ids-set(embeddings))[:10]}, coordinates missing={sorted(expected_ids-set(coordinates))[:10]}, extra embeddings={sorted(set(embeddings)-expected_ids)[:10]}, extra coordinates={sorted(set(coordinates)-expected_ids)[:10]}")
        return {sid: (embeddings[sid], coordinates[sid]) for sid in expected_ids}, conflicts

    class_dirs = [(root / "0-normal", 0), (root / "1-tumor", 1)]
    for folder, source_label in class_dirs:
        for path in folder.glob("*.csv"):
            if path.stem in paths:
                raise ValueError(f"Duplicate slide in class folders: {path.stem}")
            paths[path.stem] = (path, source_label)
    if set(paths) != expected_ids:
        raise ValueError(f"{source} coverage mismatch: missing={sorted(expected_ids-set(paths))[:10]}, extra={sorted(set(paths)-expected_ids)[:10]}")
    for row in rows:
        sid = row["slide_id"]
        source_label = paths[sid][1]
        if source_label != row["label"]:
            if KNOWN_LABEL_CONFLICTS.get(source, {}).get(sid) != (row["label"], source_label):
                raise ValueError(f"Unexpected label conflict for {sid}: canonical={row['label']} source={source_label}")
            conflicts.append({"slide_id": sid, "canonical_label": row["label"], "source_folder_label": source_label})
    return paths, conflicts


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def valid_npy(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> bool:
    if not path.exists():
        return False
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return array.shape == shape and array.dtype == dtype
    except (OSError, ValueError):
        return False


def prepare_csv(path: Path, target: Path, patches: int, dim: int) -> None:
    import pandas as pd

    if valid_npy(target, (patches, dim), np.dtype("float32")):
        return
    tmp = target.with_name(target.name + ".tmp")
    try:
        output = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(patches, dim))
        count = 0
        chunks = pd.read_csv(path, chunksize=2048, dtype=np.float32)
        expected_header = [str(i) for i in range(dim)]
        for chunk in chunks:
            if list(chunk.columns) != expected_header:
                raise ValueError(f"Unexpected CSV header in {path}")
            data = chunk.to_numpy(copy=False)
            if count + len(data) > patches or not np.isfinite(data).all():
                raise ValueError(f"Invalid rows or nonfinite values in {path}")
            output[count:count + len(data)] = data
            count += len(data)
        if count != patches:
            raise ValueError(f"Patch count mismatch in {path}: {count} != {patches}")
        output.flush()
        del output
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def prepare_uni(embedding_path: Path, coordinate_path: Path, bag_target: Path,
                coord_target: Path, patches: int) -> None:
    import h5py
    import torch

    bag_ok = valid_npy(bag_target, (patches, 1024), np.dtype("float32"))
    coord_ok = valid_npy(coord_target, (patches, 2), np.dtype("int64"))
    if bag_ok and coord_ok:
        return
    tensor = torch.load(embedding_path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tuple(tensor.shape) != (patches, 1024):
        raise ValueError(f"Invalid UNI embedding shape/dtype: {embedding_path}")
    bag = tensor.numpy()
    if not np.isfinite(bag).all():
        raise ValueError(f"Nonfinite UNI embedding: {embedding_path}")
    with h5py.File(coordinate_path, "r") as handle:
        dataset = handle["coords"]
        if dataset.shape != (patches, 2) or dataset.dtype != np.dtype("int64"):
            raise ValueError(f"Invalid UNI coordinates: {coordinate_path}")
        coords = dataset[:]
        if (coords < 0).any():
            raise ValueError(f"Negative UNI coordinates: {coordinate_path}")
        level_dim = dataset.attrs.get("level_dim")
        if level_dim is not None and (coords >= np.asarray(level_dim)).any():
            raise ValueError(f"Out-of-bounds UNI coordinates: {coordinate_path}")
    for data, target in [(bag, bag_target), (coords, coord_target)]:
        tmp = target.with_name(target.name + ".tmp")
        try:
            with tmp.open("wb") as handle:
                np.save(handle, data, allow_pickle=False)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE_SLUGS, required=True)
    parser.add_argument("--input-root", type=Path, help="Dataset root, defaults to its Kaggle input slug")
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("reference_slides.csv"))
    parser.add_argument("--output-root", type=Path, help="Defaults to /kaggle/working/camelyon16-SOURCE-prepared-v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = args.input_root or Path("/kaggle/input") / SOURCE_SLUGS[args.source]
    root = resolve_root(args.source, base)
    output = args.output_root or Path("/kaggle/working") / f"camelyon16-{args.source}-prepared-v1"
    rows = read_reference(args.reference)
    folds = folds_for(rows, args.seed)
    paths, conflicts = inventory(args.source, root, rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "bags").mkdir(exist_ok=True)
    if args.source == "uni":
        (output / "coords").mkdir(exist_ok=True)
    complete = output / "PREPROCESSING_COMPLETE.json"
    complete.unlink(missing_ok=True)

    fold_rows = [{"slide_id": r["slide_id"], "label": r["label"], "official_split": r["split"],
                  "validation_fold": "" if folds[r["slide_id"]] is None else folds[r["slide_id"]]} for r in rows]
    write_csv(output / "folds.csv", fold_rows, list(fold_rows[0]))
    manifest = []
    for index, row in enumerate(rows, 1):
        sid = row["slide_id"]
        patches = row[f"{args.source}_patches"]
        bag_rel = f"bags/{sid}.npy"
        coord_rel = f"coords/{sid}.npy" if args.source == "uni" else ""
        if args.source == "uni":
            emb, h5 = paths[sid]
            prepare_uni(emb, h5, output / bag_rel, output / coord_rel, patches)
        else:
            csv_path, _ = paths[sid]
            prepare_csv(csv_path, output / bag_rel, patches, SOURCE_DIMS[args.source])
        manifest.append({**fold_rows[index - 1], "n_patches": patches,
                         "feature_dim": SOURCE_DIMS[args.source], "bag_path": bag_rel,
                         "coords_path": coord_rel})
        if index % 25 == 0 or index == len(rows):
            print(f"Prepared {index}/{len(rows)} {args.source} slides", flush=True)
    write_csv(output / "slides.csv", manifest, list(manifest[0]))
    summary = {"schema_version": 1, "source": args.source, "input_slug": SOURCE_SLUGS[args.source],
               "input_root": str(root), "seed": args.seed, "folds": 5, "slides": len(rows),
               "train_slides": 270, "test_slides": 129,
               "patches": sum(r[f"{args.source}_patches"] for r in rows),
               "feature_dim": SOURCE_DIMS[args.source], "known_label_conflicts": conflicts,
               "label_decisions": {"test_114": {"label": 0, "class": "normal", "basis": "user instruction"}},
               "multiscale_columns": {"20x": [0, 512], "5x": [512, 1024]} if args.source == "multiscale" else None}
    complete.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
