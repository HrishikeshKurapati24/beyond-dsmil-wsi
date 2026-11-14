"""Official test inference for frozen Phase-4 Exp4 feature-assignment checkpoints.

Authors: Srimanreddy (DSMIL baseline), Hrishikesh Kurapati (Phase-4 experiments).
The paired notebook embeds this file and audited official-test coordinate metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score, roc_curve)

import flat_topk_uni_official_test as common


def hierarchy_indices(coords: torch.Tensor, x: torch.Tensor, anchors: list[int], grid: int = 4):
    """Exact feature-only, anchor-dependent partition from the validated runner."""
    xy = coords.detach().cpu().numpy()
    features = x.detach().cpu().numpy()
    cell = np.floor(xy / grid).astype(np.int64)
    per_cell: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(cell):
        per_cell.setdefault(tuple(key), []).append(i)
    groups = []
    for key, members in sorted(per_cell.items()):
        present = [a for a in anchors if a in members]
        if len(present) <= 1:
            groups.append((key, members, present[0] if present else None))
        else:
            partitions = {a: [] for a in present}
            for i in members:
                owner = (i if i in partitions else
                         min(present, key=lambda a: (np.linalg.norm(xy[i]-xy[a]), anchors.index(a))))
                partitions[owner].append(i)
            groups.extend((key, rows, a) for a, rows in partitions.items())
    branch_groups: list[list[int]] = [[] for _ in anchors]
    for group_index, (key, members, own_anchor) in enumerate(groups):
        if own_anchor is not None:
            branch = anchors.index(own_anchor)
        else:
            mean_feature = features[members].mean(axis=0)
            mean_feature = mean_feature / max(np.linalg.norm(mean_feature), 1e-12)
            ratings = []
            for a in anchors:
                anchor_feature = features[a] / max(np.linalg.norm(features[a]), 1e-12)
                ratings.append(0.5 * float(mean_feature @ anchor_feature))
            branch = int(np.argmax(ratings))
        branch_groups[branch].append(group_index)
    if any(not group for group in branch_groups):
        raise ValueError("Anchor branch has no microgroup")
    regions: dict[tuple[int, int, int], list[int]] = {}
    group_branch = {}
    for branch, members in enumerate(branch_groups):
        for gi in members:
            group_branch[gi] = branch
            key = groups[gi][0]
            regions.setdefault((branch, key[0] // 2, key[1] // 2), []).append(gi)
    region_list = [regions[key] for key in sorted(regions)]
    region_branch = [key[0] for key in sorted(regions)]
    parent = {gi: region for region, child in enumerate(region_list) for gi in child}
    assert sorted(i for _, members, _ in groups for i in members) == list(range(len(x)))
    return groups, branch_groups, region_list, region_branch, parent, group_branch


class HierarchicalFeatureAssignment(common.FlatTopKUNI):
    """Exp4 4(a), level 1, space_weight=0, grid width 4."""

    def pool(self, children: torch.Tensor, segment: torch.Tensor, anchors: torch.Tensor):
        count = len(anchors)
        query = self.b_classifier.q(children)
        anchor_query = self.b_classifier.q(anchors)
        logits = (query * anchor_query[segment]).sum(dim=1) / math.sqrt(128)
        maxima = logits.new_full((count,), -torch.inf).scatter_reduce(
            0, segment, logits, reduce="amax", include_self=True)
        exp = torch.exp(logits - maxima[segment])
        denominator = logits.new_zeros(count).index_add(0, segment, exp)
        weights = exp / denominator[segment]
        return children.new_zeros((count, children.shape[1])).index_add(
            0, segment, children * weights[:, None])

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        scores = self.i_classifier.fc(x)[:, 0]
        values = scores.detach().cpu().numpy()
        order = np.lexsort((np.arange(len(values)), -values))[:min(4, len(values))].tolist()
        anchors = x[order]
        groups, branch_groups, regions, region_branch, parent, group_branch = hierarchy_indices(
            coords, x, order, 4)
        patch_group = torch.empty(len(x), dtype=torch.long, device=x.device)
        for gi, (_, members, _) in enumerate(groups):
            patch_group[torch.as_tensor(members, device=x.device)] = gi
        group_branch_t = torch.as_tensor([group_branch[gi] for gi in range(len(groups))],
                                         dtype=torch.long, device=x.device)
        parent_t = torch.as_tensor([parent[gi] for gi in range(len(groups))],
                                   dtype=torch.long, device=x.device)
        region_branch_t = torch.as_tensor(region_branch, dtype=torch.long, device=x.device)
        group_vectors = self.pool(x, patch_group, anchors[group_branch_t])
        region_vectors = self.pool(group_vectors, parent_t, anchors[region_branch_t])
        root_vectors = self.pool(region_vectors, region_branch_t, anchors)
        return self.b_classifier.fcc(root_vectors.mean(dim=0).view(1, 1, 1024)).view(())


def discover(inputs: Path) -> tuple[Path, Path]:
    prepared = []
    for marker in inputs.rglob("PREPROCESSING_COMPLETE.json"):
        info = json.loads(marker.read_text())
        if info.get("source") == "uni" and info.get("feature_dim") == 1024:
            prepared.append(marker.parent)
    references = []
    for path in inputs.rglob("cv_summary.json"):
        if json.loads(path.read_text()).get("variant") == "exp4_4a_feature_assignment":
            references.append(path.parent)
    if len(prepared) != 1 or len(references) != 1:
        raise ValueError(f"Attach one prepared UNI dataset and one Exp4 feature assignment output: {prepared}, {references}")
    return prepared[0], references[0]


def validate(prepared: Path, reference: Path, metadata_path: Path):
    cfg = json.loads((reference / "config.json").read_text())
    cv = json.loads((reference / "cv_summary.json").read_text())
    complete = json.loads((reference / "COMPLETE.json").read_text())
    info = json.loads((prepared / "PREPROCESSING_COMPLETE.json").read_text())
    if cfg.get("variant") != cv.get("variant") or cv.get("variant") != "exp4_4a_feature_assignment":
        raise ValueError("Wrong checkpoint variant")
    if cfg.get("runner_sha256") != common.EXPECTED_RUNNER_SHA256:
        raise ValueError("Checkpoint runner differs from the verified model implementation")
    if cfg.get("grid_width_patch_units") != 4:
        raise ValueError("Unexpected hierarchy grid width")
    # The saved config incorrectly lists 0.5 as this variant's spatial weight.
    # The hash-verified runner instantiates this named variant with weight 0.0.
    if cfg.get("prepared_manifest_sha256") != common.sha256(prepared / "slides.csv"):
        raise ValueError("Checkpoint/UNI manifest mismatch")
    if cfg.get("seeds") != list(common.SEEDS) or cfg.get("folds") != list(common.FOLDS):
        raise ValueError("Unexpected seed or fold protocol")
    if len(cv.get("runs", [])) != 15 or complete.get("fits") != 15 or complete.get("test_evaluated") is not False:
        raise ValueError("Incomplete validation reference")
    if any(info.get(k) != v for k, v in {"source": "uni", "feature_dim": 1024,
                                         "slides": 399, "train_slides": 270,
                                         "test_slides": 129}.items()):
        raise ValueError("Unexpected prepared UNI dataset")
    if any(not (reference / f"seed_{s}" / f"fold_{f}" / "checkpoint.pt").is_file()
           for s in common.SEEDS for f in common.FOLDS):
        raise FileNotFoundError("Attach all 15 feature-assignment checkpoints")
    rows = common.read_csv(prepared / "slides.csv")
    train = [r for r in rows if r["official_split"] == "train"]
    test = [r for r in rows if r["official_split"] == "test"]
    if len(rows) != 399 or len(train) != 270 or len(test) != 129 or len({r["slide_id"] for r in rows}) != 399:
        raise ValueError("UNI slide manifest has incorrect coverage")
    if next(r["label"] for r in test if r["slide_id"] == "test_114") != "0":
        raise ValueError("Unexpected test_114 label")
    metadata = json.loads(metadata_path.read_text())
    items = metadata["slides"]
    by_id = {r["slide_id"]: r for r in items}
    if len(items) != 129 or len(by_id) != 129 or set(by_id) != {r["slide_id"] for r in test}:
        raise ValueError("Incomplete audited test coordinate metadata")
    for row in test:
        item = by_id[row["slide_id"]]
        n = int(row["n_patches"])
        if (item["patches"] != n or item["coordinate_shape"] != [n, 2] or
                item["patch_level"] != 0 or item["patch_size_px"] <= 0 or
                len(item["level_dim"]) != 2 or min(item["level_dim"]) <= 0 or
                item["out_of_bounds_coordinate_rows"] != 0):
            raise ValueError(f"Invalid test coordinate metadata: {row['slide_id']}")
    oof = common.read_csv(reference / "oof_predictions_ensemble.csv")
    by_train = {r["slide_id"]: r for r in train}
    if len(oof) != 270 or {r["slide_id"] for r in oof} != set(by_train):
        raise ValueError("Incomplete OOF predictions")
    if any(int(r["label"]) != int(by_train[r["slide_id"]]["label"]) or
           int(r["fold"]) != int(by_train[r["slide_id"]]["validation_fold"]) for r in oof):
        raise ValueError("OOF labels or folds differ from prepared UNI")
    y = np.array([int(r["label"]) for r in oof])
    p = np.array([float(r["probability"]) for r in oof])
    if not math.isclose(roc_auc_score(y, p), cv["oof_ensemble_auc"], abs_tol=1e-12):
        raise ValueError("OOF AUC does not match the saved summary")
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.flatnonzero(np.isfinite(thresholds))
    threshold = float(thresholds[finite[np.argmax((tpr - fpr)[finite])]])
    return test, by_id, threshold


def evaluate(prepared: Path, reference: Path, output: Path, test: list[dict], metadata: dict, threshold: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    hashes = {}
    for seed in common.SEEDS:
        for fold in common.FOLDS:
            path = reference / f"seed_{seed}" / f"fold_{fold}" / "checkpoint.pt"
            model = HierarchicalFeatureAssignment().to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
            model.eval()
            models.append((seed, fold, model))
            hashes[f"seed_{seed}/fold_{fold}/checkpoint.pt"] = common.sha256(path)
    per_checkpoint, ensemble = [], []
    with torch.inference_mode():
        for index, row in enumerate(sorted(test, key=lambda r: r["slide_id"]), 1):
            n = int(row["n_patches"])
            bag = np.load(prepared / row["bag_path"], mmap_mode="r", allow_pickle=False)
            raw = np.load(prepared / row["coords_path"], mmap_mode="r", allow_pickle=False)
            item = metadata[row["slide_id"]]
            if (bag.shape != (n, 1024) or bag.dtype != np.float32 or
                    raw.shape != (n, 2) or raw.dtype != np.int64 or
                    not np.isfinite(bag).all() or np.any(raw < 0) or
                    np.any(raw >= np.asarray(item["level_dim"]))):
                raise ValueError(f"Invalid UNI test arrays: {row['slide_id']}")
            features = torch.tensor(bag, dtype=torch.float32, device=device)
            coords = torch.tensor(np.asarray(raw, dtype=np.float32) / item["patch_size_px"],
                                  dtype=torch.float32, device=device)
            probs = []
            for seed, fold, model in models:
                p = float(torch.sigmoid(model(features, coords)))
                if not math.isfinite(p):
                    raise ValueError(f"Nonfinite probability: {row['slide_id']}")
                probs.append(p)
                per_checkpoint.append({"seed": seed, "fold": fold, "slide_id": row["slide_id"],
                                       "label": int(row["label"]), "probability": p})
            avg = float(np.mean(probs))
            ensemble.append({"slide_id": row["slide_id"], "label": int(row["label"]),
                             "probability": avg, "prediction": int(avg >= threshold)})
            print(f"{index:3d}/129 {row['slide_id']} ensemble_probability={avg:.6f}", flush=True)
            del features, coords
    if len(per_checkpoint) != 1935 or len(ensemble) != 129:
        raise ValueError("Incomplete official-test predictions")
    output.mkdir(parents=True, exist_ok=True)
    common.write_csv(output / "test_predictions_per_checkpoint.csv", per_checkpoint,
                     ["seed", "fold", "slide_id", "label", "probability"])
    common.write_csv(output / "test_predictions_ensemble.csv", ensemble,
                     ["slide_id", "label", "probability", "prediction"])
    y = [r["label"] for r in ensemble]
    pred = [r["prediction"] for r in ensemble]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    summary = {"model": "Phase-4 Exp4 4(a) feature assignment", "test_slides": 129,
               "n_models": 15, "test_auc": float(roc_auc_score(y, [r["probability"] for r in ensemble])),
               "test_accuracy": float(accuracy_score(y, pred)),
               "test_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
               "test_sensitivity": float(tp / (tp + fn)),
               "test_specificity": float(tn / (tn + fp)),
               "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
               "threshold_from_oof": threshold,
               "ensemble_rule": "mean of 15 frozen checkpoint bag probabilities",
               "assignment_spatial_weight_actual": 0.0,
               "prepared_manifest_sha256": common.sha256(prepared / "slides.csv"),
               "test_coordinate_metadata_sha256": common.sha256(Path(__file__).parent / "test_coordinate_metadata.json"),
               "checkpoint_sha256": hashes}
    (output / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoint_sha256"}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--metadata", type=Path,
                        default=Path("/kaggle/working/test_coordinate_metadata.json"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("/kaggle/working/feature-assignment-uni-official-test"))
    args = parser.parse_args()
    prepared, reference = discover(args.input_root)
    test, metadata, threshold = validate(prepared, reference, args.metadata)
    print("Prepared:", prepared, "\nFrozen checkpoints:", reference,
          "\nOOF threshold:", threshold, flush=True)
    evaluate(prepared, reference, args.output_root, test, metadata, threshold)


if __name__ == "__main__":
    main()
