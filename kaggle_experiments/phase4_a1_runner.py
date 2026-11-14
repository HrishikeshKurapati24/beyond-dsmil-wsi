"""Stage-specific A1 test reporting and Phase-4 UNI validation.

Separate notebooks invoke the A1 test stage and the validation-only UNI stages.
The UNI models retain A1's four-anchor rank selector and objective.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from torch import nn
from torch.nn import functional as F

import baseline_dsmil as base
import ab_validation_runner as ab


SEEDS = (42, 43, 44)
FOLDS = range(5)
UNISCALE_MANIFEST_SHA256 = ab.PREPARED_SHA256
EPOCHS, PATIENCE = 50, 10
LR, WEIGHT_DECAY = 1e-4, 1e-3
ANCHORS, CANDIDATES, TOP_C = 4, 32, 32
NEARBY_SCALE_PATCHES = 8.0
GAMMA_GRID = (0.0, 0.25, 0.5, 1.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_reference(bundle: Path, destination: Path) -> Path:
    """Extract only the small, checksummed A1 test reference."""
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("reference_manifest.json"))
        members = manifest["member_sha256"]
        if (manifest.get("kind") != "a1_alpha025_test_reference" or
                set(archive.namelist()) != set(members) | {"reference_manifest.json"}):
            raise ValueError("Invalid A1 reference bundle")
        if len([x for x in members if x.endswith(".pt")]) != 15:
            raise ValueError("A1 reference requires 15 checkpoints")
        destination.mkdir(parents=True, exist_ok=False)
        for name, expected in members.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe bundle member: {name}")
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"A1 bundle member checksum mismatch: {name}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return destination


def checked_reference_directory(source: Path, manifest_path: Path,
                                destination: Path) -> Path:
    """Copy and verify the archived A1 files from an attached Kaggle input."""
    if not source.is_dir():
        raise ValueError(f"Missing A1 reference directory: {source}")
    manifest = json.loads(manifest_path.read_text())
    members = manifest.get("member_sha256", {})
    if (manifest.get("kind") != "a1_alpha025_test_reference" or
            len([name for name in members if name.endswith(".pt")]) != 15):
        raise ValueError("Invalid A1 reference manifest")
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected in members.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe A1 reference member: {name}")
        path = source / relative
        if not path.is_file():
            raise ValueError(f"Missing A1 reference member: {path}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"A1 reference checksum mismatch: {path}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return destination


def load_uni_manifest(prepared: Path) -> tuple[list[dict], dict]:
    info = json.loads((prepared / "PREPROCESSING_COMPLETE.json").read_text())
    if any(info.get(k) != v for k, v in {
            "source": "uni", "feature_dim": 1024, "seed": 42, "folds": 5,
            "slides": 399, "train_slides": 270, "test_slides": 129}.items()):
        raise ValueError("UNI prepared completion metadata does not match the study protocol")
    rows = base.read_csv(prepared / "slides.csv")
    if len(rows) != 399 or len({r["slide_id"] for r in rows}) != 399:
        raise ValueError("Expected 399 distinct UNI slides")
    for row in rows:
        sid = row["slide_id"]
        row["label"] = int(row["label"])
        row["n_patches"] = int(row["n_patches"])
        if row["label"] not in (0, 1):
            raise ValueError(f"Invalid UNI label: {sid}")
        if row["official_split"] == "train":
            row["validation_fold"] = int(row["validation_fold"])
            if not 0 <= row["validation_fold"] < 5 or sid.startswith("test_"):
                raise ValueError(f"Invalid UNI training fold: {sid}")
        elif row["official_split"] == "test":
            if row["validation_fold"] or not sid.startswith("test_"):
                raise ValueError(f"Invalid UNI official-test row: {sid}")
            row["validation_fold"] = ""
        else:
            raise ValueError(f"Invalid UNI split: {sid}")
        if row["n_patches"] < 1 or int(row["feature_dim"]) != 1024:
            raise ValueError(f"Invalid UNI patch count/dimension: {sid}")
        if row["official_split"] == "test":
            # Phase-4 runs are validation-only: do not open UNI test feature files.
            continue
        for field in ("bag_path", "coords_path"):
            rel = Path(row[field])
            if not row[field] or rel.is_absolute() or ".." in rel.parts or not (prepared / rel).is_file():
                raise ValueError(f"Missing or unsafe UNI {field}: {sid}")
        # Check .npy headers without loading all 4.9M patch rows into memory.
        bag = np.load(prepared / row["bag_path"], mmap_mode="r", allow_pickle=False)
        coords = np.load(prepared / row["coords_path"], mmap_mode="r", allow_pickle=False)
        if (bag.shape != (row["n_patches"], 1024) or bag.dtype != np.float32 or
                coords.shape != (row["n_patches"], 2) or coords.dtype != np.int64):
            raise ValueError(f"Invalid UNI bag/coordinate shape or dtype: {sid}")
        del bag, coords
    train = [r for r in rows if r["official_split"] == "train"]
    test = [r for r in rows if r["official_split"] == "test"]
    if (len(train) != 270 or len(test) != 129 or
            any(sum(r["validation_fold"] == fold for r in train) != 54 for fold in FOLDS) or
            next(r["label"] for r in rows if r["slide_id"] == "test_114") != 0):
        raise ValueError("UNI cohort, folds, or test_114 label mismatch")
    return rows, info


def load_coordinate_metadata(path: Path, rows: list[dict]) -> dict[str, dict]:
    audit = json.loads(path.read_text())
    by_id = {r["slide_id"]: r for r in audit["slides"]}
    train = [r for r in rows if r["official_split"] == "train"]
    if len(by_id) != 270 or set(by_id) != {r["slide_id"] for r in train}:
        raise ValueError("UNI training coordinate audit slide coverage mismatch")
    for row in train:
        item = by_id[row["slide_id"]]
        dim = item["level_dim"]
        if (len(dim) != 2 or min(dim) <= 0 or item.get("patch_size_px", 0) <= 0 or
                item.get("patch_level") != 0 or item["patches"] != row["n_patches"] or
                item["coordinate_shape"] != [row["n_patches"], 2] or
                item["out_of_bounds_coordinate_rows"] != 0):
            raise ValueError(f"Invalid source coordinate metadata: {row['slide_id']}")
    return by_id


def load_uni(prepared: Path, row: dict, device: torch.device,
             metadata: dict[str, dict], spatial: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    array = np.load(prepared / row["bag_path"], mmap_mode="r", allow_pickle=False)
    if array.shape != (row["n_patches"], 1024) or array.dtype != np.float32:
        raise ValueError(f"Invalid UNI feature array: {row['slide_id']}")
    features = torch.tensor(array, dtype=torch.float32, device=device)
    if not torch.isfinite(features).all():
        raise ValueError(f"Nonfinite UNI features: {row['slide_id']}")
    if not spatial:
        return features, None
    raw = np.load(prepared / row["coords_path"], mmap_mode="r", allow_pickle=False)
    dim = np.asarray(metadata[row["slide_id"]]["level_dim"], dtype=np.float32)
    if (raw.shape != (len(features), 2) or raw.dtype != np.int64 or
            np.any(raw < 0) or np.any(raw >= dim)):
        raise ValueError(f"Invalid UNI coordinates or bounds: {row['slide_id']}")
    coords = torch.tensor(np.asarray(raw, dtype=np.float32) /
                          float(metadata[row["slide_id"]]["patch_size_px"]),
                          dtype=torch.float32, device=device)
    return features, coords


def select_spatial_anchors(features: torch.Tensor, scores: torch.Tensor,
                           original_ids: torch.Tensor, coords: torch.Tensor):
    """A1 top-32 candidate pool with 0.50 score/0.25 feature/0.25 space ranks."""
    if len(features) < 1 or coords.shape != (len(features), 2):
        raise ValueError("Spatial selection needs paired feature and coordinate rows")
    k = min(ANCHORS, len(features))
    values = scores.detach().cpu().numpy()
    ids = original_ids.detach().cpu().numpy()
    candidates = np.lexsort((ids, -values))[:min(CANDIDATES, len(features))].tolist()
    chosen = [candidates[0]]
    records = [{"score_rank": 1, "feature_rank": None, "spatial_rank": None}]
    normalized = F.normalize(features.detach()[candidates], p=2, dim=1, eps=1e-12)
    cosine = (normalized @ normalized.T).cpu().numpy()
    candidate_coords = coords.detach()[candidates]
    distance = torch.linalg.vector_norm(
        candidate_coords[:, None, :] - candidate_coords[None, :, :], dim=2).cpu().numpy()
    position = {row: local for local, row in enumerate(candidates)}
    while len(chosen) < k:
        remaining = [row for row in candidates if row not in chosen]
        score_rank = {row: rank for rank, row in enumerate(remaining, 1)}
        feature_gap = {row: 1.0 - max(float(cosine[position[row], position[j]])
                                      for j in chosen) for row in remaining}
        spatial_gap = {row: min(float(distance[position[row], position[j]])
                                for j in chosen) for row in remaining}
        feature_order = sorted(remaining, key=lambda row: (-feature_gap[row], int(ids[row])))
        spatial_order = sorted(remaining, key=lambda row: (-spatial_gap[row], int(ids[row])))
        feature_rank = {row: rank for rank, row in enumerate(feature_order, 1)}
        spatial_rank = {row: rank for rank, row in enumerate(spatial_order, 1)}
        winner = min(remaining, key=lambda row: (
            0.50 * score_rank[row] + 0.25 * feature_rank[row] + 0.25 * spatial_rank[row],
            -float(values[row]), int(ids[row])))
        chosen.append(winner)
        records.append({"score_rank": score_rank[winner],
                        "feature_rank": feature_rank[winner],
                        "spatial_rank": spatial_rank[winner]})
    return torch.as_tensor(chosen, dtype=torch.long, device=features.device), records


def top_c_mask(biased_logits: torch.Tensor, original_ids: torch.Tensor,
               limit: int) -> torch.Tensor:
    """Choose the highest combined appearance-and-distance scores per anchor."""
    n, k = biased_logits.shape
    keep = min(limit, n)
    ids = original_ids.detach().cpu().numpy()
    scores = biased_logits.detach().cpu().numpy()
    mask = torch.zeros((n, k), dtype=torch.bool, device=biased_logits.device)
    for column in range(k):
        ordering = np.lexsort((ids, -scores[:, column]))[:keep]
        mask[torch.as_tensor(ordering, device=biased_logits.device), column] = True
    return mask


class A1BagClassifier(base.BClassifier):
    def __init__(self, dimension: int, mode: str = "none", gamma: float = 0.0):
        super().__init__(dimension, 1, anchors=ANCHORS)
        if mode not in ("none", "relative_all", "relative_top32") or gamma < 0:
            raise ValueError("Invalid Phase-4 mode or gamma")
        self.mode, self.gamma = mode, gamma
        self.spatial = mode != "none"

    def forward(self, features, scores, original_ids=None, coords=None):
        if original_ids is None:
            original_ids = torch.arange(len(features), device=features.device)
        if self.spatial and (coords is None or coords.shape != (len(features), 2)):
            raise ValueError("Spatial A1 requires one x/y coordinate per feature")
        q = self.q(features).view(len(features), -1)
        if self.spatial:
            indices, ranks = select_spatial_anchors(features, scores[:, 0], original_ids, coords)
        else:
            indices, ranks = ab.select_diverse(features, scores[:, 0], original_ids, 0.25, ANCHORS)
        anchor_queries = q.index_select(0, indices)
        content_logits = (q @ anchor_queries.T) / math.sqrt(128)
        if self.spatial:
            distance = torch.linalg.vector_norm(
                coords[:, None, :] - coords.index_select(0, indices)[None, :, :], dim=2)
            spatial_penalty = self.gamma * torch.clamp(
                (distance / NEARBY_SCALE_PATCHES).square(), max=1.0)
            logits = content_logits - spatial_penalty
        else:
            logits = content_logits
        if self.mode == "relative_top32":
            support = top_c_mask(logits, original_ids, TOP_C)
            logits = logits.masked_fill(~support, -torch.inf)
        else:
            support = torch.ones_like(logits, dtype=torch.bool)
        attention = F.softmax(logits, dim=0)
        anchor_bags = attention.T @ self.v(features)
        bag = anchor_bags.mean(dim=0, keepdim=True).view(1, 1, features.shape[1])
        prediction = self.fcc(bag).view(1, -1)
        return prediction, attention, {"indices": indices, "rank_records": ranks,
                                       "anchor_bags": anchor_bags,
                                       "support_count": support.sum(dim=0),
                                       "support_contains_anchor": support[indices, torch.arange(len(indices), device=indices.device)]}


class A1UniNet(nn.Module):
    def __init__(self, mode: str = "none", gamma: float = 0.0):
        super().__init__()
        self.i_classifier = base.FCLayer(1024, 1)
        self.b_classifier = A1BagClassifier(1024, mode, gamma)

    def forward(self, features, original_ids=None, coords=None):
        features, scores = self.i_classifier(features)
        bag_logit, attention, details = self.b_classifier(features, scores, original_ids, coords)
        return scores, bag_logit, attention, details


def make_uni_model(mode: str, gamma: float, device: torch.device):
    # Restore RNG after construction so identical seeds initialize shared A1
    # weights identically in all three variants.
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model = A1UniNet(mode, gamma).to(device)
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.9),
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=5e-6)
    return model, optimizer, scheduler


def uni_step(model, features, label, criterion, original_ids=None, coords=None):
    scores, bag_logit, attention, details = model(features, original_ids, coords)
    target = torch.tensor([[float(label)]], device=features.device)
    loss = 0.5 * criterion(bag_logit, target) + 0.5 * criterion(scores.max().view(1, 1), target)
    return loss, float(torch.sigmoid(bag_logit.detach()).item()), scores, attention, details


def train_epoch(model, rows, prepared, device, metadata, optimizer, criterion, seed, epoch, fold):
    model.train()
    order = np.random.default_rng(seed + fold * 100000 + epoch).permutation(len(rows))
    total = 0.0
    for index in order:
        row = rows[int(index)]
        features, coords = load_uni(prepared, row, device, metadata, model.b_classifier.spatial)
        permutation = torch.randperm(len(features), device=device)
        features = features[permutation]
        if coords is not None:
            coords = coords[permutation]
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _, _ = uni_step(model, features, row["label"], criterion, permutation, coords)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite objective: {row['slide_id']}")
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
    return total / len(rows)


@torch.no_grad()
def predict_uni(model, rows, prepared, device, metadata, criterion, diagnostics=False):
    model.eval()
    predictions, details_rows = [], []
    total = 0.0
    for row in rows:
        features, coords = load_uni(prepared, row, device, metadata,
                                    model.b_classifier.spatial or diagnostics)
        loss, probability, logits, attention, details = uni_step(
            model, features, row["label"], criterion, coords=coords)
        total += float(loss.detach().cpu())
        predictions.append({"slide_id": row["slide_id"], "label": row["label"],
                            "probability": probability})
        if diagnostics:
            idx = details["indices"]
            k = len(idx)
            selected = F.normalize(features[idx], p=2, dim=1)
            bags = F.normalize(details["anchor_bags"], p=2, dim=1)
            off = ~torch.eye(k, dtype=torch.bool, device=device)
            patch_px = int(metadata[row["slide_id"]]["patch_size_px"])
            pairwise = []
            for left in range(k):
                for right in range(left + 1, k):
                    cosine = float(torch.dot(selected[left], selected[right]).item())
                    distance_patches = float(torch.linalg.vector_norm(
                        coords[idx[left]] - coords[idx[right]]).item())
                    pairwise.append({"anchor_left": left, "anchor_right": right,
                                     "feature_cosine_dissimilarity": 1.0 - cosine,
                                     "spatial_distance_patch_widths": distance_patches,
                                     "spatial_distance_pixels": distance_patches * patch_px})
            anchor_distances = torch.linalg.vector_norm(
                coords[:, None, :] - coords.index_select(0, idx)[None, :, :], dim=2)
            weighted_radius = (attention * anchor_distances).sum(dim=0)
            details_rows.append({
                "slide_id": row["slide_id"], "label": row["label"],
                "anchor_row_indices": idx.cpu().tolist(),
                "anchor_instance_logits": logits[idx, 0].cpu().tolist(),
                "anchor_rank_records": details["rank_records"],
                "anchor_coordinates_pixels": torch.round(coords[idx] * patch_px).to(torch.int64).cpu().tolist(),
                "anchor_pairwise": pairwise,
                "min_anchor_spatial_distance_patch_widths": min(
                    (p["spatial_distance_patch_widths"] for p in pairwise), default=None),
                "mean_anchor_feature_cosine_dissimilarity": float(np.mean(
                    [p["feature_cosine_dissimilarity"] for p in pairwise])) if pairwise else None,
                "attention_support_counts": details["support_count"].cpu().tolist(),
                "attention_support_contains_anchor": details["support_contains_anchor"].cpu().tolist(),
                "attention_column_sums": attention.sum(dim=0).cpu().tolist(),
                "attention_weighted_radius_patch_widths": weighted_radius.cpu().tolist(),
                "mean_pairwise_anchor_cosine": float((selected @ selected.T)[off].mean()) if k > 1 else None,
                "mean_pairwise_anchor_bag_cosine": float((bags @ bags.T)[off].mean()) if k > 1 else None,
                "mean_pairwise_attention_cosine": float(base.attention_overlap(attention)),
            })
    return total / len(rows), predictions, details_rows


def test_metrics(predictions: list[dict], threshold: float, origin: str) -> dict:
    labels = [int(r["label"]) for r in predictions]
    classes = [int(float(r["probability"]) >= threshold) for r in predictions]
    tn, fp, fn, tp = confusion_matrix(labels, classes, labels=[0, 1]).ravel()
    return {"origin": origin, "test_slides": len(predictions), "n_models": 15,
            "threshold_from_oof": threshold, "test_auc": base.auc(predictions),
            "test_accuracy": float(accuracy_score(labels, classes)),
            "test_balanced_accuracy": float(balanced_accuracy_score(labels, classes)),
            "test_sensitivity": float(tp / (tp + fn)),
            "test_specificity": float(tn / (tn + fp)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "test_114_label": 0,
            "ensemble_rule": "mean of 15 checkpoint bag probabilities"}


def write_test(out: Path, per_checkpoint: list[dict], test: list[dict], threshold: float,
               origin: str) -> dict:
    if len(per_checkpoint) != 129 * 15:
        raise ValueError(f"Expected 1935 test predictions, got {len(per_checkpoint)}")
    expected = {r["slide_id"]: r["label"] for r in test}
    by_slide = {sid: [] for sid in expected}
    for item in per_checkpoint:
        sid = item["slide_id"]
        if sid not in expected or item["label"] != expected[sid]:
            raise ValueError(f"Invalid test slide/label: {sid}")
        if not math.isfinite(item["probability"]) or not 0 <= item["probability"] <= 1:
            raise ValueError(f"Invalid test probability: {sid}")
        by_slide[sid].append(item)
    if any(len(values) != 15 or {(r["seed"], r["fold"]) for r in values} !=
           {(s, f) for s in SEEDS for f in FOLDS} for values in by_slide.values()):
        raise ValueError("Incomplete test ensemble coverage")
    ensemble = [{"slide_id": sid, "label": expected[sid],
                 "probability": float(np.mean([r["probability"] for r in by_slide[sid]]))}
                for sid in sorted(expected)]
    for row in ensemble:
        row["prediction"] = int(row["probability"] >= threshold)
    base.write_csv(out / "test_predictions_per_checkpoint.csv", per_checkpoint,
                   ["seed", "fold", "slide_id", "label", "probability"])
    base.write_csv(out / "test_predictions_ensemble.csv", ensemble,
                   ["slide_id", "label", "probability", "prediction"])
    summary = test_metrics(ensemble, threshold, origin)
    base.write_json(out / "test_summary.json", summary)
    return summary


def evaluate_a1_reference(reference: Path, prepared: Path, rows: list[dict],
                          output: Path, device: torch.device) -> dict:
    config = json.loads((reference / "config.json").read_text())
    summary = json.loads((reference / "cv_summary.json").read_text())
    variant = config["variant"]
    if (variant != ab.Variant("A1_alpha_025", 4, alpha=0.25).__dict__ or
            config["prepared_manifest_sha256"] != UNISCALE_MANIFEST_SHA256 or
            config["runner_sha256"] != digest(Path(ab.__file__)) or
            config["baseline_sha256"] != digest(Path(base.__file__)) or
            config["seeds"] != list(SEEDS) or config["folds"] != list(FOLDS) or
            config["test_evaluated"] is not False or len(summary["runs"]) != 15 or
            summary["variant"] != variant or summary["seeds"] != list(SEEDS)):
        raise ValueError("A1 archived training contract mismatch")
    oof = base.read_csv(reference / "oof_predictions_ensemble.csv")
    train = {r["slide_id"]: r for r in rows if r["official_split"] == "train"}
    if len(oof) != 270 or len({r["slide_id"] for r in oof}) != 270 or {r["slide_id"] for r in oof} != set(train):
        raise ValueError("Incomplete A1 OOF ensemble")
    for row in oof:
        original = train[row["slide_id"]]
        if (int(row["label"]), int(row["fold"])) != (original["label"], original["validation_fold"]):
            raise ValueError(f"A1 OOF identity mismatch: {row['slide_id']}")
        row["label"] = int(row["label"])
        row["probability"] = float(row["probability"])
        if not math.isfinite(row["probability"]) or not 0 <= row["probability"] <= 1:
            raise ValueError(f"Invalid A1 OOF probability: {row['slide_id']}")
    if not math.isclose(base.auc(oof), summary["oof_ensemble_auc"], abs_tol=1e-12):
        raise ValueError("A1 OOF AUC mismatch")
    threshold = base.threshold_from_oof(oof)
    test = [r for r in rows if r["official_split"] == "test"]
    output.mkdir(parents=True, exist_ok=False)
    criterion = nn.BCEWithLogitsLoss()
    all_predictions = []
    for seed in SEEDS:
        for fold in FOLDS:
            model = ab.ABMILNet(ab.Variant("A1_alpha_025", 4, alpha=0.25)).to(device)
            model.load_state_dict(torch.load(reference / f"seed_{seed}" / f"fold_{fold}.pt",
                                             map_location=device, weights_only=True), strict=True)
            _, predicted, _ = ab.predict(model, test, prepared, device, criterion,
                                         diagnostics_enabled=False)
            all_predictions.extend({"seed": seed, "fold": fold, **r} for r in predicted)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"A1 existing checkpoint test: seed={seed} fold={fold}", flush=True)
    result = write_test(output, all_predictions, test, threshold, "A1 alpha=0.25; archived 512-d uniscale")
    base.write_json(output / "COMPLETE.json", {"status": "complete", "test_summary": result,
                                                "archived_cv_source": str(reference),
                                                "prepared_manifest_sha256": digest(prepared / "slides.csv")})
    return result


def train_uni_variant(name: str, mode: str, gamma: float, rows: list[dict], prepared: Path,
                      metadata: dict[str, dict], output: Path, device: torch.device) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    config = {"name": name, "feature_source": "uni", "feature_dim": 1024,
              "a1_alpha": 0.25, "anchors": 4, "candidate_pool": 32,
              "attention_diversity_weight": 0.0, "mode": mode, "gamma": gamma,
              "anchor_rank_weights": ([0.5, 0.25, 0.25] if mode != "none" else [0.75, 0.25]),
              "relative_distance_unit": "source patch widths" if mode != "none" else None,
              "nearby_scale_patch_widths": NEARBY_SCALE_PATCHES if mode != "none" else None,
              "attention_top_c": TOP_C if mode == "relative_top32" else None,
              "top_c_rule": "highest appearance-minus-distance logits, then masked softmax" if mode == "relative_top32" else None,
              "seeds": list(SEEDS), "folds": list(FOLDS), "epochs": EPOCHS,
              "patience": PATIENCE, "lr": LR, "weight_decay": WEIGHT_DECAY,
              "betas": [0.5, 0.9], "eta_min": 5e-6,
              "checkpoint": "max validation AUC; tie by min validation loss",
              "prepared_manifest_sha256": digest(prepared / "slides.csv"),
              "runner_sha256": digest(Path(__file__)),
              "versions": {"torch": torch.__version__, "numpy": np.__version__,
                           "sklearn": sklearn.__version__},
              "coordinate_pairing": "Assumed row-aligned from UNI source .pt/.h5; not independently proven"}
    base.write_json(output / "config.json", config)
    criterion = nn.BCEWithLogitsLoss()
    all_oof, fold_results = [], []
    for seed in SEEDS:
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        for fold in FOLDS:
            train = [r for r in rows if r["official_split"] == "train" and r["validation_fold"] != fold]
            valid = [r for r in rows if r["official_split"] == "train" and r["validation_fold"] == fold]
            if len(train) != 216 or len(valid) != 54:
                raise ValueError(f"Invalid UNI fold size: {fold}")
            base.seed_all(seed)
            model, optimizer, scheduler = make_uni_model(mode, gamma, device)
            best_auc, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
            history = []
            start = time.monotonic()
            for epoch in range(1, EPOCHS + 1):
                train_loss = train_epoch(model, train, prepared, device, metadata,
                                         optimizer, criterion, seed, epoch, fold)
                valid_loss, predicted, _ = predict_uni(model, valid, prepared, device,
                                                       metadata, criterion)
                valid_auc = base.auc(predicted)
                valid_acc = float(accuracy_score([r["label"] for r in predicted],
                                                 [int(r["probability"] >= 0.5) for r in predicted]))
                if not all(map(math.isfinite, (train_loss, valid_loss, valid_auc, valid_acc))):
                    raise ValueError(f"Nonfinite training metric: {name} seed={seed} fold={fold}")
                history.append({"seed": seed, "fold": fold, "epoch": epoch,
                                "train_loss": train_loss, "valid_loss": valid_loss,
                                "valid_auc": valid_auc, "valid_accuracy_0.5": valid_acc,
                                "lr": optimizer.param_groups[0]["lr"]})
                improved = (valid_auc > best_auc + 1e-12 or
                            (abs(valid_auc - best_auc) <= 1e-12 and valid_loss < best_loss))
                if improved:
                    best_auc, best_loss, best_epoch, stale = valid_auc, valid_loss, epoch, 0
                    torch.save(model.state_dict(), seed_dir / f"fold_{fold}.pt")
                    base.write_csv(seed_dir / f"fold_{fold}_oof.csv",
                                   [{"seed": seed, "fold": fold, **r} for r in predicted],
                                   ["seed", "fold", "slide_id", "label", "probability"])
                else:
                    stale += 1
                scheduler.step()
                print(f"{name} seed={seed} fold={fold} epoch={epoch} "
                      f"train={train_loss:.4f} val={valid_loss:.4f} auc={valid_auc:.4f}", flush=True)
                if stale >= PATIENCE:
                    break
            base.write_csv(seed_dir / f"fold_{fold}_history.csv", history, list(history[0]))
            model.load_state_dict(torch.load(seed_dir / f"fold_{fold}.pt",
                                             map_location=device, weights_only=True))
            train_eval_loss, train_pred, train_diagnostics = predict_uni(
                model, train, prepared, device, metadata, criterion, diagnostics=True)
            _, _, validation_diagnostics = predict_uni(
                model, valid, prepared, device, metadata, criterion, diagnostics=True)
            if (len(train_diagnostics) != 216 or len(validation_diagnostics) != 54 or
                    any(len(item["anchor_pairwise"]) != 6 for item in
                        train_diagnostics + validation_diagnostics)):
                raise ValueError(f"Incomplete anchor-pair diagnostics: {name} {seed} {fold}")
            base.write_json(seed_dir / f"fold_{fold}_anchor_diagnostics.json",
                            {"epoch": best_epoch, "train_slides": train_diagnostics,
                             "validation_slides": validation_diagnostics})
            selected = base.read_csv(seed_dir / f"fold_{fold}_oof.csv")
            selected = [{"seed": int(r["seed"]), "fold": int(r["fold"]),
                         "slide_id": r["slide_id"], "label": int(r["label"]),
                         "probability": float(r["probability"])} for r in selected]
            all_oof.extend(selected)
            fold_result = {"seed": seed, "fold": fold, "best_epoch": best_epoch,
                           "validation_auc": best_auc, "validation_loss": best_loss,
                           "training_loss_at_best_checkpoint": train_eval_loss,
                           "training_auc_at_best_checkpoint": base.auc(train_pred),
                           "training_accuracy_0.5": float(accuracy_score(
                               [r["label"] for r in train_pred],
                               [int(r["probability"] >= 0.5) for r in train_pred])),
                           "validation_accuracy_0.5": float(accuracy_score(
                               [r["label"] for r in selected],
                               [int(r["probability"] >= 0.5) for r in selected])),
                           "elapsed_seconds": round(time.monotonic() - start, 1)}
            fold_results.append(fold_result)
            base.write_json(seed_dir / f"fold_{fold}_summary.json", fold_result)
            del model, optimizer, scheduler
            if device.type == "cuda":
                torch.cuda.empty_cache()
    all_oof.sort(key=lambda r: (r["seed"], r["slide_id"]))
    base.write_csv(output / "oof_predictions_all_seeds.csv", all_oof,
                   ["seed", "fold", "slide_id", "label", "probability"])
    ensemble = ab.ensemble_oof(all_oof)
    base.write_csv(output / "oof_predictions_ensemble.csv", ensemble,
                   ["slide_id", "fold", "label", "probability"])
    cv = {"name": name, "mode": mode, "gamma": gamma, "runs": fold_results,
          "mean_validation_auc": float(np.mean([r["validation_auc"] for r in fold_results])),
          "std_validation_auc": float(np.std([r["validation_auc"] for r in fold_results], ddof=1)),
          "mean_validation_loss": float(np.mean([r["validation_loss"] for r in fold_results])),
          "oof_ensemble_auc": base.auc(ensemble),
          "test_evaluated": False}
    base.write_json(output / "cv_summary.json", cv)
    base.write_json(output / "COMPLETE.json", {"status": "complete", "cv": cv,
                                                "checkpoints": 15, "oof_rows": 810,
                                                "anchor_diagnostic_slide_evaluations": 4050,
                                                "test_evaluated": False})
    return cv


def run_gamma_family(name: str, mode: str, rows: list[dict], prepared: Path,
                     metadata: dict[str, dict], output: Path, device: torch.device) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    candidates = []
    for gamma in GAMMA_GRID:
        tag = f"gamma_{int(round(gamma * 100)):03d}"
        cv = train_uni_variant(f"{name}_{tag}", mode, gamma, rows, prepared,
                               metadata, output / tag, device)
        candidates.append({"gamma": gamma, "output": tag,
                           "mean_validation_auc": cv["mean_validation_auc"],
                           "mean_validation_loss": cv["mean_validation_loss"],
                           "oof_ensemble_auc": cv["oof_ensemble_auc"]})
    # Use validation AUC, then validation loss, then the smaller gamma.
    best = max(candidates, key=lambda row: (
        row["mean_validation_auc"], -row["mean_validation_loss"], -row["gamma"]))
    result = {"name": name, "mode": mode, "gamma_grid": list(GAMMA_GRID),
              "selection_rule": "highest mean validation AUC across 15 fits; tie by lower mean validation loss, then smaller gamma",
              "selected_gamma": best["gamma"], "selected_output": best["output"],
              "candidates": candidates, "test_evaluated": False}
    base.write_json(output / "gamma_tuning_summary.json", result)
    base.write_csv(output / "gamma_tuning_table.csv", candidates, list(candidates[0]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a1-test", "uni", "spatial-a", "spatial-b"), required=True)
    parser.add_argument("--uniscale-prepared", type=Path)
    parser.add_argument("--uni-prepared", type=Path)
    parser.add_argument("--a1-reference-bundle", type=Path)
    parser.add_argument("--a1-reference-dir", type=Path)
    parser.add_argument("--a1-reference-manifest", type=Path)
    parser.add_argument("--uni-audit", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.stage in ("a1-test", "uni"):
        if args.output_root.exists():
            raise FileExistsError(f"Choose a new output root: {args.output_root}")
        args.output_root.mkdir(parents=True)
    elif not (args.output_root / "Phase4_1_UNI_A1/COMPLETE.json").is_file():
        raise ValueError("UNI feature validation must complete before spatial validation")
    if (args.stage == "spatial-b" and not
            (args.output_root / "Phase4_2a_relative_all/gamma_tuning_summary.json").is_file()):
        raise ValueError("Experiment 2(a) must complete before 2(b)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "a1-test":
        if (args.uniscale_prepared is None or
                (args.a1_reference_bundle is None) == (args.a1_reference_dir is None)):
            raise ValueError("A1 test requires prepared uniscale data and one reference source")
        uniscale_rows, _ = base.load_manifest(args.uniscale_prepared)
        if digest(args.uniscale_prepared / "slides.csv") != UNISCALE_MANIFEST_SHA256:
            raise ValueError("Uniscale manifest differs from A1 training data")
        if args.a1_reference_dir is not None:
            if args.a1_reference_manifest is None:
                raise ValueError("An A1 reference directory requires its checksum manifest")
            reference = checked_reference_directory(
                args.a1_reference_dir, args.a1_reference_manifest,
                args.output_root / "a1_reference")
        else:
            reference = checked_reference(args.a1_reference_bundle,
                                          args.output_root / "a1_reference")
        print("Archived A1 alpha=0.25 official test evaluation", flush=True)
        a1_test = evaluate_a1_reference(reference, args.uniscale_prepared, uniscale_rows,
                                        args.output_root / "A1_uniscale_test", device)
        base.write_json(args.output_root / "COMPLETE.json", {
            "status": "complete", "a1_test": a1_test, "archived_checkpoints_tested": 15})
        print("A1 TEST RESULT", json.dumps(a1_test, indent=2), flush=True)
        return
    if args.uni_prepared is None or args.uni_audit is None:
        raise ValueError("Phase-4 validation requires prepared UNI data and coordinate metadata")
    uni_rows, _ = load_uni_manifest(args.uni_prepared)
    metadata = load_coordinate_metadata(args.uni_audit, uni_rows)
    if args.stage == "uni":
        print("Phase 4 experiment 1: UNI features with A1 selector", flush=True)
        result = train_uni_variant("Phase4_1_UNI_A1", "none", 0.0, uni_rows, args.uni_prepared,
                                   metadata, args.output_root / "Phase4_1_UNI_A1", device)
        print("UNI VALIDATION RESULT", json.dumps(result, indent=2), flush=True)
        return
    if args.stage == "spatial-a":
        print("Phase 4 experiment 2(a): relative-distance attention over all patches", flush=True)
        result = run_gamma_family("Phase4_2a_relative_all", "relative_all", uni_rows,
                                  args.uni_prepared, metadata,
                                  args.output_root / "Phase4_2a_relative_all", device)
        print("EXPERIMENT 2(a) GAMMA SELECTION", json.dumps(result, indent=2), flush=True)
        return
    print("Phase 4 experiment 2(b): relative-distance attention over top-32 biased scores", flush=True)
    spatial_b = run_gamma_family("Phase4_2b_relative_top32", "relative_top32", uni_rows,
                                 args.uni_prepared, metadata,
                                 args.output_root / "Phase4_2b_relative_top32", device)
    uni_cv = json.loads((args.output_root / "Phase4_1_UNI_A1/cv_summary.json").read_text())
    spatial_a = json.loads((args.output_root / "Phase4_2a_relative_all/gamma_tuning_summary.json").read_text())
    base.write_json(args.output_root / "COMPLETE.json", {
        "status": "complete", "sequence": ["Phase4_1_UNI_A1", "Phase4_2a_relative_all",
                                          "Phase4_2b_relative_top32"],
        "uni_cv": uni_cv, "spatial_a_tuning": spatial_a, "spatial_b_tuning": spatial_b,
        "total_new_fits": 15 * (1 + 2 * len(GAMMA_GRID)), "test_evaluated": False})
    print("ALL STEPS COMPLETE", args.output_root, flush=True)


if __name__ == "__main__":
    main()
