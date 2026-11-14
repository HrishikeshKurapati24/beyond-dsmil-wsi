# Designated code author: Hrishikesh Kurapati
# Scope/provenance: A1/A2 and K-by-lambda extension experiments; project contribution credited in the progress presentation.
# Attribution header added to this review copy; original source remains unchanged.

"""Single-run, validation-only CAMELYON16 A/B experiments on prepared uniscale bags.

The Phase 1-3 runner is imported unchanged for data validation and shared
utilities. This module has no official-test evaluation path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from torch import nn
from torch.nn import functional as F

import phase1_to_3_dsmil_runner as base


BASELINE_SHA256 = "32dc2a316278566e4c41dbe4fc01ca04e003c726beffcc8b7311f59f13cd2385"
PREPARED_SHA256 = "12d13547a03ce93dacf2087fe237f16f9a69d1ac2611c6a6b88dadab945ca604"
SEEDS = (42, 43, 44)
FOLDS = tuple(range(5))
EPOCHS, PATIENCE = 50, 10
LR, WEIGHT_DECAY = 0.0001, 0.001


@dataclass(frozen=True)
class Variant:
    name: str
    anchors: int
    alpha: float = 0.0
    fusion: str = "mean"
    diversity_weight: float = 0.0


NEW_VARIANTS = (
    Variant("A1_alpha_025", 4, alpha=0.25),
    Variant("A1_alpha_050", 4, alpha=0.50),
    Variant("A1_alpha_075", 4, alpha=0.75),
    Variant("A2_learned_fusion", 4, fusion="learned"),
    Variant("B_K2_L0", 2),
    Variant("B_K2_L01", 2, diversity_weight=0.1),
)
PHASE2 = Variant("Phase2_K4_L0", 4)
PHASE3 = Variant("Phase3_K4_L01", 4, diversity_weight=0.1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_diverse(feats: torch.Tensor, scores: torch.Tensor,
                   original_ids: torch.Tensor, alpha: float, anchors: int):
    """Return original-bag positions and rank records; selection is discrete."""
    if not 0.0 <= alpha <= 1.0 or len(feats) < 1:
        raise ValueError("Invalid alpha or empty bag")
    k = min(anchors, len(feats))
    if alpha == 0.0:
        indices = torch.topk(scores, k=k, sorted=True).indices
        ranks = [{"score_rank": rank + 1, "diversity_rank": None}
                 for rank in range(k)]
        return indices, ranks
    score_values = scores.detach().cpu().numpy()
    row_ids = original_ids.detach().cpu().numpy()
    ordering = np.lexsort((row_ids, -score_values))
    candidates = ordering[:min(32, len(feats))].tolist()
    chosen = [candidates[0]]
    rank_records = [{"score_rank": 1, "diversity_rank": None}]
    candidate_feats = F.normalize(feats.detach()[candidates], p=2, dim=1, eps=1e-12)
    cosine = (candidate_feats @ candidate_feats.T).cpu().numpy()
    candidate_pos = {index: position for position, index in enumerate(candidates)}
    while len(chosen) < k:
        remaining = [index for index in candidates if index not in chosen]
        score_ranks = {index: rank for rank, index in enumerate(remaining, 1)}
        distances = {index: 1.0 - float(max(cosine[candidate_pos[index], candidate_pos[j]]
                                            for j in chosen)) for index in remaining}
        diversity_order = sorted(remaining, key=lambda index: (-distances[index], int(row_ids[index])))
        diversity_ranks = {index: rank for rank, index in enumerate(diversity_order, 1)}
        winner = min(remaining, key=lambda index: (
            (1.0 - alpha) * score_ranks[index] + alpha * diversity_ranks[index],
            -float(score_values[index]), int(row_ids[index])))
        chosen.append(winner)
        rank_records.append({"score_rank": score_ranks[winner],
                             "diversity_rank": diversity_ranks[winner]})
    return torch.as_tensor(chosen, dtype=torch.long, device=feats.device), rank_records


class ABBClassifier(base.BClassifier):
    def __init__(self, variant: Variant):
        super().__init__(512, 1, anchors=variant.anchors)
        self.variant = variant
        self.fusion_u = nn.Parameter(torch.zeros(512)) if variant.fusion == "learned" else None

    def forward(self, feats, c, original_ids=None):
        if original_ids is None:
            original_ids = torch.arange(len(feats), device=feats.device)
        v = self.v(feats)
        q = self.q(feats).view(feats.shape[0], -1)
        indices, ranks = select_diverse(
            feats, c[:, 0], original_ids, self.variant.alpha, self.anchors)
        critical = torch.index_select(feats, dim=0, index=indices)
        q_max = self.q(critical)
        attention = torch.mm(q, q_max.transpose(0, 1))
        attention = F.softmax(
            attention / torch.sqrt(torch.tensor(q.shape[1], dtype=torch.float32, device=feats.device)), 0)
        anchor_bags = torch.mm(attention.transpose(0, 1), v)
        if self.variant.fusion == "learned":
            weights = F.softmax(anchor_bags @ self.fusion_u, dim=0)
            fused = torch.sum(weights[:, None] * anchor_bags, dim=0, keepdim=True)
        else:
            weights = torch.full((len(indices),), 1.0 / len(indices), device=feats.device)
            fused = anchor_bags.mean(dim=0, keepdim=True)
        bag = fused.view(1, c.shape[1], v.shape[1])
        prediction = self.fcc(bag).view(1, -1)
        details = {"indices": indices, "rank_records": ranks,
                   "anchor_bags": anchor_bags, "fusion_weights": weights}
        return prediction, attention, bag, details


class ABMILNet(nn.Module):
    def __init__(self, variant: Variant):
        super().__init__()
        self.i_classifier = base.FCLayer(512, 1)
        self.b_classifier = ABBClassifier(variant)

    def forward(self, x, original_ids=None):
        feats, classes = self.i_classifier(x)
        prediction, attention, bag, details = self.b_classifier(feats, classes, original_ids)
        return classes, prediction, attention, bag, details


def make_model(variant: Variant, device: torch.device):
    model = ABMILNet(variant).to(device)
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    # fusion_u is initialized to zero in ABBClassifier and is not a Linear module.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.9),
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 0.000005)
    return model, optimizer, scheduler


def bag_step(model, feats, label, criterion, original_ids=None):
    instance_logits, bag_logits, attention, _, details = model(feats, original_ids)
    max_logit = instance_logits.max(dim=0).values.view(1, 1)
    target = torch.tensor([[float(label)]], device=feats.device)
    loss = 0.5 * criterion(bag_logits.view(1, 1), target) + 0.5 * criterion(max_logit, target)
    probability = torch.sigmoid(bag_logits).item()
    return loss, probability, instance_logits, attention, details


def train_epoch(model, train, prepared, device, optimizer, criterion, shuffle_seed, weight):
    model.train()
    order = np.random.default_rng(shuffle_seed).permutation(len(train))
    classification_total = objective_total = overlap_total = 0.0
    for index in order:
        row = train[int(index)]
        feats = base.load_bag(prepared, row, device)
        permutation = torch.randperm(len(feats), device=device)
        feats = feats[permutation]
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, attention, _ = bag_step(model, feats, row["label"], criterion, permutation)
        overlap = base.attention_overlap(attention) if weight > 0 else attention.new_zeros(())
        objective = loss + weight * overlap
        if not torch.isfinite(objective):
            raise ValueError(f"Nonfinite training objective for {row['slide_id']}")
        objective.backward()
        optimizer.step()
        classification_total += float(loss.detach().cpu())
        objective_total += float(objective.detach().cpu())
        overlap_total += float(overlap.detach().cpu())
    count = len(train)
    return classification_total / count, objective_total / count, overlap_total / count


@torch.no_grad()
def predict(model, rows, prepared, device, criterion, diagnostics_enabled=True):
    model.eval()
    predictions, diagnostics = [], []
    total_loss = 0.0
    for row in rows:
        feats = base.load_bag(prepared, row, device)
        loss, probability, logits, attention, details = bag_step(model, feats, row["label"], criterion)
        total_loss += float(loss.detach().cpu())
        predictions.append({"slide_id": row["slide_id"], "label": row["label"],
                            "probability": probability})
        if not diagnostics_enabled:
            continue
        indices = details["indices"]
        k = len(indices)
        selected = F.normalize(feats[indices], p=2, dim=1, eps=1e-12)
        feature_cos = selected @ selected.T
        normalized_bags = F.normalize(details["anchor_bags"], p=2, dim=1, eps=1e-12)
        bag_cos = normalized_bags @ normalized_bags.T
        off = ~torch.eye(k, dtype=torch.bool, device=device)
        weights = details["fusion_weights"]
        diagnostics.append({
            "slide_id": row["slide_id"], "label": row["label"],
            "anchor_row_indices": indices.cpu().tolist(),
            "anchor_instance_logits": logits[indices, 0].cpu().tolist(),
            "anchor_rank_records": details["rank_records"],
            "mean_pairwise_anchor_cosine": float(feature_cos[off].mean().item()) if k > 1 else None,
            "mean_pairwise_attention_cosine": float(base.attention_overlap(attention).item()),
            "mean_pairwise_anchor_bag_cosine": float(bag_cos[off].mean().item()) if k > 1 else None,
            "fusion_weights": weights.cpu().tolist(),
            "fusion_weight_entropy": float((-(weights * weights.clamp_min(1e-12).log()).sum()).item()),
            "normalized_attention_entropy": (
                -(attention * attention.clamp_min(1e-12).log()).sum(dim=0) /
                np.log(max(len(feats), 2))).cpu().tolist(),
        })
    return total_loss / len(rows), predictions, diagnostics


def train_variant(variant: Variant, rows: list[dict], info: dict,
                  prepared: Path, root: Path, device: torch.device,
                  prepared_hash: str, runner_hash: str):
    out = root / variant.name
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")
    out.mkdir(parents=True)
    config = {"variant": variant.__dict__, "seeds": list(SEEDS), "folds": list(FOLDS),
              "feature_source": "uniscale", "feature_dim": 512,
              "epochs": EPOCHS, "patience": PATIENCE, "lr": LR, "weight_decay": WEIGHT_DECAY,
              "adam_betas": [0.5, 0.9], "scheduler": "CosineAnnealingLR",
              "eta_min": 0.000005, "checkpoint": "max validation AUC; tie by min validation loss",
              "probability": "sigmoid(bag logit)", "prepared_info": info,
              "prepared_manifest_sha256": prepared_hash, "runner_sha256": runner_hash,
              "baseline_sha256": BASELINE_SHA256, "device": str(device),
              "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                           "sklearn": sklearn.__version__}, "test_evaluated": False}
    base.write_json(out / "config.json", config)
    criterion = nn.BCEWithLogitsLoss()
    all_oof, runs, seed_means = [], [], []
    for seed in SEEDS:
        seed_root = out / f"seed_{seed}"
        seed_root.mkdir()
        seed_oof, fold_summaries = [], []
        for fold in FOLDS:
            train = [r for r in rows if r["official_split"] == "train" and r["validation_fold"] != fold]
            valid = [r for r in rows if r["official_split"] == "train" and r["validation_fold"] == fold]
            assert len(train) == 216 and len(valid) == 54
            base.seed_all(seed)
            model, optimizer, scheduler = make_model(variant, device)
            best_auc, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
            history = []
            fold_start = time.monotonic()
            for epoch in range(1, EPOCHS + 1):
                epoch_start = time.monotonic()
                train_loss, train_objective, train_overlap = train_epoch(
                    model, train, prepared, device, optimizer, criterion,
                    seed + fold * 100000 + epoch, variant.diversity_weight)
                val_loss, predictions, diagnostics = predict(model, valid, prepared, device, criterion)
                val_auc = base.auc(predictions)
                val_acc = float(accuracy_score([r["label"] for r in predictions],
                                               [int(r["probability"] >= 0.5) for r in predictions]))
                epoch_row = {"fold": fold, "epoch": epoch, "train_loss": train_loss,
                             "train_objective": train_objective,
                             "train_attention_overlap": train_overlap,
                             "valid_loss": val_loss, "valid_auc": val_auc,
                             "valid_accuracy_0.5": val_acc, "lr": optimizer.param_groups[0]["lr"],
                             "elapsed_seconds": round(time.monotonic() - epoch_start, 2)}
                history.append(epoch_row)
                if not all(math.isfinite(float(value)) for value in (
                    train_loss, train_objective, val_loss, val_auc, val_acc)):
                    raise ValueError(f"Nonfinite history: {variant.name} seed={seed} fold={fold} epoch={epoch}")
                improved = (val_auc > best_auc + 1e-12 or
                            (abs(val_auc - best_auc) <= 1e-12 and val_loss < best_loss))
                if improved:
                    best_auc, best_loss, best_epoch, stale = val_auc, val_loss, epoch, 0
                    torch.save(model.state_dict(), seed_root / f"fold_{fold}.pt")
                    base.write_csv(seed_root / f"fold_{fold}_oof.csv",
                                   [{"fold": fold, **r} for r in predictions],
                                   ["fold", "slide_id", "label", "probability"])
                    base.write_json(seed_root / f"fold_{fold}_anchor_diagnostics.json",
                                    {"fold": fold, "epoch": epoch, "slides": diagnostics})
                else:
                    stale += 1
                scheduler.step()
                print(f"{variant.name} seed={seed} fold={fold} epoch={epoch} "
                      f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                      f"val_auc={val_auc:.4f} best={best_auc:.4f}", flush=True)
                if stale >= PATIENCE:
                    break
            base.write_csv(seed_root / f"fold_{fold}_history.csv", history, list(history[0]))
            selected = base.read_csv(seed_root / f"fold_{fold}_oof.csv")
            selected = [{"fold": int(r["fold"]), "slide_id": r["slide_id"],
                         "label": int(r["label"]), "probability": float(r["probability"])}
                        for r in selected]
            seed_oof.extend(selected)
            all_oof.extend({"seed": seed, **r} for r in selected)
            model.load_state_dict(torch.load(seed_root / f"fold_{fold}.pt",
                                             map_location=device, weights_only=True))
            train_eval_loss, train_predictions, _ = predict(
                model, train, prepared, device, criterion, diagnostics_enabled=False)
            train_acc = float(accuracy_score([r["label"] for r in train_predictions],
                                             [int(r["probability"] >= 0.5) for r in train_predictions]))
            labels = [r["label"] for r in selected]
            classes = [int(r["probability"] >= 0.5) for r in selected]
            tn, fp, fn, tp = confusion_matrix(labels, classes, labels=[0, 1]).ravel()
            fold_summary = {"fold": fold, "best_epoch": best_epoch,
                            "validation_auc": best_auc, "validation_loss": best_loss,
                            "training_loss_at_best_checkpoint": train_eval_loss,
                            "training_accuracy_0.5": train_acc,
                            "validation_accuracy_0.5": float(accuracy_score(labels, classes)),
                            "validation_balanced_accuracy_0.5": float(balanced_accuracy_score(labels, classes)),
                            "validation_sensitivity_0.5": float(tp / (tp + fn)),
                            "validation_specificity_0.5": float(tn / (tn + fp)),
                            "elapsed_seconds": round(time.monotonic() - fold_start, 1)}
            fold_summaries.append(fold_summary)
            runs.append({"seed": seed, **fold_summary})
            base.write_json(seed_root / f"fold_{fold}_summary.json", fold_summary)
            if len(runs) == 1:
                print(f"First fold took {fold_summary['elapsed_seconds']:.1f}s; "
                      f"90-fold linear projection {fold_summary['elapsed_seconds'] * 90 / 3600:.2f}h "
                      "(early stopping and variants may differ).", flush=True)
            del model, optimizer, scheduler
            if device.type == "cuda":
                torch.cuda.empty_cache()
        seed_oof.sort(key=lambda r: r["slide_id"])
        if len(seed_oof) != 270 or len({r["slide_id"] for r in seed_oof}) != 270:
            raise ValueError(f"Incomplete OOF output: {variant.name}, seed {seed}")
        base.write_csv(seed_root / "oof_predictions.csv", seed_oof,
                       ["fold", "slide_id", "label", "probability"])
        mean_auc = float(np.mean([r["validation_auc"] for r in fold_summaries]))
        seed_means.append(mean_auc)
        base.write_json(seed_root / "cv_summary.json",
                        {"folds": fold_summaries, "mean_validation_auc": mean_auc,
                         "std_validation_auc": float(np.std([r["validation_auc"] for r in fold_summaries], ddof=1)),
                         "oof_auc": base.auc(seed_oof), "test_evaluated": False})
    base.write_csv(out / "oof_predictions_all_seeds.csv", all_oof,
                   ["seed", "fold", "slide_id", "label", "probability"])
    ensemble = ensemble_oof(all_oof)
    base.write_csv(out / "oof_predictions_ensemble.csv", ensemble,
                   ["slide_id", "fold", "label", "probability"])
    summary = {"variant": variant.__dict__, "seeds": list(SEEDS), "runs": runs,
               "seed_mean_validation_auc": seed_means,
               "mean_validation_auc": float(np.mean(seed_means)),
               "std_seed_mean_validation_auc": float(np.std(seed_means, ddof=1)),
               "oof_ensemble_auc": base.auc(ensemble), "test_evaluated": False}
    base.write_json(out / "cv_summary.json", summary)
    verify_new_variant(out, variant, rows)
    return summary


def ensemble_oof(all_oof: list[dict]):
    grouped = {}
    for row in all_oof:
        grouped.setdefault(row["slide_id"], []).append(row)
    if len(grouped) != 270 or any(len(values) != 3 or
                                  {int(x["seed"]) for x in values} != set(SEEDS) or
                                  len({int(x["fold"]) for x in values}) != 1 or
                                  len({int(x["label"]) for x in values}) != 1
                                  for values in grouped.values()):
        raise ValueError("OOF must contain three seeds on the same fold for each of 270 slides")
    return [{"slide_id": sid, "fold": int(values[0]["fold"]),
             "label": int(values[0]["label"]),
             "probability": float(np.mean([float(x["probability"]) for x in values]))}
            for sid, values in sorted(grouped.items())]


def verify_new_variant(out: Path, variant: Variant, rows: list[dict]):
    run = json.loads((out / "cv_summary.json").read_text())
    oof = base.read_csv(out / "oof_predictions_all_seeds.csv")
    expected = {r["slide_id"]: (r["label"], r["validation_fold"])
                for r in rows if r["official_split"] == "train"}
    if len(run["runs"]) != 15 or len(oof) != 810:
        raise ValueError(f"Incomplete variant: {variant.name}")
    observed = set()
    for r in oof:
        key = (int(r["seed"]), r["slide_id"])
        if key in observed or key[0] not in SEEDS or r["slide_id"] not in expected:
            raise ValueError(f"Invalid OOF key: {key}")
        observed.add(key)
        if (int(r["label"]), int(r["fold"])) != expected[r["slide_id"]]:
            raise ValueError(f"OOF label/fold mismatch: {key}")
        p = float(r["probability"])
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"Invalid OOF probability: {key}")
    for seed in SEEDS:
        for fold in FOLDS:
            directory = out / f"seed_{seed}"
            for suffix in (".pt", "_history.csv", "_summary.json", "_oof.csv", "_anchor_diagnostics.json"):
                if not (directory / f"fold_{fold}{suffix}").is_file():
                    raise FileNotFoundError(f"Missing fold artifact: {variant.name} {seed} {fold} {suffix}")
            diag = json.loads((directory / f"fold_{fold}_anchor_diagnostics.json").read_text())
            if len(diag["slides"]) != 54:
                raise ValueError(f"Missing diagnostics: {variant.name} {seed} {fold}")
    if run["test_evaluated"] is not False:
        raise ValueError("Test evaluation found in validation output")


def read_reference(bundle: Path, expected_sha: str, rows: list[dict], prepared_hash: str):
    if sha256(bundle) != expected_sha:
        raise ValueError("Reference bundle SHA-256 mismatch")
    expected_slides = {r["slide_id"]: (r["label"], r["validation_fold"])
                       for r in rows if r["official_split"] == "train"}
    result = {}
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("reference_manifest.json"))
        if manifest["kind"] != "ab_phase2_phase3_validation_reference" or manifest["member_count"] != 72:
            raise ValueError("Invalid reference bundle manifest")
        if set(archive.namelist()) != set(manifest["member_sha256"]) | {"reference_manifest.json"}:
            raise ValueError("Unexpected or missing reference member")
        for name, digest in manifest["member_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Reference member checksum mismatch: {name}")
        for phase, variant in (("phase2", PHASE2), ("phase3", PHASE3)):
            phase_info = manifest["phases"][phase]
            if (phase_info["code_sha256"] != BASELINE_SHA256 or
                    phase_info["prepared_manifest_sha256"] != prepared_hash or
                    phase_info["anchors"] != 4 or
                    phase_info["diversity_weight"] != variant.diversity_weight):
                raise ValueError(f"Reference provenance mismatch: {phase}")
            config = json.loads(archive.read(f"{phase}/config.json"))
            if (config["seeds"] != list(SEEDS) or config["folds"] != 5 or
                    config["anchors"] != 4 or config["diversity_weight"] != variant.diversity_weight or
                    config["manifest_sha256"] != prepared_hash):
                raise ValueError(f"Reference root settings mismatch: {phase}")
            summary = json.loads(archive.read(f"{phase}/cv_summary.json"))
            if len(summary["runs"]) != 15 or summary["test_evaluated"] is not False:
                raise ValueError(f"Incomplete reference summary: {phase}")
            text = archive.read(f"{phase}/oof_predictions_all_seeds.csv").decode()
            oof = list(csv.DictReader(io.StringIO(text)))
            validate_oof(oof, expected_slides)
            diagnostics = []
            for seed in SEEDS:
                seed_config = json.loads(archive.read(f"{phase}/seed_{seed}/config.json"))
                if any(seed_config[key] != value for key, value in {
                    "seed": seed, "anchors": 4, "diversity_weight": variant.diversity_weight,
                    "epochs": EPOCHS, "patience": PATIENCE, "lr": LR,
                    "weight_decay": WEIGHT_DECAY, "manifest_sha256": prepared_hash,
                }.items()):
                    raise ValueError(f"Reference seed settings mismatch: {phase} {seed}")
                for fold in FOLDS:
                    f = json.loads(archive.read(f"{phase}/seed_{seed}/fold_{fold}_summary.json"))
                    matching = [r for r in summary["runs"] if r["seed"] == seed and r["fold"] == fold]
                    if len(matching) != 1 or matching[0]["validation_auc"] != f["validation_auc"]:
                        raise ValueError(f"Reference fold summary mismatch: {phase} {seed} {fold}")
                    diagnostic = json.loads(archive.read(
                        f"{phase}/seed_{seed}/fold_{fold}_anchor_diagnostics.json"))
                    if diagnostic["fold"] != fold or len(diagnostic["slides"]) != 54:
                        raise ValueError(f"Reference fold diagnostics missing: {phase} {seed} {fold}")
                    for slide in diagnostic["slides"]:
                        sid = slide["slide_id"]
                        if sid not in expected_slides or expected_slides[sid][1] != fold:
                            raise ValueError(f"Reference diagnostic slide mismatch: {sid}")
                        diagnostics.append({"seed": seed, "fold": fold,
                                            "label": expected_slides[sid][0], **slide})
            result[variant.name] = {"variant": variant, "summary": summary,
                                    "oof": oof, "diagnostics": diagnostics, "source": "archive"}
    if any(len(v["diagnostics"]) != 810 for v in result.values()):
        raise ValueError("Incomplete archived diagnostics")
    return result


def validate_oof(oof: list[dict], expected_slides: dict):
    if len(oof) != 810:
        raise ValueError("Expected 810 OOF rows")
    seen = set()
    for row in oof:
        seed, sid = int(row["seed"]), row["slide_id"]
        if seed not in SEEDS or sid not in expected_slides or (seed, sid) in seen:
            raise ValueError(f"Invalid OOF identity: {seed} {sid}")
        seen.add((seed, sid))
        if (int(row["label"]), int(row["fold"])) != expected_slides[sid]:
            raise ValueError(f"OOF label or fold mismatch: {seed} {sid}")
        p = float(row["probability"])
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"Invalid OOF probability: {seed} {sid}")


def load_new_result(root: Path, variant: Variant, rows: list[dict]):
    out = root / variant.name
    verify_new_variant(out, variant, rows)
    expected = {r["slide_id"]: (r["label"], r["validation_fold"])
                for r in rows if r["official_split"] == "train"}
    summary = json.loads((out / "cv_summary.json").read_text())
    oof = base.read_csv(out / "oof_predictions_all_seeds.csv")
    validate_oof(oof, expected)
    diagnostics = []
    for seed in SEEDS:
        for fold in FOLDS:
            data = json.loads((out / f"seed_{seed}" / f"fold_{fold}_anchor_diagnostics.json").read_text())
            for slide in data["slides"]:
                diagnostics.append({"seed": seed, "fold": fold, **slide})
    return {"variant": variant, "summary": summary, "oof": oof,
            "diagnostics": diagnostics, "source": "new"}


def preflight(prepared: Path, rows: list[dict], root: Path, device: torch.device):
    base.seed_all(42)
    reference = base.MILNet(anchors=4).to(device)
    new = ABMILNet(PHASE2).to(device)
    new.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(9, 512, device=device)
    for model in (reference, new):
        model.zero_grad(set_to_none=True)
    old_instance, old_bag, old_attention, old_vec = reference(x)
    new_instance, new_bag, new_attention, new_vec, _ = new(x)
    for left, right in ((old_instance, new_instance), (old_bag, new_bag),
                        (old_attention, new_attention), (old_vec, new_vec)):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
    old_loss = 0.5 * F.binary_cross_entropy_with_logits(old_bag, torch.ones_like(old_bag)) + 0.5 * F.binary_cross_entropy_with_logits(old_instance.max().view(1, 1), torch.ones_like(old_bag))
    new_loss = 0.5 * F.binary_cross_entropy_with_logits(new_bag, torch.ones_like(new_bag)) + 0.5 * F.binary_cross_entropy_with_logits(new_instance.max().view(1, 1), torch.ones_like(new_bag))
    old_loss.backward()
    new_loss.backward()
    for (name_old, old_param), (name_new, new_param) in zip(reference.named_parameters(), new.named_parameters()):
        assert name_old == name_new
        torch.testing.assert_close(old_param.grad, new_param.grad, rtol=1e-6, atol=1e-7)
    for alpha in (0.0, 0.25, 0.5, 0.75):
        for n in (1, 3, 7, 33):
            feats = torch.randn(n, 512, device=device)
            feats[0].zero_()
            scores = torch.ones(n, device=device) if n == 7 else torch.arange(n, device=device).float()
            original_ids = torch.arange(n - 1, -1, -1, device=device)
            selected, _ = select_diverse(feats, scores, original_ids, alpha, 4)
            assert len(selected) == min(4, n) and len(set(selected.cpu().tolist())) == len(selected)
            if alpha == 0.0:
                torch.testing.assert_close(selected, torch.topk(scores, k=min(4, n), sorted=True).indices)
    fusion = ABMILNet(NEW_VARIANTS[3]).to(device)
    fusion.load_state_dict(reference.state_dict(), strict=False)
    _, fused_logit, _, _, details = fusion(x)
    torch.testing.assert_close(fused_logit, old_bag.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(details["fusion_weights"], torch.full((4,), 0.25, device=device), rtol=0, atol=0)
    criterion = nn.BCEWithLogitsLoss()
    train = [r for r in rows if r["official_split"] == "train"]
    sampled = [min(train, key=lambda row: row["n_patches"]),
               max(train, key=lambda row: row["n_patches"])]
    observations = []
    for row in sampled:
        feats = base.load_bag(prepared, row, device)
        for variant in (NEW_VARIANTS[2], NEW_VARIANTS[3], NEW_VARIANTS[5]):
            base.seed_all(42)
            model, optimizer, scheduler = make_model(variant, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, attention, _ = bag_step(model, feats, row["label"], criterion)
            objective = loss + variant.diversity_weight * base.attention_overlap(attention)
            if not torch.isfinite(objective):
                raise ValueError(f"Nonfinite preflight objective: {variant.name} {row['slide_id']}")
            objective.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError(f"Nonfinite preflight gradient: {variant.name} {row['slide_id']}")
            checkpoint = root / "preflight_checkpoint.pt"
            torch.save(model.state_dict(), checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            checkpoint.unlink()
            observations.append({"variant": variant.name, "slide_id": row["slide_id"],
                                 "n_patches": len(feats), "objective": float(objective.item())})
            del model, optimizer, scheduler
            if device.type == "cuda":
                torch.cuda.empty_cache()
    base.write_json(root / "preflight.json", {"checks": observations,
                    "alpha0_exact_phase2_forward_and_gradients": True,
                    "fusion_zero_initially_uniform": True, "test_bags_loaded": False})


def mean_field(rows: list[dict], field: str, label=None):
    values = [float(r[field]) for r in rows if r.get(field) is not None and
              (label is None or int(r["label"]) == label)]
    return float(np.mean(values)) if values else None


def analyze(root: Path, results: dict):
    expected_keys = {PHASE2.name, PHASE3.name} | {v.name for v in NEW_VARIANTS}
    if set(results) != expected_keys:
        raise ValueError("Analysis requires six new configurations and two archives")
    identity = None
    comparison, all_ensembles = [], {}
    for name, data in results.items():
        runs = data["summary"]["runs"]
        if len(runs) != 15 or len(data["oof"]) != 810 or len(data["diagnostics"]) != 810:
            raise ValueError(f"Incomplete comparison input: {name}")
        if any(not math.isfinite(float(r["validation_auc"])) for r in runs):
            raise ValueError(f"Nonfinite fold AUC in {name}")
        keys = {(int(r["seed"]), r["slide_id"], int(r["label"]), int(r["fold"]))
                for r in data["oof"]}
        if identity is None:
            identity = keys
        elif keys != identity:
            raise ValueError(f"OOF cohorts differ: {name}")
        ensemble = ensemble_oof(data["oof"])
        all_ensembles[name] = ensemble
        labels = [r["label"] for r in ensemble]
        predictions = [int(r["probability"] >= 0.5) for r in ensemble]
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        weights = [r.get("fusion_weights") for r in data["diagnostics"]]
        weight_departure = [float(np.mean(np.abs(np.asarray(w) - 0.25)))
                            for w in weights if w is not None and len(w) == 4]
        comparison.append({
            "variant": name, "source": data["source"], "anchors": data["variant"].anchors,
            "alpha": data["variant"].alpha, "fusion": data["variant"].fusion,
            "diversity_weight": data["variant"].diversity_weight,
            "fold_seed_runs": len(runs),
            "mean_validation_auc": float(np.mean([r["validation_auc"] for r in runs])),
            "sd_validation_auc_descriptive": float(np.std([r["validation_auc"] for r in runs], ddof=1)),
            "mean_training_loss_at_best": mean_field(runs, "training_loss_at_best_checkpoint"),
            "mean_validation_loss_at_best": mean_field(runs, "validation_loss"),
            "mean_training_accuracy_0.5_at_best": mean_field(runs, "training_accuracy_0.5"),
            "mean_validation_accuracy_0.5_at_best": mean_field(runs, "validation_accuracy_0.5"),
            "oof_ensemble_auc": base.auc(ensemble),
            "oof_ensemble_accuracy_0.5": float(accuracy_score(labels, predictions)),
            "oof_ensemble_balanced_accuracy_0.5": float(balanced_accuracy_score(labels, predictions)),
            "oof_ensemble_sensitivity_0.5": float(tp / (tp + fn)),
            "oof_ensemble_specificity_0.5": float(tn / (tn + fp)),
            "positive_oof_mean_probability": mean_field(ensemble, "probability", 1),
            "negative_oof_mean_probability": mean_field(ensemble, "probability", 0),
            "mean_selected_feature_cosine": mean_field(data["diagnostics"], "mean_pairwise_anchor_cosine"),
            "mean_attention_cosine": mean_field(data["diagnostics"], "mean_pairwise_attention_cosine"),
            "mean_anchor_bag_cosine": mean_field(data["diagnostics"], "mean_pairwise_anchor_bag_cosine"),
            "positive_attention_cosine": mean_field(data["diagnostics"], "mean_pairwise_attention_cosine", 1),
            "negative_attention_cosine": mean_field(data["diagnostics"], "mean_pairwise_attention_cosine", 0),
            "mean_fusion_weight_entropy": mean_field(data["diagnostics"], "fusion_weight_entropy"),
            "mean_absolute_fusion_weight_departure_from_quarter": float(np.mean(weight_departure)) if weight_departure else None,
        })
    order = [PHASE2.name, *[v.name for v in NEW_VARIANTS], PHASE3.name]
    comparison.sort(key=lambda row: order.index(row["variant"]))
    base.write_csv(root / "comparison_table.csv", comparison, list(comparison[0]))
    slide_rows = [{"variant": name, "source": results[name]["source"],
                   "slide_id": row["slide_id"], "fold": row["fold"],
                   "label": row["label"], "probability": row["probability"],
                   "prediction_0.5": int(row["probability"] >= 0.5)}
                  for name in order for row in all_ensembles[name]]
    base.write_csv(root / "oof_predictions_all_variants.csv", slide_rows, list(slide_rows[0]))

    paired_contrasts = [
        ("A1_alpha_025_vs_Phase2", "A1_alpha_025", PHASE2.name),
        ("A1_alpha_050_vs_Phase2", "A1_alpha_050", PHASE2.name),
        ("A1_alpha_075_vs_Phase2", "A1_alpha_075", PHASE2.name),
        ("A2_vs_Phase2", "A2_learned_fusion", PHASE2.name),
        ("B_penalty_K2", "B_K2_L01", "B_K2_L0"),
        ("B_penalty_K4", PHASE3.name, PHASE2.name),
        ("B_K2_vs_K4_L0", "B_K2_L0", PHASE2.name),
        ("B_K2_vs_K4_L01", "B_K2_L01", PHASE3.name),
    ]
    run_map = {name: {(int(r["seed"]), int(r["fold"])): r for r in data["summary"]["runs"]}
               for name, data in results.items()}
    paired = []
    for contrast, left, right in paired_contrasts:
        for seed in SEEDS:
            for fold in FOLDS:
                delta = (float(run_map[left][seed, fold]["validation_auc"]) -
                         float(run_map[right][seed, fold]["validation_auc"]))
                paired.append({"contrast": contrast, "left": left, "right": right,
                               "seed": seed, "fold": fold, "auc_delta": delta})
    base.write_csv(root / "paired_fold_auc_differences.csv", paired, list(paired[0]))
    interactions = []
    for seed in SEEDS:
        for fold in FOLDS:
            k2 = (run_map["B_K2_L01"][seed, fold]["validation_auc"] -
                  run_map["B_K2_L0"][seed, fold]["validation_auc"])
            k4 = (run_map[PHASE3.name][seed, fold]["validation_auc"] -
                  run_map[PHASE2.name][seed, fold]["validation_auc"])
            interactions.append({"seed": seed, "fold": fold,
                                 "penalty_effect_K2": k2, "penalty_effect_K4": k4,
                                 "K_by_penalty_interaction": k2 - k4})
    base.write_csv(root / "B_interaction.csv", interactions, list(interactions[0]))
    alpha_candidates = [row for row in comparison if row["variant"] in
                        {PHASE2.name, "A1_alpha_025", "A1_alpha_050", "A1_alpha_075"}]
    selected = max(alpha_candidates, key=lambda row: (row["mean_validation_auc"], -row["alpha"]))
    alpha_report = {"selection_metric": "mean of 15 fold/seed validation AUC values",
                    "tie_rule": "smaller alpha", "selected_alpha": selected["alpha"],
                    "selected_variant": selected["variant"],
                    "candidates": [{"alpha": row["alpha"],
                                    "mean_validation_auc": row["mean_validation_auc"],
                                    "oof_ensemble_auc": row["oof_ensemble_auc"]}
                                   for row in alpha_candidates],
                    "test_evaluated": False}
    base.write_json(root / "A1_alpha_selection.json", alpha_report)
    report = {"scope": "validation only", "new_fold_seed_runs": 90,
              "reference_fold_seed_runs": 30, "test_evaluated": False,
              "alpha_selection": alpha_report,
              "A2_paired_mean_auc_delta_vs_Phase2": mean_field(
                  [r for r in paired if r["contrast"] == "A2_vs_Phase2"], "auc_delta"),
              "B_mean_interaction": mean_field(interactions, "K_by_penalty_interaction"),
              "interpretation": "Paired validation comparisons are descriptive; no official-test result is produced."}
    base.write_json(root / "AB_validation_summary.json", report)
    for name, ensemble in all_ensembles.items():
        if results[name]["source"] == "new":
            continue
        archive_out = root / "reference" / name
        archive_out.mkdir(parents=True, exist_ok=True)
        base.write_csv(archive_out / "oof_predictions_ensemble.csv", ensemble,
                       ["slide_id", "fold", "label", "probability"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    args = parser.parse_args()
    runner_hash = sha256(Path(__file__))
    if sha256(Path(base.__file__)) != BASELINE_SHA256:
        raise ValueError("The imported DSMIL baseline differs from the archived Phase 2/3 source")
    prepared_hash = sha256(args.prepared_root / "slides.csv")
    if prepared_hash != PREPARED_SHA256:
        raise ValueError(f"Prepared manifest differs from archived Phase 2/3: {prepared_hash}")
    rows, info = base.load_manifest(args.prepared_root)
    refs = read_reference(args.reference_bundle, args.reference_sha256, rows, prepared_hash)
    if args.output_root.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output_root}")
    root = args.output_root
    root.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {"kind": "pre_phase4_AB_validation", "prepared_manifest_sha256": prepared_hash,
              "baseline_sha256": BASELINE_SHA256, "runner_sha256": runner_hash,
              "reference_sha256": args.reference_sha256, "seeds": list(SEEDS),
              "folds": list(FOLDS), "new_variants": [v.__dict__ for v in NEW_VARIANTS],
              "reference_variants": [PHASE2.__dict__, PHASE3.__dict__],
              "device": str(device), "test_evaluated": False}
    base.write_json(root / "run_config.json", config)
    shutil.copy2(args.reference_bundle, root / "reference_bundle.zip")
    preflight(args.prepared_root, rows, root, device)
    for variant in NEW_VARIANTS:
        print(f"Starting {variant.name}: 3 seeds x 5 folds", flush=True)
        train_variant(variant, rows, info, args.prepared_root, root, device,
                      prepared_hash, runner_hash)
    results = {**refs}
    results.update({v.name: load_new_result(root, v, rows) for v in NEW_VARIANTS})
    report = analyze(root, results)
    checkpoints = list(root.glob("A*/seed_*/fold_*.pt")) + list(root.glob("B_K2*/seed_*/fold_*.pt"))
    histories = list(root.glob("A*/seed_*/fold_*_history.csv")) + list(root.glob("B_K2*/seed_*/fold_*_history.csv"))
    if len(checkpoints) != 90 or len(histories) != 90:
        raise ValueError("Missing checkpoints or histories in final output")
    if list(root.rglob("test_predictions*")) or list(root.rglob("test_summary.json")):
        raise ValueError("Test evaluation artifacts are forbidden in this run")
    base.write_json(root / "COMPLETE.json", {"status": "complete", "new_runs": 90,
                    "new_oof_rows": 4860, "reference_runs": 30,
                    "reference_sha256": args.reference_sha256,
                    "prepared_manifest_sha256": prepared_hash,
                    "test_evaluated": False})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
