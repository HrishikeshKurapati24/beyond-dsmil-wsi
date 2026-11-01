"""DSMIL baseline on prepared CAMELYON16 uniscale bags.

The model and optimizer follow dsmil.py/train_tcga.py in the repository.
The official test slides are only opened by --mode test, after CV finishes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torch import nn
from torch.nn import functional as F


class FCLayer(nn.Module):
    def __init__(self, in_size: int, out_size: int = 1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        return feats, self.fc(feats)


class BClassifier(nn.Module):
    def __init__(self, input_size: int, output_class: int = 1, anchors: int = 1):
        super().__init__()
        if output_class != 1 or anchors < 1:
            raise ValueError("This binary CAMELYON16 runner requires one class and at least one anchor")
        self.anchors = anchors
        self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(),
                               nn.Linear(128, 128), nn.Tanh())
        self.v = nn.Identity()
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):
        v = self.v(feats)
        q = self.q(feats).view(feats.shape[0], -1)
        if self.anchors == 1:
            # Exact released DSMIL critical-instance path.
            _, indices = torch.sort(c, 0, descending=True)
            critical = torch.index_select(feats, dim=0, index=indices[0, :])
        else:
            k = min(self.anchors, len(feats))
            indices = torch.topk(c[:, 0], k=k, sorted=True).indices
            critical = torch.index_select(feats, dim=0, index=indices)
        q_max = self.q(critical)
        attention = torch.mm(q, q_max.transpose(0, 1))
        attention = F.softmax(attention / torch.sqrt(torch.tensor(q.shape[1], dtype=torch.float32, device=feats.device)), 0)
        anchor_bags = torch.mm(attention.transpose(0, 1), v)
        bag = anchor_bags.mean(dim=0, keepdim=True).view(1, c.shape[1], v.shape[1])
        prediction = self.fcc(bag).view(1, -1)
        return prediction, attention, bag


class MILNet(nn.Module):
    def __init__(self, feature_dim: int = 512, anchors: int = 1):
        super().__init__()
        self.i_classifier = FCLayer(feature_dim, 1)
        self.b_classifier = BClassifier(feature_dim, 1, anchors=anchors)

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction, attention, bag = self.b_classifier(feats, classes)
        return classes, prediction, attention, bag


def make_model(device: torch.device, lr: float, weight_decay: float, epochs: int, anchors: int = 1):
    model = MILNet(anchors=anchors).to(device)
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9),
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0.000005)
    return model, optimizer, scheduler


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_manifest(prepared: Path) -> tuple[list[dict], dict]:
    completion = prepared / "PREPROCESSING_COMPLETE.json"
    if not completion.is_file():
        raise FileNotFoundError(f"Prepared dataset is incomplete: {completion}")
    info = json.loads(completion.read_text())
    if info.get("source") != "uniscale" or info.get("feature_dim") != 512 or info.get("seed") != 42:
        raise ValueError(f"Unexpected prepared source or split seed: {info}")
    rows = read_csv(prepared / "slides.csv")
    if len(rows) != 399 or len({r["slide_id"] for r in rows}) != 399:
        raise ValueError("Expected 399 unique slide rows")
    train = [r for r in rows if r["official_split"] == "train"]
    test = [r for r in rows if r["official_split"] == "test"]
    if len(train) != 270 or len(test) != 129:
        raise ValueError("Expected 270 official train and 129 official test slides")
    for row in rows:
        sid = row["slide_id"]
        row["label"] = int(row["label"])
        row["n_patches"] = int(row["n_patches"])
        if row["label"] not in (0, 1) or int(row["feature_dim"]) != 512:
            raise ValueError(f"Invalid slide label or dimension: {sid}")
        if Path(row["bag_path"]).is_absolute() or ".." in Path(row["bag_path"]).parts:
            raise ValueError(f"Unsafe bag path: {sid}")
        if not (prepared / row["bag_path"]).is_file():
            raise FileNotFoundError(f"Missing bag: {sid}")
        if row["official_split"] == "train":
            row["validation_fold"] = int(row["validation_fold"])
            if not 0 <= row["validation_fold"] < 5 or sid.startswith("test_"):
                raise ValueError(f"Invalid train fold: {sid}")
        elif row["validation_fold"] or not sid.startswith("test_"):
            raise ValueError(f"Invalid held-out slide: {sid}")
    if any(sum(r["validation_fold"] == fold for r in train) != 54 for fold in range(5)):
        raise ValueError("Expected 54 validation slides per fold")
    if next(r["label"] for r in rows if r["slide_id"] == "test_114") != 0:
        raise ValueError("test_114 must be normal per the chosen study protocol")
    return rows, info


def load_bag(prepared: Path, row: dict, device: torch.device) -> torch.Tensor:
    array = np.load(prepared / row["bag_path"], mmap_mode="r", allow_pickle=False)
    if array.shape != (row["n_patches"], 512) or array.dtype != np.dtype("float32"):
        raise ValueError(f"Invalid bag shape/dtype: {row['slide_id']}")
    # torch.tensor copies from the read-only memmap and transfers one bag at a time.
    return torch.tensor(array, dtype=torch.float32, device=device)


def bag_step(model: MILNet, feats: torch.Tensor, label: int, criterion):
    instance_logits, bag_logits, attention, _ = model(feats)
    max_logit = instance_logits.max(dim=0).values.view(1, 1)
    target = torch.tensor([[float(label)]], device=feats.device)
    loss = 0.5 * criterion(bag_logits.view(1, 1), target) + 0.5 * criterion(max_logit, target)
    probability = torch.sigmoid(bag_logits).item()
    return loss, probability, instance_logits, attention


def attention_overlap(attention: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine between anchor attention distributions."""
    k = attention.shape[1]
    if k < 2:
        return attention.new_zeros(())
    normalized = F.normalize(attention, p=2, dim=0)
    similarity = normalized.transpose(0, 1) @ normalized
    off_diagonal = ~torch.eye(k, dtype=torch.bool, device=attention.device)
    return similarity[off_diagonal].mean()


def train_epoch(model, rows, prepared, device, optimizer, criterion, seed, diversity_weight):
    model.train()
    order = np.random.default_rng(seed).permutation(len(rows))
    classification_total = 0.0
    objective_total = 0.0
    overlap_total = 0.0
    for index in order:
        row = rows[int(index)]
        feats = load_bag(prepared, row, device)
        feats = feats[torch.randperm(len(feats), device=device)]
        optimizer.zero_grad(set_to_none=True)
        classification_loss, _, _, attention = bag_step(model, feats, row["label"], criterion)
        overlap = attention_overlap(attention) if diversity_weight > 0 else attention.new_zeros(())
        objective = classification_loss + diversity_weight * overlap
        objective.backward()
        optimizer.step()
        classification_total += float(classification_loss.detach().cpu())
        objective_total += float(objective.detach().cpu())
        overlap_total += float(overlap.detach().cpu())
    return (classification_total / len(rows), objective_total / len(rows),
            overlap_total / len(rows))


@torch.no_grad()
def predict(model, rows, prepared, device, criterion):
    model.eval()
    predictions = []
    diagnostics = []
    total_loss = 0.0
    for row in rows:
        feats = load_bag(prepared, row, device)
        loss, probability, instance_logits, attention = bag_step(model, feats, row["label"], criterion)
        total_loss += float(loss.detach().cpu())
        predictions.append({"slide_id": row["slide_id"], "label": row["label"],
                            "probability": probability})
        if model.b_classifier.anchors > 1:
            k = min(model.b_classifier.anchors, len(feats))
            indices = torch.topk(instance_logits[:, 0], k=k, sorted=True).indices
            normalized = F.normalize(feats[indices], dim=1)
            similarities = normalized @ normalized.T
            off_diagonal = ~torch.eye(k, dtype=torch.bool, device=device)
            cosine_mean = float(similarities[off_diagonal].mean().item()) if k > 1 else None
            entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=0)
            entropy = entropy / np.log(max(len(feats), 2))
            diagnostics.append({"slide_id": row["slide_id"],
                                "anchor_row_indices": indices.cpu().tolist(),
                                "anchor_instance_logits": instance_logits[indices, 0].cpu().tolist(),
                                "mean_pairwise_anchor_cosine": cosine_mean,
                                "mean_pairwise_attention_cosine": float(attention_overlap(attention).item()),
                                "normalized_attention_entropy": entropy.cpu().tolist()})
    return total_loss / len(rows), predictions, diagnostics


def auc(rows: list[dict]) -> float:
    return float(roc_auc_score([r["label"] for r in rows], [r["probability"] for r in rows]))


def threshold_from_oof(rows: list[dict]) -> float:
    labels = [r["label"] for r in rows]
    scores = [r["probability"] for r in rows]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    candidates = np.flatnonzero(np.isfinite(thresholds))
    winner = candidates[np.argmax((tpr - fpr)[candidates])]
    return float(thresholds[winner])


def train_cv_single_seed(args, rows, info, device):
    prepared = args.prepared_root
    out = args.output_root
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output root for a new CV run: {out}")
    out.mkdir(parents=True)
    config = {"architecture": "DSMIL FCLayer+BClassifier" if args.anchors == 1 else "DSMIL mean-pooled Top-K anchor bags",
              "anchors": args.anchors,
              "diversity_weight": args.diversity_weight,
              "diversity_penalty": "mean pairwise cosine of L2-normalized anchor attention columns",
              "feature_source": "uniscale", "feature_dim": 512, "seed": args.seed,
              "epochs": args.epochs, "patience": args.patience, "lr": args.lr,
              "weight_decay": args.weight_decay, "device": str(device),
              "checkpoint_metric": "validation_auc", "decision_threshold": "Youden J on 270 OOF scores",
              "test_rule": "mean of five fold model probabilities; apply one OOF threshold",
              "prepared_input": str(prepared), "prepared_info": info,
              "manifest_sha256": hashlib.sha256((prepared / "slides.csv").read_bytes()).hexdigest()}
    write_json(out / "config.json", config)
    criterion = nn.BCEWithLogitsLoss()
    all_oof = []
    fold_summary = []
    for fold in range(5):
        train = [r for r in rows if r["official_split"] == "train" and r["validation_fold"] != fold]
        valid = [r for r in rows if r["official_split"] == "train" and r["validation_fold"] == fold]
        assert len(train) == 216 and len(valid) == 54
        seed_all(args.seed)
        model, optimizer, scheduler = make_model(device, args.lr, args.weight_decay, args.epochs, args.anchors)
        best_auc = -1.0
        best_loss = float("inf")
        best_epoch = 0
        stale = 0
        history = []
        fold_started = time.monotonic()
        for epoch in range(1, args.epochs + 1):
            started = time.monotonic()
            train_loss, train_objective, train_overlap = train_epoch(
                model, train, prepared, device, optimizer, criterion,
                args.seed + fold * 100000 + epoch, args.diversity_weight)
            valid_loss, predictions, diagnostics = predict(model, valid, prepared, device, criterion)
            valid_auc = auc(predictions)
            accuracy05 = float(accuracy_score([r["label"] for r in predictions],
                                                 [int(r["probability"] >= 0.5) for r in predictions]))
            row = {"fold": fold, "epoch": epoch, "train_loss": train_loss,
                   "train_objective": train_objective,
                   "train_attention_overlap": train_overlap,
                   "valid_loss": valid_loss, "valid_auc": valid_auc,
                   "valid_accuracy_0.5": accuracy05, "lr": optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": round(time.monotonic() - started, 2)}
            history.append(row)
            improved = valid_auc > best_auc + 1e-12 or (abs(valid_auc - best_auc) <= 1e-12 and valid_loss < best_loss)
            if improved:
                best_auc, best_loss, best_epoch, stale = valid_auc, valid_loss, epoch, 0
                torch.save(model.state_dict(), out / f"fold_{fold}.pt")
                selected = [{"fold": fold, **r} for r in predictions]
                write_csv(out / f"fold_{fold}_oof.csv", selected,
                          ["fold", "slide_id", "label", "probability"])
                if args.anchors > 1:
                    write_json(out / f"fold_{fold}_anchor_diagnostics.json",
                               {"fold": fold, "epoch": epoch, "slides": diagnostics})
            else:
                stale += 1
            scheduler.step()
            print(f"fold={fold} epoch={epoch} train_loss={train_loss:.4f} overlap={train_overlap:.4f} val_loss={valid_loss:.4f} val_auc={valid_auc:.4f} best={best_auc:.4f} sec={row['elapsed_seconds']:.1f}", flush=True)
            if stale >= args.patience:
                break
        write_csv(out / f"fold_{fold}_history.csv", history, list(history[0]))
        selected = read_csv(out / f"fold_{fold}_oof.csv")
        for r in selected:
            r["fold"] = int(r["fold"])
            r["label"] = int(r["label"])
            r["probability"] = float(r["probability"])
        all_oof.extend(selected)
        model.load_state_dict(torch.load(out / f"fold_{fold}.pt", map_location=device, weights_only=True))
        selected_train_loss, train_predictions, _ = predict(model, train, prepared, device, criterion)
        train_accuracy = float(accuracy_score([r["label"] for r in train_predictions],
                                              [int(r["probability"] >= 0.5) for r in train_predictions]))
        valid_labels = [r["label"] for r in selected]
        valid_classes = [int(r["probability"] >= 0.5) for r in selected]
        tn, fp, fn, tp = confusion_matrix(valid_labels, valid_classes, labels=[0, 1]).ravel()
        fold_summary.append({"fold": fold, "best_epoch": best_epoch,
                             "validation_auc": best_auc, "validation_loss": best_loss,
                             "training_loss_at_best_checkpoint": selected_train_loss,
                             "training_accuracy_0.5": train_accuracy,
                             "validation_accuracy_0.5": float(accuracy_score(valid_labels, valid_classes)),
                             "validation_balanced_accuracy_0.5": float(balanced_accuracy_score(valid_labels, valid_classes)),
                             "validation_sensitivity_0.5": float(tp / (tp + fn)),
                             "validation_specificity_0.5": float(tn / (tn + fp)),
                             "elapsed_seconds": round(time.monotonic() - fold_started, 1)})
        write_json(out / f"fold_{fold}_summary.json", fold_summary[-1])
        del model, optimizer, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    all_oof.sort(key=lambda r: r["slide_id"])
    if len(all_oof) != 270 or len({r["slide_id"] for r in all_oof}) != 270:
        raise ValueError("OOF predictions must cover the 270 train slides exactly once")
    write_csv(out / "oof_predictions.csv", all_oof, ["fold", "slide_id", "label", "probability"])
    threshold = threshold_from_oof(all_oof)
    summary = {"folds": fold_summary, "mean_validation_auc": float(np.mean([r["validation_auc"] for r in fold_summary])),
               "std_validation_auc": float(np.std([r["validation_auc"] for r in fold_summary], ddof=1)),
               "oof_auc": auc(all_oof), "oof_threshold": threshold,
               "oof_accuracy_at_threshold": float(accuracy_score([r["label"] for r in all_oof],
                                                                   [int(r["probability"] >= threshold) for r in all_oof])),
               "test_evaluated": False}
    write_json(out / "cv_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def train_cv(args, rows, info, device):
    out = args.output_root
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output root: {out}")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {
        "seeds": args.seeds, "folds": 5, "anchors": args.anchors,
        "diversity_weight": args.diversity_weight,
        "manifest_sha256": hashlib.sha256((args.prepared_root / "slides.csv").read_bytes()).hexdigest(),
        "test_ensemble": "arithmetic mean of all 15 seed-by-fold checkpoint probabilities",
        "threshold": "Youden J on three-seed averaged OOF probabilities",
    })
    all_oof = []
    seed_means = []
    runs = []
    for seed in args.seeds:
        single = copy.copy(args)
        single.seed = seed
        single.output_root = out / f"seed_{seed}"
        train_cv_single_seed(single, rows, info, device)
        report = json.loads((single.output_root / "cv_summary.json").read_text())
        seed_means.append(report["mean_validation_auc"])
        runs.extend({"seed": seed, **fold} for fold in report["folds"])
        all_oof.extend({"seed": seed, **r} for r in read_csv(single.output_root / "oof_predictions.csv"))
    if len(runs) != 5 * len(args.seeds) or len(all_oof) != 270 * len(args.seeds):
        raise ValueError("Incomplete repeated CV")
    write_csv(out / "oof_predictions_all_seeds.csv", all_oof,
              ["seed", "fold", "slide_id", "label", "probability"])
    grouped = {}
    for r in all_oof:
        grouped.setdefault(r["slide_id"], []).append(r)
    if len(grouped) != 270 or any(len(v) != len(args.seeds) or
                                  len({x["fold"] for x in v}) != 1 or
                                  len({x["label"] for x in v}) != 1 for v in grouped.values()):
        raise ValueError("Each training slide needs one OOF prediction per seed on the same fold")
    ensemble_oof = [{"slide_id": sid, "fold": values[0]["fold"],
                     "label": int(values[0]["label"]),
                     "probability": float(np.mean([float(x["probability"]) for x in values]))}
                    for sid, values in sorted(grouped.items())]
    write_csv(out / "oof_predictions_ensemble.csv", ensemble_oof,
              ["slide_id", "fold", "label", "probability"])
    threshold = threshold_from_oof(ensemble_oof)
    summary = {
        "seeds": args.seeds, "runs": runs, "seed_mean_validation_auc": seed_means,
        "mean_validation_auc": float(np.mean(seed_means)),
        "std_seed_mean_validation_auc": float(np.std(seed_means, ddof=1)),
        "fold_auc_by_seed": {str(fold): [next(x["validation_auc"] for x in runs
                                                  if x["seed"] == seed and x["fold"] == fold)
                                        for seed in args.seeds] for fold in range(5)},
        "oof_ensemble_auc": auc(ensemble_oof), "oof_ensemble_threshold": threshold,
        "test_evaluated": False,
    }
    write_json(out / "cv_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def evaluate_test(args, rows, device):
    prepared = args.prepared_root
    cv_root = args.cv_root
    summary = json.loads((cv_root / "cv_summary.json").read_text())
    config = json.loads((cv_root / "config.json").read_text())
    if len(summary["runs"]) != 5 * len(args.seeds) or config["seeds"] != args.seeds or config["manifest_sha256"] != hashlib.sha256((prepared / "slides.csv").read_bytes()).hexdigest() or config["anchors"] != args.anchors or config["diversity_weight"] != args.diversity_weight:
        raise ValueError("CV output and prepared manifest do not match")
    test = [r for r in rows if r["official_split"] == "test"]
    assert len(test) == 129
    out = args.output_root
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty test output root: {out}")
    out.mkdir(parents=True)
    per_fold = []
    criterion = nn.BCEWithLogitsLoss()
    for seed in args.seeds:
        for fold in range(5):
            checkpoint = cv_root / f"seed_{seed}" / f"fold_{fold}.pt"
            model = MILNet(anchors=args.anchors).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            _, predictions, _ = predict(model, test, prepared, device, criterion)
            per_fold.extend({"seed": seed, "fold": fold, **r} for r in predictions)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(per_fold) != 129 * 5 * len(args.seeds):
        raise ValueError("Incomplete test predictions")
    write_csv(out / "test_predictions_per_checkpoint.csv", per_fold,
              ["seed", "fold", "slide_id", "label", "probability"])
    by_slide = {r["slide_id"]: [] for r in test}
    for r in per_fold:
        by_slide[r["slide_id"]].append(r["probability"])
    if any(len(v) != 5 * len(args.seeds) for v in by_slide.values()):
        raise ValueError("Each test slide must have one prediction from every checkpoint")
    threshold = float(summary["oof_ensemble_threshold"])
    ensemble = [{"slide_id": r["slide_id"], "label": r["label"],
                 "probability": float(np.mean(by_slide[r["slide_id"]])),
                 "prediction": int(np.mean(by_slide[r["slide_id"]]) >= threshold)} for r in test]
    write_csv(out / "test_predictions_ensemble.csv", ensemble,
              ["slide_id", "label", "probability", "prediction"])
    labels = [r["label"] for r in ensemble]
    predictions = [r["prediction"] for r in ensemble]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    report = {"test_slides": 129, "ensemble": "mean of all seed-by-fold checkpoint probabilities",
              "n_models": 5 * len(args.seeds), "seeds": args.seeds,
              "threshold_from_oof": threshold, "test_auc": auc(ensemble),
              "test_accuracy": float(accuracy_score(labels, predictions)),
              "test_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
              "test_sensitivity": float(tp / (tp + fn)),
              "test_specificity": float(tn / (tn + fp)),
              "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
              "test_114_label": 0, "cv_summary": str(cv_root)}
    write_json(out / "test_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cv", "test"], default="cv")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cv-root", type=Path, help="Required for test mode")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--anchors", type=int, default=1,
                        help="1 reproduces baseline DSMIL; 4 is the Phase 2 Top-K experiment")
    parser.add_argument("--diversity-weight", type=float, default=0.0,
                        help="Phase 3 uses 0.1; 0 leaves Phases 1 and 2 unchanged")
    args = parser.parse_args()
    if args.anchors < 1:
        parser.error("--anchors must be >= 1")
    if args.diversity_weight < 0 or (args.diversity_weight > 0 and args.anchors < 2):
        parser.error("Diversity requires a nonnegative weight and at least two anchors")
    if args.mode == "test" and args.cv_root is None:
        parser.error("--cv-root is required for test mode")
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        parser.error("Provide exactly three distinct training seeds")
    rows, info = load_manifest(args.prepared_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seeds[0])
    if args.mode == "cv":
        train_cv(args, rows, info, device)
    else:
        evaluate_test(args, rows, device)


if __name__ == "__main__":
    main()
