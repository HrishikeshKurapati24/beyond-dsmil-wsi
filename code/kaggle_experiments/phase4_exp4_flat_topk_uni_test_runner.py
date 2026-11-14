# Designated code authors: Srimanreddy and Hrishikesh Kurapati
# Scope/provenance: Srimanreddy baseline DSMIL; Hrishikesh Phase 4 model and experiment protocol.
# Attribution header added to this review copy; original source remains unchanged.

"""Evaluate the 15 frozen Phase-4 flat Top-K UNI checkpoints on CAMELYON16 test.

Authors: Srimanreddy (DSMIL baseline), Hrishikesh Kurapati (Phase-4 experiments).
The notebook `flat_topk_uni_official_test.ipynb` invokes this standalone script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score, roc_curve)
from torch import nn


SEEDS = (42, 43, 44)
FOLDS = tuple(range(5))
EXPECTED_RUNNER_SHA256 = "faeba55515d5c92d66de5714d9346f59b0851725e89b24c444751aa35e190d22"


class FlatTopKUNI(nn.Module):
    """Inference-equivalent to phase4_exp3_4_runner.FlatTopKUNI."""

    def __init__(self):
        super().__init__()
        self.i_classifier = nn.Module()
        self.i_classifier.fc = nn.Sequential(nn.Linear(1024, 1))
        self.b_classifier = nn.Module()
        self.b_classifier.q = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.Tanh())
        self.b_classifier.v = nn.Identity()
        self.b_classifier.fcc = nn.Conv1d(1, 1, kernel_size=1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.i_classifier.fc(x)[:, 0]
        values = scores.detach().cpu().numpy()
        order = np.lexsort((np.arange(len(values)), -values))[:min(4, len(values))]
        index = torch.as_tensor(order.copy(), device=x.device)
        q = self.b_classifier.q(x)
        attention = torch.softmax((q @ q[index].T) / math.sqrt(128), dim=0)
        anchor_bags = attention.T @ self.b_classifier.v(x)
        bag = anchor_bags.mean(dim=0).view(1, 1, 1024)
        return self.b_classifier.fcc(bag).view(())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover(inputs: Path) -> tuple[Path, Path]:
    prepared = []
    for marker in inputs.rglob("PREPROCESSING_COMPLETE.json"):
        info = json.loads(marker.read_text())
        if info.get("source") == "uni" and info.get("feature_dim") == 1024:
            prepared.append(marker.parent)
    references = []
    for path in inputs.rglob("cv_summary.json"):
        if json.loads(path.read_text()).get("variant") == "exp4_flat_topk_uni":
            references.append(path.parent)
    if len(prepared) != 1 or len(references) != 1:
        raise ValueError(f"Attach one prepared UNI dataset and one flat Top-K output: {prepared}, {references}")
    return prepared[0], references[0]


def validate(prepared: Path, reference: Path) -> tuple[list[dict], float]:
    cfg = json.loads((reference / "config.json").read_text())
    cv = json.loads((reference / "cv_summary.json").read_text())
    complete = json.loads((reference / "COMPLETE.json").read_text())
    info = json.loads((prepared / "PREPROCESSING_COMPLETE.json").read_text())
    if cfg.get("variant") != cv.get("variant") or cv.get("variant") != "exp4_flat_topk_uni":
        raise ValueError("Wrong checkpoint variant")
    if cfg.get("runner_sha256") != EXPECTED_RUNNER_SHA256:
        raise ValueError("Checkpoint runner differs from the verified model implementation")
    if cfg.get("prepared_manifest_sha256") != sha256(prepared / "slides.csv"):
        raise ValueError("Checkpoint/UNI manifest mismatch")
    if cfg.get("seeds") != list(SEEDS) or cfg.get("folds") != list(FOLDS):
        raise ValueError("Unexpected seed or fold protocol")
    if len(cv.get("runs", [])) != 15 or complete.get("fits") != 15 or complete.get("test_evaluated") is not False:
        raise ValueError("Incomplete validation reference")
    if any(info.get(k) != v for k, v in {"source": "uni", "feature_dim": 1024,
                                         "slides": 399, "train_slides": 270,
                                         "test_slides": 129}.items()):
        raise ValueError("Unexpected prepared UNI dataset")
    if any(not (reference / f"seed_{s}" / f"fold_{f}" / "checkpoint.pt").is_file()
           for s in SEEDS for f in FOLDS):
        raise FileNotFoundError("Attach the complete output with all 15 checkpoint.pt files")
    rows = read_csv(prepared / "slides.csv")
    train = [r for r in rows if r["official_split"] == "train"]
    test = [r for r in rows if r["official_split"] == "test"]
    if len(rows) != 399 or len(train) != 270 or len(test) != 129 or len({r["slide_id"] for r in rows}) != 399:
        raise ValueError("UNI slide manifest has incorrect coverage")
    if next(r["label"] for r in test if r["slide_id"] == "test_114") != "0":
        raise ValueError("Unexpected test_114 label")
    oof = read_csv(reference / "oof_predictions_ensemble.csv")
    by_id = {r["slide_id"]: r for r in train}
    if len(oof) != 270 or {r["slide_id"] for r in oof} != set(by_id):
        raise ValueError("Incomplete OOF predictions")
    if any(int(r["label"]) != int(by_id[r["slide_id"]]["label"]) or
           int(r["fold"]) != int(by_id[r["slide_id"]]["validation_fold"]) for r in oof):
        raise ValueError("OOF labels or fold assignments do not match")
    y = np.array([int(r["label"]) for r in oof])
    prob = np.array([float(r["probability"]) for r in oof])
    if not math.isclose(roc_auc_score(y, prob), cv["oof_ensemble_auc"], abs_tol=1e-12):
        raise ValueError("OOF AUC does not match saved CV summary")
    fpr, tpr, thresholds = roc_curve(y, prob)
    finite = np.flatnonzero(np.isfinite(thresholds))
    threshold = float(thresholds[finite[np.argmax((tpr - fpr)[finite])]])
    return test, threshold


def evaluate(prepared: Path, reference: Path, output: Path, test: list[dict], threshold: float) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    checkpoints = {}
    for seed in SEEDS:
        for fold in FOLDS:
            path = reference / f"seed_{seed}" / f"fold_{fold}" / "checkpoint.pt"
            model = FlatTopKUNI().to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
            model.eval()
            models.append((seed, fold, model))
            checkpoints[f"seed_{seed}/fold_{fold}/checkpoint.pt"] = sha256(path)
    per_checkpoint, ensemble = [], []
    with torch.inference_mode():
        for index, row in enumerate(sorted(test, key=lambda r: r["slide_id"]), 1):
            path = prepared / row["bag_path"]
            x = np.load(path, mmap_mode="r", allow_pickle=False)
            if x.shape != (int(row["n_patches"]), 1024) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError(f"Invalid UNI test bag: {row['slide_id']}")
            features = torch.tensor(x, dtype=torch.float32, device=device)
            probs = []
            for seed, fold, model in models:
                p = float(torch.sigmoid(model(features)))
                if not math.isfinite(p):
                    raise ValueError(f"Nonfinite probability: {row['slide_id']}")
                probs.append(p)
                per_checkpoint.append({"seed": seed, "fold": fold, "slide_id": row["slide_id"],
                                       "label": int(row["label"]), "probability": p})
            avg = float(np.mean(probs))
            ensemble.append({"slide_id": row["slide_id"], "label": int(row["label"]),
                             "probability": avg, "prediction": int(avg >= threshold)})
            print(f"{index:3d}/129 {row['slide_id']} ensemble_probability={avg:.6f}", flush=True)
            del features
    if len(per_checkpoint) != 1935 or len(ensemble) != 129:
        raise ValueError("Incomplete official-test predictions")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "test_predictions_per_checkpoint.csv", per_checkpoint,
              ["seed", "fold", "slide_id", "label", "probability"])
    write_csv(output / "test_predictions_ensemble.csv", ensemble,
              ["slide_id", "label", "probability", "prediction"])
    y = [r["label"] for r in ensemble]
    pred = [r["prediction"] for r in ensemble]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    summary = {"model": "Phase-4 flat Top-K UNI", "test_slides": 129, "n_models": 15,
               "test_auc": float(roc_auc_score(y, [r["probability"] for r in ensemble])),
               "test_accuracy": float(accuracy_score(y, pred)),
               "test_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
               "test_sensitivity": float(tp / (tp + fn)),
               "test_specificity": float(tn / (tn + fp)),
               "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
               "threshold_from_oof": threshold,
               "ensemble_rule": "mean of 15 frozen checkpoint bag probabilities",
               "prepared_manifest_sha256": sha256(prepared / "slides.csv"),
               "checkpoint_sha256": checkpoints}
    (output / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoint_sha256"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/flat-topk-uni-official-test"))
    args = parser.parse_args()
    prepared, reference = discover(args.input_root)
    test, threshold = validate(prepared, reference)
    print("Prepared:", prepared, "\nFrozen checkpoints:", reference,
          "\nOOF threshold:", threshold, flush=True)
    evaluate(prepared, reference, args.output_root, test, threshold)


if __name__ == "__main__":
    main()
