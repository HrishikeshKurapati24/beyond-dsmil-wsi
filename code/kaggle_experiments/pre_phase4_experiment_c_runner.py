# Designated code author: Hrishikesh Kurapati
# Scope/provenance: Experiment C seed-sensitivity orchestration; uses the Srimanreddy baseline runner.
# Attribution header added to this review copy; original source remains unchanged.

"""Experiment C: ten fixed random seeds x five folds of the Phase 1 DSMIL baseline.

This orchestration layer imports the archived Phase 1 training implementation
unchanged. It never evaluates the official test split. Later optimization
diagnostics must be designed after reviewing this complete seed distribution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import phase1_to_3_dsmil_runner as baseline


# The current baseline_dsmil.py and archived Phase 1 copy have this same hash.
EXPECTED_BASELINE_SHA256 = "32dc2a316278566e4c41dbe4fc01ca04e003c726beffcc8b7311f59f13cd2385"
SEED_COUNT = 10
FOLD_COUNT = 5
TRAIN_COUNT = 270


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_seed_manifest(path: Path) -> tuple[dict, list[int]]:
    manifest = json.loads(path.read_text())
    seeds = manifest.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) != SEED_COUNT or
            any(type(seed) is not int or seed <= 0 or seed >= 2**31 for seed in seeds) or
            len(set(seeds)) != SEED_COUNT):
        raise ValueError("Seed manifest must contain ten distinct positive 31-bit integers")
    if manifest.get("experiment") != "pre_phase_4_experiment_c_baseline_seed_sensitivity":
        raise ValueError("Unexpected experiment in seed manifest")
    return manifest, seeds


def complete_seed_output(path: Path, seed: int, args, manifest_sha256: str) -> bool:
    if not path.exists():
        return False
    expected = [path / "config.json", path / "cv_summary.json", path / "oof_predictions.csv"]
    for fold in range(FOLD_COUNT):
        expected.extend([path / f"fold_{fold}.pt", path / f"fold_{fold}_history.csv",
                         path / f"fold_{fold}_summary.json", path / f"fold_{fold}_oof.csv"])
    if not all(file.is_file() for file in expected):
        raise RuntimeError(f"Partial seed output at {path}; use a fresh output directory or restore the missing files")
    config = json.loads((path / "config.json").read_text())
    if any((config[key] != value) for key, value in {
        "seed": seed, "anchors": 1, "diversity_weight": 0.0,
        "epochs": args.epochs, "patience": args.patience, "lr": args.lr,
        "weight_decay": args.weight_decay, "manifest_sha256": manifest_sha256,
    }.items()):
        raise ValueError(f"Existing seed output has different settings: {path}")
    report = json.loads((path / "cv_summary.json").read_text())
    oof = baseline.read_csv(path / "oof_predictions.csv")
    if len(report["folds"]) != FOLD_COUNT or len(oof) != TRAIN_COUNT or len({r["slide_id"] for r in oof}) != TRAIN_COUNT:
        raise ValueError(f"Incomplete fold summaries or OOF predictions: {path}")
    return True


def summarize(root: Path, seeds: list[int], manifest: dict, baseline_hash: str, manifest_hash: str) -> None:
    fold_rows = []
    stability_rows = []
    seed_rows = []
    all_oof = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        report = json.loads((seed_root / "cv_summary.json").read_text())
        for fold in report["folds"]:
            fold_rows.append({"seed": seed, **fold})
            history = baseline.read_csv(seed_root / f"fold_{fold['fold']}_history.csv")
            if not history:
                raise ValueError(f"Missing epoch history for seed={seed}, fold={fold['fold']}")
            stability_rows.append({
                "seed": seed, "fold": fold["fold"], "epochs_ran": len(history),
                "best_epoch": fold["best_epoch"],
                "train_loss_first": float(history[0]["train_loss"]),
                "train_loss_last": float(history[-1]["train_loss"]),
                "validation_loss_first": float(history[0]["valid_loss"]),
                "validation_loss_last": float(history[-1]["valid_loss"]),
                "validation_auc_first": float(history[0]["valid_auc"]),
                "validation_auc_best": fold["validation_auc"],
                "validation_auc_last": float(history[-1]["valid_auc"]),
                "validation_accuracy_0.5_at_best": fold["validation_accuracy_0.5"],
                "training_accuracy_0.5_at_best": fold["training_accuracy_0.5"],
            })
        seed_rows.append({"seed": seed, "mean_validation_auc": report["mean_validation_auc"],
                          "std_validation_auc": report["std_validation_auc"],
                          "oof_auc": report["oof_auc"],
                          "oof_accuracy_at_threshold": report["oof_accuracy_at_threshold"],
                          "mean_best_epoch": float(np.mean([f["best_epoch"] for f in report["folds"]]))})
        for row in baseline.read_csv(seed_root / "oof_predictions.csv"):
            all_oof.append({"seed": seed, "fold": int(row["fold"]),
                            "slide_id": row["slide_id"], "label": int(row["label"]),
                            "probability": float(row["probability"])})
    if (len(fold_rows) != SEED_COUNT * FOLD_COUNT or
            len(stability_rows) != SEED_COUNT * FOLD_COUNT or
            len(all_oof) != SEED_COUNT * TRAIN_COUNT):
        raise ValueError("Incomplete Experiment C fold results or OOF predictions")
    if any(not math.isfinite(r["validation_auc"]) or not math.isfinite(r["validation_loss"])
           for r in fold_rows) or any(not math.isfinite(r["probability"]) for r in all_oof):
        raise ValueError("Nonfinite Experiment C result")

    grouped = {}
    for row in all_oof:
        grouped.setdefault(row["slide_id"], []).append(row)
    if len(grouped) != TRAIN_COUNT or any(
        len(values) != SEED_COUNT or len({v["fold"] for v in values}) != 1 or
        len({v["label"] for v in values}) != 1 for values in grouped.values()
    ):
        raise ValueError("Every training slide must have one OOF prediction per seed, from the same fold")
    mean_oof = [{"slide_id": sid, "fold": values[0]["fold"], "label": values[0]["label"],
                 "probability": float(np.mean([v["probability"] for v in values]))}
                for sid, values in sorted(grouped.items())]
    baseline.write_csv(root / "fold_results.csv", fold_rows, list(fold_rows[0]))
    baseline.write_csv(root / "stability_diagnostics.csv", stability_rows, list(stability_rows[0]))
    baseline.write_csv(root / "seed_results.csv", seed_rows, list(seed_rows[0]))
    baseline.write_csv(root / "oof_predictions_all_seeds.csv", all_oof, list(all_oof[0]))
    baseline.write_csv(root / "oof_predictions_mean_10_seeds.csv", mean_oof, list(mean_oof[0]))
    summary = {
        "experiment": manifest["experiment"], "model": "Phase 1 single-anchor DSMIL baseline",
        "baseline_source_sha256": baseline_hash, "seed_manifest_sha256": manifest_hash,
        "seed_selection": manifest, "seeds": seeds, "folds_per_seed": FOLD_COUNT,
        "completed_runs": len(fold_rows), "test_evaluated": False,
        "fold_validation_auc_mean": float(np.mean([r["validation_auc"] for r in fold_rows])),
        "fold_validation_auc_sd": float(np.std([r["validation_auc"] for r in fold_rows], ddof=1)),
        "seed_mean_validation_auc_mean": float(np.mean([r["mean_validation_auc"] for r in seed_rows])),
        "seed_mean_validation_auc_sd": float(np.std([r["mean_validation_auc"] for r in seed_rows], ddof=1)),
        "mean_10_seed_oof_auc": baseline.auc(mean_oof),
        "interpretation_note": "Fold and seed metrics are descriptive; folds share training slides. Inspect all histories before defining weak-run criteria or changing optimization.",
    }
    baseline.write_json(root / "experiment_c_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    args = parser.parse_args()
    args.epochs, args.patience, args.lr, args.weight_decay = 50, 10, 0.0001, 0.001
    args.anchors, args.diversity_weight = 1, 0.0

    baseline_source = Path(baseline.__file__)
    baseline_hash = sha256(baseline_source)
    if baseline_hash != EXPECTED_BASELINE_SHA256:
        raise ValueError(f"Baseline source differs from archived Phase 1: {baseline_hash}")
    manifest, seeds = read_seed_manifest(args.seed_manifest)
    manifest_hash = sha256(args.seed_manifest)
    rows, info = baseline.load_manifest(args.prepared_root)
    prepared_manifest_hash = sha256(args.prepared_root / "slides.csv")
    device = baseline.torch.device("cuda" if baseline.torch.cuda.is_available() else "cpu")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    experiment_config = {
        "model": "Phase 1 single-anchor DSMIL baseline", "anchors": 1, "diversity_weight": 0.0,
        "epochs": args.epochs, "patience": args.patience, "lr": args.lr,
        "weight_decay": args.weight_decay, "seed_manifest_sha256": manifest_hash,
        "seeds": seeds, "folds": FOLD_COUNT, "baseline_source_sha256": baseline_hash,
        "prepared_input": str(args.prepared_root), "prepared_info": info,
        "prepared_manifest_sha256": prepared_manifest_hash, "device": str(device),
        "test_evaluation": "not included; review the complete CV distribution first",
    }
    config_file = root / "experiment_config.json"
    if config_file.is_file():
        if json.loads(config_file.read_text()) != experiment_config:
            raise ValueError("Output root belongs to a different Experiment C configuration")
    elif any(root.iterdir()):
        raise FileExistsError(f"Choose an empty Experiment C output root: {root}")
    else:
        baseline.write_json(config_file, experiment_config)
    baseline.write_json(root / "seed_manifest.json", manifest)

    for position, seed in enumerate(seeds, 1):
        seed_root = root / f"seed_{seed}"
        if complete_seed_output(seed_root, seed, args, prepared_manifest_hash):
            print(f"[{position}/{SEED_COUNT}] Seed {seed} already complete; reusing verified output", flush=True)
            continue
        single = copy.copy(args)
        single.seed = seed
        single.output_root = seed_root
        print(f"[{position}/{SEED_COUNT}] Training baseline seed={seed} across five folds", flush=True)
        baseline.train_cv_single_seed(single, rows, info, device)
        complete_seed_output(seed_root, seed, args, prepared_manifest_hash)
    summarize(root, seeds, manifest, baseline_hash, manifest_hash)


if __name__ == "__main__":
    main()
