# Designated code author: Hrishikesh Kurapati
# Scope/provenance: Multiscale A1 extension implementation.
# Attribution header added to this review copy; original source remains unchanged.

"""Fixed and gated multiscale A1 validation; frozen three-variant test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix
from torch import nn
from torch.nn import functional as F

import phase1_to_3_dsmil_runner as base
import pre_phase4_ab_runner as ab
import phase4_uni_spatial_runner as p4

SEEDS = (42, 43, 44)
FOLDS = range(5)
EPOCHS, PATIENCE = 50, 10
VARIANTS = ("M-fixed", "M-adaptive")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_multi(prepared):
    info = json.loads((prepared / "PREPROCESSING_COMPLETE.json").read_text())
    required = {"source": "multiscale", "feature_dim": 1024, "seed": 42,
                "folds": 5, "slides": 399, "train_slides": 270, "test_slides": 129,
                "multiscale_columns": {"20x": [0, 512], "5x": [512, 1024]}}
    if any(info.get(k) != v for k, v in required.items()):
        raise ValueError("Multiscale completion metadata mismatch")
    rows = base.read_csv(prepared / "slides.csv")
    if len(rows) != 399 or len({r["slide_id"] for r in rows}) != 399:
        raise ValueError("Expected 399 unique multiscale slides")
    for r in rows:
        r["label"] = int(r["label"])
        r["n_patches"] = int(r["n_patches"])
        if r["label"] not in (0, 1) or int(r["feature_dim"]) != 1024 or r["n_patches"] < 1:
            raise ValueError(f"Invalid multiscale row: {r['slide_id']}")
        rel = Path(r["bag_path"])
        if rel.is_absolute() or ".." in rel.parts or not (prepared / rel).is_file():
            raise ValueError(f"Missing or unsafe bag: {r['slide_id']}")
        if r["official_split"] == "train":
            r["validation_fold"] = int(r["validation_fold"])
            if not 0 <= r["validation_fold"] < 5 or r["slide_id"].startswith("test_"):
                raise ValueError("Invalid train fold")
        elif r["official_split"] == "test":
            if r["validation_fold"] or not r["slide_id"].startswith("test_"):
                raise ValueError("Invalid test fold")
        else:
            raise ValueError("Invalid split")
    if [sum(r["official_split"] == "train" and r["validation_fold"] == f for r in rows)
            for f in FOLDS] != [54] * 5:
        raise ValueError("Fold sizes mismatch")
    if sum(r["official_split"] == "test" for r in rows) != 129:
        raise ValueError("Test size mismatch")
    if next(r["label"] for r in rows if r["slide_id"] == "test_114") != 0:
        raise ValueError("test_114 label mismatch")
    return rows, info


def check_cohort(uniscale_rows, multi_rows):
    def signature(rows):
        return {r["slide_id"]: (r["label"], r["official_split"],
                                r["validation_fold"] if r["official_split"] == "train" else None)
                for r in rows}
    if signature(uniscale_rows) != signature(multi_rows):
        raise ValueError("Uniscale/multiscale cohort or folds differ")


def bag(prepared, row, device):
    a = np.load(prepared / row["bag_path"], mmap_mode="r", allow_pickle=False)
    if a.shape != (row["n_patches"], 1024) or a.dtype != np.dtype("float32"):
        raise ValueError(f"Invalid bag shape: {row['slide_id']}")
    x = torch.tensor(a, dtype=torch.float32, device=device)
    if not torch.isfinite(x).all():
        raise ValueError(f"Nonfinite bag: {row['slide_id']}")
    return x


class MultiA1(nn.Module):
    def __init__(self, adaptive=False):
        super().__init__()
        variant = ab.Variant("A1_alpha_025", 4, alpha=0.25)
        self.i_classifier = base.FCLayer(1024, 1)
        self.b_classifier = ab.ABBClassifier(variant)
        # The original ABBClassifier has 512-dimensional layers; replace only its input-facing modules.
        self.b_classifier.q = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(),
                                            nn.Linear(128, 128), nn.Tanh())
        self.b_classifier.fcc = nn.Conv1d(1, 1, kernel_size=1024)
        for layer in (self.i_classifier, self.b_classifier):
            for module in layer.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    nn.init.orthogonal_(module.weight)
                    nn.init.zeros_(module.bias)
        self.gate = nn.Linear(1024, 2) if adaptive else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        weights = None
        if self.gate is not None:
            weights = F.softmax(self.gate(x), dim=1)
            x = torch.cat((2 * weights[:, :1] * x[:, :512],
                           2 * weights[:, 1:] * x[:, 512:]), dim=1)
        feats, classes = self.i_classifier(x)
        prediction, attention, _, details = self.b_classifier(feats, classes)
        return classes, prediction, attention, details, weights


def make_model(adaptive, device):
    model = MultiA1(adaptive).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 5e-6)
    return model, optimizer, scheduler


def step(model, x, label, criterion):
    logits, bag_logit, attention, details, gate = model(x)
    target = torch.tensor([[float(label)]], device=x.device)
    loss = .5 * criterion(bag_logit.view(1, 1), target) + .5 * criterion(logits.max().view(1, 1), target)
    return loss, float(torch.sigmoid(bag_logit).item()), details, gate


@torch.no_grad()
def predict(model, rows, prepared, device, criterion, diagnostics=False):
    model.eval()
    records, extra, total = [], [], 0.0
    for r in rows:
        loss, prob, details, gate = step(model, bag(prepared, r, device), r["label"], criterion)
        records.append({"slide_id": r["slide_id"], "label": r["label"], "probability": prob})
        total += float(loss.item())
        if diagnostics:
            item = {"slide_id": r["slide_id"], "anchor_rows": details["indices"].cpu().tolist(),
                    "anchor_ranks": details["rank_records"]}
            if gate is not None:
                item["mean_gate_20x"] = float(gate[:, 0].mean().item())
                item["mean_gate_5x"] = float(gate[:, 1].mean().item())
                item["gate_20x_quantiles"] = torch.quantile(gate[:, 0],
                    torch.tensor([0., .25, .5, .75, 1.], device=device)).cpu().tolist()
                item["mean_gate_entropy"] = float((-(gate * gate.clamp_min(1e-12).log()).sum(1)).mean().item())
            extra.append(item)
    return total / len(rows), records, extra


def train(variant, prepared, out):
    if variant not in VARIANTS:
        raise ValueError(variant)
    rows, info = load_multi(prepared)
    out.mkdir(parents=True, exist_ok=False)
    config = {"variant": variant, "alpha": .25, "anchors": 4, "seeds": list(SEEDS),
              "folds": list(FOLDS), "epochs": EPOCHS, "patience": PATIENCE,
              "prepared_sha256": digest(prepared / "slides.csv"),
              "completion_sha256": digest(prepared / "PREPROCESSING_COMPLETE.json"),
              "runner_sha256": digest(__file__), "test_evaluated": False,
              "prepared_info": info}
    base.write_json(out / "config.json", config)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_oof, summaries = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            train_rows = [r for r in rows if r["official_split"] == "train" and r["validation_fold"] != fold]
            val_rows = [r for r in rows if r["official_split"] == "train" and r["validation_fold"] == fold]
            assert len(train_rows) == 216 and len(val_rows) == 54
            base.seed_all(seed)
            model, optimizer, scheduler = make_model(variant == "M-adaptive", device)
            run = out / f"seed_{seed}" / f"fold_{fold}"
            run.mkdir(parents=True)
            best_auc, best_loss, best_epoch, stale = -1., float("inf"), 0, 0
            history = []
            for epoch in range(1, EPOCHS + 1):
                model.train()
                order = np.random.default_rng(seed + fold * 100000 + epoch).permutation(len(train_rows))
                total = 0.
                for i in order:
                    row = train_rows[int(i)]
                    x = bag(prepared, row, device)
                    x = x[torch.randperm(len(x), device=device)]
                    optimizer.zero_grad(set_to_none=True)
                    loss, _, _, _ = step(model, x, row["label"], criterion)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                    total += float(loss.item())
                val_loss, predictions, diagnostics = predict(model, val_rows, prepared, device, criterion, True)
                auc = base.auc(predictions)
                acc = accuracy_score([r["label"] for r in predictions],
                                     [int(r["probability"] >= .5) for r in predictions])
                history.append({"epoch": epoch, "train_loss": total / len(train_rows),
                                "val_loss": val_loss, "val_auc": auc, "val_accuracy_0.5": float(acc),
                                "lr": optimizer.param_groups[0]["lr"]})
                if not all(math.isfinite(float(v)) for v in history[-1].values()):
                    raise ValueError("Nonfinite history")
                if auc > best_auc + 1e-12 or abs(auc - best_auc) <= 1e-12 and val_loss < best_loss:
                    best_auc, best_loss, best_epoch, stale = auc, val_loss, epoch, 0
                    torch.save(model.state_dict(), run / "checkpoint.pt")
                    base.write_csv(run / "oof.csv", [{"seed": seed, "fold": fold, **r} for r in predictions],
                                   ["seed", "fold", "slide_id", "label", "probability"])
                    base.write_json(run / "diagnostics.json", {"slides": diagnostics})
                else:
                    stale += 1
                scheduler.step()
                print(f"{variant} seed={seed} fold={fold} epoch={epoch} val_auc={auc:.4f}", flush=True)
                if stale >= PATIENCE:
                    break
            base.write_csv(run / "history.csv", history, list(history[0]))
            model.load_state_dict(torch.load(run / "checkpoint.pt", map_location=device, weights_only=True))
            train_loss, train_preds, _ = predict(model, train_rows, prepared, device, criterion)
            selected = base.read_csv(run / "oof.csv")
            all_oof.extend(selected)
            summary = {"seed": seed, "fold": fold, "best_epoch": best_epoch,
                       "validation_auc": best_auc, "validation_loss": best_loss,
                       "training_loss_at_best": train_loss,
                       "training_accuracy_0.5": float(accuracy_score(
                           [r["label"] for r in train_preds],
                           [int(r["probability"] >= .5) for r in train_preds])),
                       "validation_accuracy_0.5": float(accuracy_score(
                           [int(r["label"]) for r in selected],
                           [int(float(r["probability"]) >= .5) for r in selected]))}
            base.write_json(run / "summary.json", summary)
            summaries.append(summary)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(all_oof) != 810 or len(summaries) != 15:
        raise ValueError("Incomplete CV")
    base.write_csv(out / "oof_all.csv", all_oof, ["seed", "fold", "slide_id", "label", "probability"])
    by_slide = {}
    for r in all_oof:
        by_slide.setdefault(r["slide_id"], []).append(r)
    if len(by_slide) != 270 or any(len(v) != 3 for v in by_slide.values()):
        raise ValueError("OOF slide coverage mismatch")
    ensemble = [{"slide_id": sid, "label": int(v[0]["label"]),
                 "probability": float(np.mean([float(r["probability"]) for r in v]))}
                for sid, v in sorted(by_slide.items())]
    base.write_csv(out / "oof_predictions_ensemble.csv", ensemble,
                   ["slide_id", "label", "probability"])
    base.write_csv(out / "fold_results.csv", summaries, list(summaries[0]))
    base.write_json(out / "COMPLETE.json", {"status": "complete", "variant": variant,
                                          "fold_seed_runs": 15, "oof_rows": 810,
                                          "oof_auc": base.auc(ensemble), "test_evaluated": False})


def verify_a1_results(prior, multi_rows):
    config = json.loads((prior / "a1_reference/config.json").read_text())
    summary = json.loads((prior / "A1_uniscale_test/test_summary.json").read_text())
    complete = json.loads((prior / "COMPLETE.json").read_text())
    if (config["variant"] != ab.Variant("A1_alpha_025", 4, alpha=.25).__dict__ or
            config["seeds"] != list(SEEDS) or config["folds"] != list(FOLDS) or
            complete["status"] != "complete" or complete["archived_checkpoints_tested"] != 15 or
            summary["test_slides"] != 129 or summary["n_models"] != 15 or
            summary["test_114_label"] != 0):
        raise ValueError("A1 prior test metadata mismatch")
    oof = base.read_csv(prior / "a1_reference/oof_predictions_ensemble.csv")
    if len(oof) != 270 or len({r["slide_id"] for r in oof}) != 270:
        raise ValueError("A1 prior OOF coverage mismatch")
    train_expected = {r["slide_id"]: (r["label"], r["validation_fold"])
                      for r in multi_rows if r["official_split"] == "train"}
    if (set(train_expected) != {r["slide_id"] for r in oof} or
            any((int(r["label"]), int(r["fold"])) != train_expected[r["slide_id"]]
                for r in oof)):
        raise ValueError("A1 OOF labels or folds differ from multiscale manifest")
    threshold = base.threshold_from_oof([{"label": int(r["label"]),
        "probability": float(r["probability"])} for r in oof])
    if not math.isclose(threshold, summary["threshold_from_oof"], abs_tol=1e-12):
        raise ValueError("A1 test threshold does not match OOF")
    expected = {r["slide_id"]: r["label"] for r in multi_rows if r["official_split"] == "test"}
    ensemble = base.read_csv(prior / "A1_uniscale_test/test_predictions_ensemble.csv")
    single = base.read_csv(prior / "A1_uniscale_test/test_predictions_per_checkpoint.csv")
    if (len(expected) != 129 or len(ensemble) != 129 or len(single) != 1935 or
            {r["slide_id"] for r in ensemble} != set(expected)):
        raise ValueError("A1 prior test coverage mismatch")
    groups = {sid: {} for sid in expected}
    for r in single:
        sid, key = r["slide_id"], (int(r["seed"]), int(r["fold"]))
        p = float(r["probability"])
        if sid not in expected or int(r["label"]) != expected[sid] or key in groups[sid] or not 0 <= p <= 1:
            raise ValueError("Invalid A1 checkpoint prediction")
        groups[sid][key] = p
    required = {(s, f) for s in SEEDS for f in FOLDS}
    for r in ensemble:
        sid, p = r["slide_id"], float(r["probability"])
        if (int(r["label"]) != expected[sid] or set(groups[sid]) != required or
                not math.isclose(p, float(np.mean(list(groups[sid].values()))), abs_tol=1e-12) or
                int(r["prediction"]) != int(p >= threshold)):
            raise ValueError("A1 ensemble is inconsistent with checkpoint predictions")
    records = [{"label": int(r["label"]), "probability": float(r["probability"])} for r in ensemble]
    if not math.isclose(base.auc(records), summary["test_auc"], abs_tol=1e-12):
        raise ValueError("A1 prior test AUC mismatch")
    return summary


def verify_variant(variant, prepared, output):
    config = json.loads((output / "config.json").read_text())
    complete = json.loads((output / "COMPLETE.json").read_text())
    if (config["variant"] != variant or config["alpha"] != .25 or
            config["prepared_sha256"] != digest(prepared / "slides.csv") or
            config["completion_sha256"] != digest(prepared / "PREPROCESSING_COMPLETE.json") or
            config["runner_sha256"] != digest(__file__) or
            complete["fold_seed_runs"] != 15 or complete["test_evaluated"] is not False):
        raise ValueError(f"Invalid saved variant: {variant}")
    for seed in SEEDS:
        for fold in FOLDS:
            if not (output / f"seed_{seed}" / f"fold_{fold}" / "checkpoint.pt").is_file():
                raise ValueError("Missing checkpoint")
    oof = base.read_csv(output / "oof_predictions_ensemble.csv")
    if len(oof) != 270 or len({r["slide_id"] for r in oof}) != 270:
        raise ValueError("Incomplete OOF")
    return [{"slide_id": r["slide_id"], "label": int(r["label"]),
             "probability": float(r["probability"])} for r in oof]


def final_test(multi, prior, fixed, adaptive, out):
    mrows, _ = load_multi(multi)
    verify_a1_results(prior, mrows)
    verify_variant("M-fixed", multi, fixed)
    verify_variant("M-adaptive", multi, adaptive)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    results = []
    shutil.copytree(prior / "A1_uniscale_test", out / "U-uniscale")
    old_summary = json.loads((out / "U-uniscale" / "test_summary.json").read_text())
    old_ensemble = base.read_csv(out / "U-uniscale" / "test_predictions_ensemble.csv")
    old_summary["test_pr_auc"] = float(average_precision_score(
        [int(r["label"]) for r in old_ensemble], [float(r["probability"]) for r in old_ensemble]))
    old_summary["test_accuracy_0.5"] = float(accuracy_score(
        [int(r["label"]) for r in old_ensemble],
        [int(float(r["probability"]) >= .5) for r in old_ensemble]))
    base.write_json(out / "U-uniscale" / "test_summary.json", old_summary)
    results.append({"variant": "U-uniscale", **old_summary})
    test = [r for r in mrows if r["official_split"] == "test"]
    for variant, source in (("M-fixed", fixed), ("M-adaptive", adaptive)):
        oof = verify_variant(variant, multi, source)
        threshold = base.threshold_from_oof(oof)
        all_preds = []
        for seed in SEEDS:
            for fold in FOLDS:
                model = MultiA1(variant == "M-adaptive").to(device)
                model.load_state_dict(torch.load(source / f"seed_{seed}" / f"fold_{fold}" / "checkpoint.pt",
                                                 map_location=device, weights_only=True), strict=True)
                _, preds, _ = predict(model, test, multi, device, criterion)
                all_preds.extend({"seed": seed, "fold": fold, **r} for r in preds)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        dest = out / variant
        dest.mkdir()
        summary = p4.write_test(dest, all_preds, test, threshold, variant)
        ensemble = base.read_csv(dest / "test_predictions_ensemble.csv")
        summary["test_pr_auc"] = float(average_precision_score(
            [int(r["label"]) for r in ensemble], [float(r["probability"]) for r in ensemble]))
        summary["test_accuracy_0.5"] = float(accuracy_score(
            [int(r["label"]) for r in ensemble], [int(float(r["probability"]) >= .5) for r in ensemble]))
        base.write_json(dest / "test_summary.json", summary)
        results.append({"variant": variant, **summary})
    base.write_csv(out / "test_comparison.csv", results, sorted(set().union(*(r.keys() for r in results))))
    base.write_json(out / "COMPLETE.json", {"status": "complete", "variants": [r["variant"] for r in results],
                                          "test_slides_each": 129, "models_per_variant": 15,
                                          "test_evaluated": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("validate_fixed", "validate_adaptive", "final_test"), required=True)
    ap.add_argument("--multi", type=Path, required=True)
    ap.add_argument("--prior-a1", type=Path)
    ap.add_argument("--fixed", type=Path)
    ap.add_argument("--adaptive", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.stage.startswith("validate"):
        train("M-fixed" if args.stage == "validate_fixed" else "M-adaptive", args.multi, args.output)
    else:
        final_test(args.multi, args.prior_a1, args.fixed, args.adaptive, args.output)


if __name__ == "__main__":
    main()
