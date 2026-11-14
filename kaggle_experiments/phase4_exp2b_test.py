"""Evaluate frozen Phase-4 2(b), gamma=1, and plot representative test slides."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import baseline_dsmil as base
import phase4_a1_runner as p4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_reference(reference: Path, prepared: Path) -> tuple[dict, dict]:
    manifest = json.loads((reference / "reference_manifest.json").read_text())
    members = manifest["member_sha256"]
    required = {"config.json", "cv_summary.json", "oof_predictions_ensemble.csv",
                "test_coordinate_metadata.json"}
    required |= {f"seed_{s}/fold_{f}.pt" for s in p4.SEEDS for f in p4.FOLDS}
    if set(members) != required:
        raise ValueError("Reference must contain precisely the 15 trained checkpoints and metadata")
    for relative, expected in members.items():
        path = reference / relative
        if sha256(path) != expected:
            raise ValueError(f"Reference checksum mismatch: {relative}")
    config = json.loads((reference / "config.json").read_text())
    cv = json.loads((reference / "cv_summary.json").read_text())
    if (config["mode"] != "relative_top32" or config["gamma"] != 1.0 or
            config["feature_source"] != "uni" or config["feature_dim"] != 1024 or
            config["anchors"] != 4 or config["candidate_pool"] != 32 or
            config["attention_top_c"] != 32 or
            config["seeds"] != list(p4.SEEDS) or config["folds"] != list(p4.FOLDS) or
            cv["test_evaluated"] is not False or len(cv["runs"]) != 15 or
            config["runner_sha256"] != sha256(Path(p4.__file__)) or
            config["prepared_manifest_sha256"] != sha256(prepared / "slides.csv")):
        raise ValueError("Frozen validation configuration or prepared UNI manifest mismatch")
    return config, cv


def metadata_for_test(reference: Path, rows: list[dict]) -> dict:
    items = json.loads((reference / "test_coordinate_metadata.json").read_text())["slides"]
    by_id = {r["slide_id"]: r for r in items}
    test = [r for r in rows if r["official_split"] == "test"]
    if len(items) != 129 or len(by_id) != 129 or set(by_id) != {r["slide_id"] for r in test}:
        raise ValueError("Test coordinate metadata has incomplete coverage")
    for row in test:
        item = by_id[row["slide_id"]]
        if (item["patches"] != row["n_patches"] or
                item["coordinate_shape"] != [row["n_patches"], 2] or
                item["patch_level"] != 0 or item["patch_size_px"] <= 0 or
                len(item["level_dim"]) != 2 or min(item["level_dim"]) <= 0 or
                item["out_of_bounds_coordinate_rows"] != 0):
            raise ValueError(f"Invalid test coordinate metadata: {row['slide_id']}")
    return by_id


def validate_test_arrays(prepared: Path, test: list[dict]) -> None:
    for row in test:
        for key, expected_shape, dtype in (
                ("bag_path", (row["n_patches"], 1024), np.float32),
                ("coords_path", (row["n_patches"], 2), np.int64)):
            relative = Path(row[key])
            if (not row[key] or relative.is_absolute() or ".." in relative.parts or
                    not (prepared / relative).is_file()):
                raise ValueError(f"Missing or unsafe test {key}: {row['slide_id']}")
            array = np.load(prepared / relative, mmap_mode="r", allow_pickle=False)
            if array.shape != expected_shape or array.dtype != dtype:
                raise ValueError(f"Invalid test {key} array: {row['slide_id']}")


def oof_threshold(reference: Path, rows: list[dict]) -> float:
    oof = base.read_csv(reference / "oof_predictions_ensemble.csv")
    train = {r["slide_id"]: r for r in rows if r["official_split"] == "train"}
    if len(oof) != 270 or {r["slide_id"] for r in oof} != set(train):
        raise ValueError("Incomplete OOF ensemble")
    for row in oof:
        original = train[row["slide_id"]]
        if (int(row["label"]) != original["label"] or
                int(row["fold"]) != original["validation_fold"]):
            raise ValueError(f"OOF label or fold mismatch: {row['slide_id']}")
        row["label"] = int(row["label"])
        row["probability"] = float(row["probability"])
        if not 0 <= row["probability"] <= 1 or not math.isfinite(row["probability"]):
            raise ValueError(f"Invalid OOF probability: {row['slide_id']}")
    return base.threshold_from_oof(oof)


@torch.no_grad()
def predict_one(model, prepared: Path, row: dict, metadata: dict, device):
    features, coords = p4.load_uni(prepared, row, device, metadata, spatial=True)
    model.eval()
    _, bag_logit, attention, details = model(features, coords=coords)
    probability = float(torch.sigmoid(bag_logit).item())
    if not math.isfinite(probability):
        raise ValueError(f"Invalid probability: {row['slide_id']}")
    return probability, attention, details


def choose_examples(ensemble: list[dict]) -> tuple[list[dict], list[str]]:
    # Fixed, documented selection; test labels are used only to choose figures after scoring.
    groups = [
        ("tumor_correct", [r for r in ensemble if r["label"] == 1 and r["prediction"] == 1],
         lambda r: (-r["probability"], r["slide_id"])),
        ("tumor_missed", [r for r in ensemble if r["label"] == 1 and r["prediction"] == 0],
         lambda r: (r["probability"], r["slide_id"])),
        ("normal_correct", [r for r in ensemble if r["label"] == 0 and r["prediction"] == 0],
         lambda r: (r["probability"], r["slide_id"])),
    ]
    chosen, missing = [], []
    for category, candidates, key in groups:
        if not candidates:
            missing.append(category)
            continue
        chosen.append({"category": category, **sorted(candidates, key=key)[0]})
    return chosen, missing


def plot_slide(path: Path, slide: dict, raw_coords: np.ndarray, anchors: list[int],
               top_patches: list[list[dict]], checkpoint: str, threshold: float) -> None:
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    fig, grid = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    axes = grid.ravel()
    for ax in axes[:5]:
        ax.scatter(raw_coords[:, 0], raw_coords[:, 1], s=1.5, c="#c5ccd3", alpha=.4,
                   linewidths=0, rasterized=True)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xlabel("Source x coordinate (pixels)")
        ax.set_ylabel("Source y coordinate (pixels)")
    axes[0].set_title("Four selected anchors")
    for index, anchor in enumerate(anchors):
        xy = raw_coords[anchor]
        for ax in (axes[0], axes[index + 1]):
            ax.scatter([xy[0]], [xy[1]], s=165, c=colors[index], marker="*",
                       edgecolors="black", linewidths=.6, zorder=5)
            ax.annotate(f"A{index + 1}", xy, xytext=(5, 5), textcoords="offset points",
                        fontsize=10, fontweight="bold", color=colors[index])
        patches = top_patches[index]
        xy_top = raw_coords[[p["row_index"] for p in patches]]
        axes[index + 1].scatter(xy_top[:, 0], xy_top[:, 1], s=25 + 320 * np.array(
            [p["attention_weight"] for p in patches]), c=colors[index], alpha=.75,
            edgecolors="white", linewidths=.35)
        axes[index + 1].set_title(f"A{index + 1}: top {len(patches)} attended patches")
    axes[5].axis("off")
    axes[5].text(.02, .95, "Reading the maps", va="top", fontsize=14, fontweight="bold")
    axes[5].text(.02, .82, "Gray dots: all UNI patch coordinates\n"
                 "Colored stars: selected anchors\n"
                 "Colored circles: highest attention weights\n"
                 "Circle area scales with attention weight\n\n"
                 "Each A1–A4 panel uses the same reference checkpoint.\n"
                 "The classification uses all 15 checkpoints.", va="top", fontsize=11)
    category = slide["category"].replace("_", " ")
    fig.suptitle(f"{slide['slide_id']} | {category} | 15-model p(tumor)={slide['probability']:.3f} "
                 f"| threshold={threshold:.3f}", fontsize=13)
    fig.text(.5, .015, f"Anchors and attention from {checkpoint}; classification from the 15-model ensemble. "
             "Patch coordinates are a spatial map, not a histology image.", ha="center", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}")
    args.output.mkdir(parents=True)
    config, cv = verify_reference(args.reference, args.prepared)
    rows, _ = p4.load_uni_manifest(args.prepared)
    metadata = metadata_for_test(args.reference, rows)
    test = [r for r in rows if r["official_split"] == "test"]
    validate_test_arrays(args.prepared, test)
    threshold = oof_threshold(args.reference, rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating 15 frozen checkpoints on {len(test)} UNI test slides ({device})", flush=True)
    all_predictions = []
    for seed in p4.SEEDS:
        for fold in p4.FOLDS:
            model = p4.A1UniNet("relative_top32", 1.0).to(device)
            state = torch.load(args.reference / f"seed_{seed}" / f"fold_{fold}.pt",
                               map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)
            for row in test:
                probability, _, _ = predict_one(model, args.prepared, row, metadata, device)
                all_predictions.append({"seed": seed, "fold": fold, "slide_id": row["slide_id"],
                                        "label": row["label"], "probability": probability})
            print(f"Finished seed={seed}, fold={fold}", flush=True)
            del model, state
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = p4.write_test(args.output, all_predictions, test, threshold,
                            "Phase4 exp2b; UNI; gamma=1.0; archived validation checkpoints")
    ensemble = base.read_csv(args.output / "test_predictions_ensemble.csv")
    for row in ensemble:
        row["label"] = int(row["label"])
        row["probability"] = float(row["probability"])
        row["prediction"] = int(row["prediction"])
    examples, missing = choose_examples(ensemble)
    # This checkpoint is selected by validation AUC/loss alone, independent of test labels.
    representative = max(cv["runs"], key=lambda r: (r["validation_auc"],
                                                      -r["validation_loss"], -r["seed"], -r["fold"]))
    seed, fold = representative["seed"], representative["fold"]
    checkpoint = f"seed_{seed}/fold_{fold}.pt"
    model = p4.A1UniNet("relative_top32", 1.0).to(device)
    model.load_state_dict(torch.load(args.reference / checkpoint, map_location=device,
                                     weights_only=True), strict=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir()
    figure_rows = []
    by_id = {r["slide_id"]: r for r in test}
    for item in examples:
        row = by_id[item["slide_id"]]
        _, attention, details = predict_one(model, args.prepared, row, metadata, device)
        anchors = [int(v) for v in details["indices"].cpu().tolist()]
        top_patches = []
        for col in range(len(anchors)):
            weights = attention[:, col].detach().cpu().numpy()
            order = np.lexsort((np.arange(len(weights)), -weights))[:8]
            top_patches.append([{"row_index": int(i), "attention_weight": float(weights[i])}
                                for i in order])
        raw_coords = np.load(args.prepared / row["coords_path"], allow_pickle=False)
        image = figure_dir / f"{item['category']}_{item['slide_id']}.png"
        plot_slide(image, item, raw_coords, anchors, top_patches, checkpoint, threshold)
        figure_rows.append({**item, "representative_checkpoint": checkpoint,
                            "anchor_row_indices": anchors,
                            "anchor_coordinates_pixels": raw_coords[anchors].tolist(),
                            "top_attended_patches": [[{**p, "coordinates_pixels":
                                                        raw_coords[p["row_index"]].tolist()}
                                                       for p in group] for group in top_patches],
                            "figure": str(image.relative_to(args.output))})
    base.write_json(args.output / "visualization_examples.json", {
        "selection": "highest probability true positive, lowest probability false negative, "
                     "lowest probability true negative; ties by slide ID",
        "representative_checkpoint_rule": "maximum validation AUC, tie by lower validation loss",
        "coordinate_pairing": config["coordinate_pairing"],
        "missing_categories": missing,
        "examples": figure_rows})
    base.write_json(args.output / "COMPLETE.json", {"status": "complete", "test_summary": summary,
        "n_checkpoints": 15, "n_test_slides": 129, "figures": len(figure_rows),
        "missing_figure_categories": missing,
        "reference_manifest_sha256": sha256(args.reference / "reference_manifest.json"),
        "prepared_manifest_sha256": sha256(args.prepared / "slides.csv")})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
