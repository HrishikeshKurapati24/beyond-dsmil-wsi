"""Build separate A1 test and Phase-4 validation Kaggle notebooks."""

from __future__ import annotations

import base64
import hashlib
import json
import textwrap
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "obtained_results/ab_validation_single_run/ab-validation/A1_alpha_025"
BUNDLE = HERE / "a1_alpha025_test_reference.zip"
A1_NOTEBOOK = HERE / "A1_alpha025_official_test.ipynb"
PHASE4_NOTEBOOK = HERE / "04_a1_uni_spatial_validation.ipynb"
AUDIT = HERE.parent / "dataset_audits/camelyon16_uni_audit.json"


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def encoded_cell(name: str, payload: bytes, target: str) -> str:
    encoded = base64.b64encode(payload).decode()
    parts = "\n".join(f"    {json.dumps(piece)}" for piece in textwrap.wrap(encoded, 100))
    sha = hashlib.sha256(payload).hexdigest()
    return ("from pathlib import Path\nimport base64, hashlib\n\n"
            f"{name} = base64.b64decode(\n{parts}\n)\n"
            f"assert hashlib.sha256({name}).hexdigest() == {sha!r}\n"
            f"{name}_path = Path({target!r})\n"
            f"{name}_path.write_bytes({name})\n"
            f"del {name}\n"
            f"print('Verified {target}, SHA-256 {sha}')\n")


def source_cells() -> list[dict]:
    return [cell("code", "%%writefile /kaggle/working/" + name + "\n" +
                 (HERE / name).read_text())
            for name in ("baseline_dsmil.py", "ab_validation_runner.py",
                         "phase4_a1_runner.py")]


def reference_files_and_manifest() -> tuple[list[Path], dict]:
    paths = [SOURCE / "config.json", SOURCE / "cv_summary.json",
             SOURCE / "oof_predictions_ensemble.csv"]
    paths += [SOURCE / f"seed_{seed}" / f"fold_{fold}.pt"
              for seed in (42, 43, 44) for fold in range(5)]
    members = {str(path.relative_to(SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in paths}
    manifest = {"kind": "a1_alpha025_test_reference", "member_sha256": members,
                "source": "obtained_results/ab_validation_single_run/ab-validation/A1_alpha_025"}
    return paths, manifest


def build_bundle(paths: list[Path], manifest: dict) -> str:

    def add_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
        item = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
        item.compress_type = zipfile.ZIP_DEFLATED
        item.external_attr = 0o644 << 16
        archive.writestr(item, data, compresslevel=6)

    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in paths:
            add_bytes(archive, str(path.relative_to(SOURCE)), path.read_bytes())
        add_bytes(archive, "reference_manifest.json", json.dumps(manifest, indent=2).encode())
    return hashlib.sha256(BUNDLE.read_bytes()).hexdigest()


def build_a1_notebook(manifest: dict) -> None:
    cells = [cell("markdown", """# Archived A1 α=0.25: official CAMELYON16 test

Attach two inputs: the **prepared uniscale** dataset and the saved **A/B validation notebook output** containing `ab-validation/A1_alpha_025/`. Alternatively, upload `a1_alpha025_test_reference.zip` as a Kaggle Dataset and attach it; Kaggle expands ZIP inputs automatically. Choose **Save Version → Save & Run All** with output enabled. The 15 archived A1 checkpoints are read from the attached input, not embedded in this notebook. The small embedded checksum manifest verifies every checkpoint and metadata file before evaluation.

The notebook verifies the archived A1 configuration and the prepared dataset manifest, predicts the 129 official test slides with each checkpoint, averages 15 bag probabilities per slide, and applies a Youden J threshold calculated from the archived training-slide OOF ensemble. It does not retrain A1. Results are saved under `/kaggle/working/a1-alpha025-test/`.
"""), *source_cells(),
        cell("code", encoded_cell("a1_manifest", json.dumps(manifest, sort_keys=True).encode(),
                                  "/kaggle/working/a1_reference_manifest.json")),
        cell("code", '''from pathlib import Path
import json, subprocess, sys

prepared = []
for marker in Path("/kaggle/input").rglob("PREPROCESSING_COMPLETE.json"):
    info = json.loads(marker.read_text())
    if info.get("source") == "uniscale" and info.get("feature_dim") == 512:
        prepared.append(marker.parent)
assert len(prepared) == 1, f"Attach exactly one prepared uniscale dataset: {prepared}"
reference_roots = [path.parent for path in Path("/kaggle/input").rglob("config.json")
                   if path.parent.name == "A1_alpha_025"]
reference_roots += [path.parent for path in Path("/kaggle/input").rglob("reference_manifest.json")
                    if json.loads(path.read_text()).get("kind") == "a1_alpha025_test_reference"]
reference_roots = list(dict.fromkeys(reference_roots))
assert len(reference_roots) == 1, (
    "Attach the A/B validation notebook output or the A1 checkpoint dataset; "
    f"found A1 reference directories: {reference_roots}")
output_root = Path("/kaggle/working/a1-alpha025-test")
subprocess.run([sys.executable, "/kaggle/working/phase4_a1_runner.py",
                "--stage", "a1-test", "--uniscale-prepared", str(prepared[0]),
                "--a1-reference-dir", str(reference_roots[0]),
                "--a1-reference-manifest", str(a1_manifest_path),
                "--output-root", str(output_root)], check=True)
'''),
        cell("code", '''import pandas as pd
from IPython.display import display

complete = json.loads((output_root / "COMPLETE.json").read_text())
assert complete["status"] == "complete" and complete["archived_checkpoints_tested"] == 15
result = pd.read_csv(output_root / "A1_uniscale_test/test_predictions_ensemble.csv")
assert len(result) == 129
display(pd.DataFrame([complete["a1_test"]]))
print("Saved A1 test predictions and summary:", output_root)
''')]
    A1_NOTEBOOK.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n")


def build_phase4_notebook() -> None:
    audit = json.loads(AUDIT.read_text())
    train = [row for row in audit["slides"] if row["split"] == "official_train"]
    if len(train) != 270:
        raise ValueError("Expected coordinate metadata for 270 official training slides")
    compact = {"source_audit_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
               "slides": [{key: row[key] for key in (
                   "slide_id", "patches", "coordinate_shape", "level_dim",
                   "patch_size_px", "patch_level", "out_of_bounds_coordinate_rows")}
                   for row in train]}
    cells = [cell("markdown", """# Phase 4 validation: UNI A1 and relative spatial attention

Attach only the **prepared UNI** dataset, enable a GPU, and choose **Save Version → Save & Run All** with output enabled. This notebook runs (1) A1 α=0.25 on 1024-dimensional UNI embeddings, (2a) relative-distance attention over all patches, and (2b) relative-distance attention restricted to the top 32 combined-score patches per anchor. All runs use the same five folds and seeds 42, 43, 44. Experiments 2a and 2b each tune gamma over `0, 0.25, 0.5, 1.0`, yielding 135 total fold/seed fits. **The notebook never opens UNI test bags or produces test predictions.**

Both spatial experiments retain the top-32 instance-score candidate pool but select four anchors by `0.50 score-rank + 0.25 feature-dissimilarity-rank + 0.25 spatial-distance-rank`. The first anchor is the highest instance scorer. Attention uses `appearance_logit - gamma * min((distance_in_patch_widths / 8)^2, 1)`. In 2a, softmax covers all patches. In 2b, the model computes these **distance-biased logits first**, keeps the 32 highest for each anchor, applies masked softmax, and pools only those patches. The four anchor bag vectors are averaged, and the original half bag/half max-instance BCE objective remains. No attention-overlap penalty is applied.

Each selected fold checkpoint records feature dissimilarity and spatial distance for every pair of anchors on **all 216 training and 54 validation slides**. Gamma selection uses mean validation AUC over the 15 fits, with validation loss and smaller gamma as ties; all tried values are reported. The UNI models are newly initialized and do not load archived uniscale A1 weights. Patch sizes and level dimensions are embedded from the audited source HDF5 attributes for the **training slides only**. The source author states that coordinate and embedding rows have the same order. Output is under `/kaggle/working/phase4-a1-validation/`.
"""), *source_cells(),
        cell("code", encoded_cell("audit_metadata", json.dumps(compact, separators=(",", ":")).encode(),
                                  "/kaggle/working/uni_training_coordinate_metadata.json")),
        cell("code", '''from pathlib import Path
import json, subprocess, sys

prepared = []
for marker in Path("/kaggle/input").rglob("PREPROCESSING_COMPLETE.json"):
    info = json.loads(marker.read_text())
    if info.get("source") == "uni" and info.get("feature_dim") == 1024:
        prepared.append(marker.parent)
assert len(prepared) == 1, f"Attach exactly one prepared UNI dataset: {prepared}"
output_root = Path("/kaggle/working/phase4-a1-validation")
command = [sys.executable, "/kaggle/working/phase4_a1_runner.py",
           "--uni-prepared", str(prepared[0]),
           "--uni-audit", str(audit_metadata_path),
           "--output-root", str(output_root)]
subprocess.run(command + ["--stage", "uni"], check=True)
'''),
        cell("code", '''import pandas as pd
from IPython.display import display

uni = json.loads((output_root / "Phase4_1_UNI_A1/cv_summary.json").read_text())
assert uni["test_evaluated"] is False and len(uni["runs"]) == 15
display(pd.DataFrame([{"experiment": "UNI A1", "mean_fold_auc": uni["mean_validation_auc"],
                       "sd_fold_auc": uni["std_validation_auc"],
                       "oof_ensemble_auc": uni["oof_ensemble_auc"]}]))
'''),
        cell("code", '''subprocess.run(command + ["--stage", "spatial-a"], check=True)
spatial_a = json.loads((output_root / "Phase4_2a_relative_all/gamma_tuning_summary.json").read_text())
display(pd.read_csv(output_root / "Phase4_2a_relative_all/gamma_tuning_table.csv"))
print("Selected gamma for 2(a):", spatial_a["selected_gamma"])
'''),
        cell("code", '''subprocess.run(command + ["--stage", "spatial-b"], check=True)
spatial_b = json.loads((output_root / "Phase4_2b_relative_top32/gamma_tuning_summary.json").read_text())
display(pd.read_csv(output_root / "Phase4_2b_relative_top32/gamma_tuning_table.csv"))
print("Selected gamma for 2(b):", spatial_b["selected_gamma"])
'''),
        cell("code", '''complete = json.loads((output_root / "COMPLETE.json").read_text())
assert complete["status"] == "complete" and complete["total_new_fits"] == 135
assert complete["test_evaluated"] is False
summary = []
variants = [("Phase4_1_UNI_A1", "UNI A1")]
for family, title in [("Phase4_2a_relative_all", "2(a), all patches"),
                      ("Phase4_2b_relative_top32", "2(b), top 32")]:
    for gamma in (0.0, 0.25, 0.5, 1.0):
        variants.append((f"{family}/gamma_{int(round(gamma * 100)):03d}",
                         f"{title}, gamma={gamma:g}"))
for folder, title in variants:
    root = output_root / folder
    cv = json.loads((root / "cv_summary.json").read_text())
    oof = pd.read_csv(root / "oof_predictions_all_seeds.csv")
    assert cv["test_evaluated"] is False and len(cv["runs"]) == 15 and len(oof) == 810
    assert json.loads((root / "COMPLETE.json").read_text())["anchor_diagnostic_slide_evaluations"] == 4050
    assert not list(root.rglob("test_predictions*")) and not list(root.rglob("test_summary.json"))
    summary.append({"experiment": title, "mean_fold_auc": cv["mean_validation_auc"],
                    "sd_fold_auc": cv["std_validation_auc"],
                    "oof_ensemble_auc": cv["oof_ensemble_auc"]})
display(pd.DataFrame(summary))
print("Saved validation-only outputs:", output_root)
''')]
    PHASE4_NOTEBOOK.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    paths, manifest = reference_files_and_manifest()
    bundle_sha = build_bundle(paths, manifest)
    build_a1_notebook(manifest)
    build_phase4_notebook()
    print(A1_NOTEBOOK, PHASE4_NOTEBOOK, "A1 bundle SHA-256", bundle_sha)


if __name__ == "__main__":
    main()
