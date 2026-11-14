"""Generate one Kaggle notebook for Phase 4 experiments 3 and 4."""
from __future__ import annotations

import base64
import hashlib
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "dataset_audits/camelyon16_uni_audit.json"
NOTEBOOK = HERE / "05_phase4_experiments_3_and_4.ipynb"


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update({"execution_count": None, "outputs": []})
    return item


def encoded_metadata(payload):
    raw = json.dumps(payload, separators=(",", ":")).encode()
    encoded = base64.b64encode(raw).decode()
    chunks = "\n".join(f"    {piece!r}" for piece in textwrap.wrap(encoded, 100))
    digest = hashlib.sha256(raw).hexdigest()
    return ("from pathlib import Path\nimport base64, hashlib\n"
            f"payload = base64.b64decode(\n{chunks}\n)\n"
            f"assert hashlib.sha256(payload).hexdigest() == {digest!r}\n"
            "audit_path = Path('/kaggle/working/uni_training_coordinate_metadata.json')\n"
            "audit_path.write_bytes(payload)\ndel payload\n")


def main():
    source = json.loads(AUDIT.read_text())
    train = [row for row in source["slides"] if row["split"] == "official_train"]
    assert len(train) == 270
    compact = {"source_audit_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
               "slides": [{key: row[key] for key in (
                   "slide_id", "patches", "coordinate_shape", "level_dim", "patch_size_px",
                   "patch_level", "out_of_bounds_coordinate_rows")} for row in train]}
    cells = [cell("markdown", """# Phase 4 experiments 3 and 4: UNI validation

Attach the **prepared CAMELYON16-UNI dataset** and enable a GPU. Choose **Save Version → Save & Run All** with output enabled. This notebook contains the independent-head spatial-anchor experiment 3 and the flat Top-K UNI reference plus hierarchical experiments 4(a), 4(b), and 4(c). All comparisons use the same five training-slide folds and seeds 42, 43, 44. **No UNI official-test feature file is opened and no official-test prediction is written.**

The editable `STAGE` below can be `"all"`, `"exp3"`, `"exp4"`, or an individual variant name printed in the final cell. The full sequence has **150 fold/seed fits** and may exceed one Kaggle session. To continue in a new session with this same notebook, attach the prior saved notebook **Output** as an additional input; its completed runs are copied and hash-checked before the remaining stage begins. Do not attach more than one previous output. The saved output is `/kaggle/working/phase4-exp3-exp4/`.

Experiment 3: one selected patch per independent head, hard eight-patch-width separation, optional soft repulsion, full-slide attention for each selected head, and mean bag fusion. Experiment 4: **one shared scorer** selects the four highest-scoring patches; the 4(a) spatial hierarchy partitions patches into microgroups and branch regions, 4(b) adds existential intermediate supervision, and 4(c) adds a bounded nested-path objective. These are separate model families; experiment 4 does not inherit experiment 3's heads or repulsion. See the accompanying implementation plans for equations and interpretation limits.
""")]
    for name in ("baseline_dsmil.py", "ab_validation_runner.py", "phase4_a1_runner.py",
                 "phase4_exp3_4_runner.py"):
        cells.append(cell("code", f"%%writefile /kaggle/working/{name}\n" + (HERE/name).read_text()))
    cells.append(cell("code", encoded_metadata(compact)))
    cells.append(cell("code", '''from pathlib import Path
import json, shutil, subprocess, sys
import numpy as np
import pandas as pd
import torch

# Use "exp3" or "exp4" for a staged Kaggle run; attach prior saved output to resume.
STAGE = "all"
GRID_WIDTH = 4  # change to 8 only after the training-coordinate occupancy pilot
VARIANT_NAMES = (
    "exp3_control", "exp3_spatial_lam_000", "exp3_spatial_lam_010",
    "exp3_spatial_lam_025", "exp3_spatial_lam_050", "exp4_flat_topk_uni",
    "exp4_4a_feature_assignment", "exp4_4a_spatial",
    "exp4_4b_intermediate", "exp4_4c_path")
assert STAGE in ("all", "exp3", "exp4", *VARIANT_NAMES) and GRID_WIDTH in (4, 8)

prepared = []
for marker in Path("/kaggle/input").rglob("PREPROCESSING_COMPLETE.json"):
    info = json.loads(marker.read_text())
    if info.get("source") == "uni" and info.get("feature_dim") == 1024:
        prepared.append(marker.parent)
assert len(prepared) == 1, f"Attach exactly one prepared UNI dataset: {prepared}"
assert torch.cuda.is_available(), "Enable a Kaggle GPU before the full experiment"
print("GPU:", torch.cuda.get_device_name(0))

# Coordinate-only occupancy check on small, median, and large training bags.
manifest = pd.read_csv(prepared[0] / "slides.csv")
audit = {r["slide_id"]: r for r in json.loads(audit_path.read_text())["slides"]}
train_manifest = manifest[manifest["official_split"] == "train"].sort_values("n_patches")
for row in (train_manifest.iloc[0], train_manifest.iloc[len(train_manifest)//2], train_manifest.iloc[-1]):
    raw = np.load(prepared[0] / row["coords_path"], mmap_mode="r")
    cells_xy = np.floor(np.asarray(raw, dtype=np.float32) /
                        audit[row["slide_id"]]["patch_size_px"] / GRID_WIDTH).astype(np.int32)
    _, sizes = np.unique(cells_xy, axis=0, return_counts=True)
    print(row["slide_id"], "patches", len(raw), "occupied microgroups", len(sizes),
          "median patches/group", float(np.median(sizes)), "max patches/group", int(sizes.max()))

# One representative forward/backward pass for each requested model family.
import time
sys.path.insert(0, "/kaggle/working")
import phase4_a1_runner as p4
import phase4_exp3_4_runner as experiments
sample = train_manifest.iloc[len(train_manifest)//2].to_dict()
device = torch.device("cuda")
features, coordinates = p4.load_uni(prepared[0], sample, device, audit, spatial=True)
pilots = []
if STAGE in ("all", "exp3") or STAGE.startswith("exp3_"):
    pilots.append(("experiment 3", experiments.MultiHeadSpatial(True, 0.1).to(device)))
if STAGE in ("all", "exp4") or STAGE.startswith("exp4_"):
    pilots.append(("experiment 4(c)", experiments.HierarchicalTopK(3, 0.5, GRID_WIDTH).to(device)))
for title, model in pilots:
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    _, trial_loss, _ = model(features, coordinates, label=int(sample["label"]), diagnostics=False)
    assert torch.isfinite(trial_loss)
    trial_loss.backward()
    torch.cuda.synchronize()
    print(title, "smoke loss", float(trial_loss.detach()),
          "seconds", round(time.monotonic() - start, 2),
          "peak GPU GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
    del model
del features, coordinates
torch.cuda.empty_cache()
'''))
    cells.append(cell("markdown", """Inspect the coordinate-only occupancy above before the full run. Change `GRID_WIDTH` in the preceding cell to 8 if the 4-width microgroups are too sparse; keep the chosen value fixed across every experiment-4 variant. Then run the remaining cells or use **Save Version → Save & Run All**."""))
    cells.append(cell("code", '''
output_root = Path("/kaggle/working/phase4-exp3-exp4")
prior = list(Path("/kaggle/input").rglob("PHASE4_EXPERIMENTS_3_4.json"))
assert len(prior) <= 1, f"Attach at most one previous notebook output: {prior}"
if prior:
    shutil.copytree(prior[0].parent, output_root, dirs_exist_ok=True)
command = [sys.executable, "/kaggle/working/phase4_exp3_4_runner.py",
           "--prepared", str(prepared[0]), "--audit", str(audit_path),
           "--output", str(output_root), "--stage", STAGE,
           "--grid-width", str(GRID_WIDTH)]
subprocess.run(command, check=True)
'''))
    cells.append(cell("code", '''import pandas as pd
from IPython.display import display

expected = {
    "exp3": ["exp3_control", "exp3_spatial_lam_000", "exp3_spatial_lam_010",
             "exp3_spatial_lam_025", "exp3_spatial_lam_050"],
    "exp4": ["exp4_flat_topk_uni", "exp4_4a_feature_assignment",
             "exp4_4a_spatial", "exp4_4b_intermediate", "exp4_4c_path"],
}
selected = (expected["exp3"] + expected["exp4"] if STAGE == "all" else
            expected[STAGE] if STAGE in expected else [STAGE])
summary = []
for name in selected:
    root = output_root / name
    cv = json.loads((root / "cv_summary.json").read_text())
    complete = json.loads((root / "COMPLETE.json").read_text())
    oof = pd.read_csv(root / "oof_predictions_all_seeds.csv")
    assert complete["status"] == "complete" and complete["fits"] == 15
    assert cv["test_evaluated"] is False and len(cv["runs"]) == 15 and len(oof) == 810
    assert not list(root.rglob("test_predictions*"))
    summary.append({"variant": name, "mean_validation_auc": cv["mean_validation_auc"],
                    "mean_validation_loss": cv["mean_validation_loss"],
                    "oof_ensemble_auc": cv["oof_ensemble_auc"]})
display(pd.DataFrame(summary))
if STAGE == "all":
    overall = json.loads((output_root / "COMPLETE.json").read_text())
    assert overall["status"] == "complete" and overall["fits"] == 150
    assert overall["test_evaluated"] is False
print("Saved validation-only results:", output_root)
'''))
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
              "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    if NOTEBOOK.stat().st_size >= 1_000_000:
        raise ValueError("Notebook source exceeds Kaggle's 1 MB limit")
    print(NOTEBOOK, NOTEBOOK.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
