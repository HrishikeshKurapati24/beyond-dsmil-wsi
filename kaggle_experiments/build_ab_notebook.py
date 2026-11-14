"""Build the self-contained A/B Kaggle notebook from the tested runner and reference."""

from __future__ import annotations

import base64
import hashlib
import json
import textwrap
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "AB_validation_single_run.ipynb"


def cell(kind: str, source: str):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def main() -> None:
    baseline = (HERE / "baseline_dsmil.py").read_text()
    runner = (HERE / "ab_validation_runner.py").read_text()
    bundle = (HERE / "ab_phase2_phase3_reference.zip").read_bytes()
    bundle_hash = hashlib.sha256(bundle).hexdigest()
    encoded = base64.b64encode(bundle).decode()
    quoted_chunks = "\n".join(f"    {json.dumps(chunk)}" for chunk in textwrap.wrap(encoded, 100))
    bundle_cell = f'''from pathlib import Path
import base64, hashlib

reference_bytes = base64.b64decode(
{quoted_chunks}
)
reference_hash = "{bundle_hash}"
assert hashlib.sha256(reference_bytes).hexdigest() == reference_hash
reference_path = Path("/kaggle/working/ab_phase2_phase3_reference.zip")
reference_path.write_bytes(reference_bytes)
print("Verified Phase 2/3 validation reference:", reference_hash)
'''
    notebook = {
        "cells": [
            cell("markdown", """# Pre-Phase-4 A/B validation experiments

Attach the versioned **prepared CAMELYON16 uniscale** dataset, enable a GPU, and choose **Save Version -> Save & Run All** with output enabled. This notebook runs six new configurations (90 fold/seed fits) across the same five folds and seeds 42/43/44. It embeds the previously completed Phase 2/3 validation results as checked references. It **does not retrain** the K=4 references and **does not evaluate the official test set**.

A1 uses diversity-aware anchor selection with alpha 0.25, 0.50, or 0.75 and **no attention penalty**. Alpha 0 is exactly the archived Phase 2 control. A2 learns how to combine four ordinary anchor bag vectors, also with **no attention penalty**. B compares K=2 with penalty weights 0 and 0.1 against the archived K=4 cells. Check the final `COMPLETE.json` and comparison tables under `/kaggle/working/ab-validation/` in the saved version's Output tab.
"""),
            cell("code", "%%writefile /kaggle/working/baseline_dsmil.py\n" + baseline),
            cell("code", "%%writefile /kaggle/working/ab_validation_runner.py\n" + runner),
            cell("code", bundle_cell),
            cell("code", '''from pathlib import Path
import hashlib, json, subprocess, sys

candidates = []
for marker in Path("/kaggle/input").rglob("PREPROCESSING_COMPLETE.json"):
    info = json.loads(marker.read_text())
    if info.get("source") == "uniscale" and info.get("feature_dim") == 512:
        candidates.append(marker.parent)
assert len(candidates) == 1, f"Attach exactly one prepared uniscale dataset; found {candidates}"
prepared_root = candidates[0]
prepared_hash = hashlib.sha256((prepared_root / "slides.csv").read_bytes()).hexdigest()
assert prepared_hash == "12d13547a03ce93dacf2087fe237f16f9a69d1ac2611c6a6b88dadab945ca604", prepared_hash
output_root = Path("/kaggle/working/ab-validation")
command = [sys.executable, "/kaggle/working/ab_validation_runner.py",
           "--prepared-root", str(prepared_root),
           "--reference-bundle", str(reference_path),
           "--reference-sha256", reference_hash,
           "--output-root", str(output_root)]
print("Prepared dataset:", prepared_root, flush=True)
print("Running six validation-only configurations; expected new fits: 90", flush=True)
subprocess.run(command, check=True)
'''),
            cell("code", '''import pandas as pd
from IPython.display import display

complete = json.loads((output_root / "COMPLETE.json").read_text())
summary = json.loads((output_root / "AB_validation_summary.json").read_text())
comparison = pd.read_csv(output_root / "comparison_table.csv")
paired = pd.read_csv(output_root / "paired_fold_auc_differences.csv")
interaction = pd.read_csv(output_root / "B_interaction.csv")
all_slides = pd.read_csv(output_root / "oof_predictions_all_variants.csv")
assert complete["status"] == "complete" and complete["new_runs"] == 90
assert complete["new_oof_rows"] == 4860 and complete["test_evaluated"] is False
assert len(comparison) == 8 and len(paired) == 120 and len(interaction) == 15
assert len(all_slides) == 8 * 270
assert not list(output_root.rglob("test_predictions*"))
assert not list(output_root.rglob("test_summary.json"))
print("All 90 new fits and 30 archived reference fits verified; no official-test evaluation.")
display(comparison)
print("A1 alpha chosen by mean fold/seed validation AUC:", summary["alpha_selection"]["selected_alpha"])
print("B mean K-by-penalty interaction:", summary["B_mean_interaction"])
print("Saved output:", output_root)
'''),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(OUTPUT, "bundle SHA-256", bundle_hash)


if __name__ == "__main__":
    main()
