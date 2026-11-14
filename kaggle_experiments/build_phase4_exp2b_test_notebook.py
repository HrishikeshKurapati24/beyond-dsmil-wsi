"""Package frozen exp2b checkpoints and build its Kaggle test notebook."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "obtained_results/phase4_uni_validation/phase4-a1-validation/Phase4_2b_relative_top32/gamma_100"
AUDIT = HERE.parent / "dataset_audits/camelyon16_uni_audit.json"
BUNDLE = HERE / "phase4_exp2b_gamma100_test_reference.zip"
NOTEBOOK = HERE / "phase4_exp2b_gamma100_test_and_visualize.ipynb"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build() -> None:
    config = json.loads((SOURCE / "config.json").read_text())
    if config["mode"] != "relative_top32" or config["gamma"] != 1.0:
        raise ValueError("Expected Phase 4 exp2b gamma=1.0 archive")
    audit = json.loads(AUDIT.read_text())
    test = [row for row in audit["slides"] if row["split"] == "official_test"]
    if len(test) != 129:
        raise ValueError("Expected 129 official test coordinate records")
    fields = ("slide_id", "patches", "coordinate_shape", "level_dim", "patch_size_px",
              "patch_level", "out_of_bounds_coordinate_rows")
    coordinate_metadata = {"source_audit_sha256": sha_bytes(AUDIT.read_bytes()),
                           "slides": [{key: row[key] for key in fields} for row in test]}
    paths = [SOURCE / name for name in ("config.json", "cv_summary.json",
                                         "oof_predictions_ensemble.csv")]
    paths += [SOURCE / f"seed_{s}" / f"fold_{f}.pt"
              for s in (42, 43, 44) for f in range(5)]
    contents = {str(path.relative_to(SOURCE)): path.read_bytes() for path in paths}
    contents["test_coordinate_metadata.json"] = (json.dumps(coordinate_metadata, separators=(",", ":"))
                                                 + "\n").encode()
    manifest = {"kind": "phase4_exp2b_gamma100_test_reference",
                "member_sha256": {name: sha_bytes(data) for name, data in contents.items()}}
    contents["reference_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for name, data in contents.items():
            item = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            archive.writestr(item, data)

    cells = [cell("markdown", """# Phase 4 experiment 2(b): official test and three spatial figures

Attach **two Kaggle inputs**: (1) the prepared UNI dataset containing `PREPROCESSING_COMPLETE.json`, and (2) the companion `phase4_exp2b_gamma100_test_reference.zip` as an uploaded Kaggle Dataset. Enable a GPU **and Internet**, then choose **Save Version → Save & Run All**. This notebook loads the 15 frozen γ=1.0 checkpoints. It does not train or tune on the official test set.

It averages the 15 bag probabilities per official test slide and applies a threshold fixed from the training OOF ensemble (Youden J). It writes test metrics and per-slide predictions to `/kaggle/working/phase4-exp2b-gamma100-test/`. It then plots a correctly predicted tumor, a missed tumor, and a correctly predicted normal slide. Each plot marks four anchors and each anchor's eight largest attention weights from **one checkpoint chosen by validation performance**. The displayed test probability is from the 15-checkpoint ensemble.

The first figures plot patch **coordinates**. After test evaluation, the notebook downloads only the three selected original CAMELYON16 TIFFs and available matching XML lesion annotations from the [public AWS archive](https://registry.opendata.aws/camelyon/), verifies slide dimensions, and saves compressed whole-slide overlays and `three_wsi_overlays.zip`. Each large TIFF is removed after rendering. Tumor boundaries and excluded regions are drawn on the overlays. For each anchor and each of its top eight attended patches, the notebook reports whether the patch center falls in annotated tumor and estimates the tumor-covered fraction of its square using a 32×32 sample grid. The summary counts patches with at least 10% annotated tumor overlap. Missing XML is recorded as unavailable, never interpreted as a negative patch label; `test_114` has non-exhaustive annotations. The source audit's row alignment between embeddings and coordinates was assumed rather than independently proven, so the spatial analysis inherits that limitation.
""")]
    for name in ("baseline_dsmil.py", "ab_validation_runner.py", "phase4_a1_runner.py",
                 "phase4_exp2b_test.py", "phase4_annotation_utils.py",
                 "phase4_wsi_overlays.py"):
        cells.append(cell("code", f"%%writefile /kaggle/working/{name}\n" +
                          (HERE / name).read_text()))
    cells.append(cell("code", '''from pathlib import Path
import json, subprocess, sys, zipfile

input_root = Path('/kaggle/input')
prepared = []
for marker in input_root.rglob('PREPROCESSING_COMPLETE.json'):
    info = json.loads(marker.read_text())
    if info.get('source') == 'uni' and info.get('feature_dim') == 1024:
        prepared.append(marker.parent)
assert len(prepared) == 1, f'Attach exactly one prepared UNI dataset; found {prepared}'

references = [path.parent for path in input_root.rglob('reference_manifest.json')
              if json.loads(path.read_text()).get('kind') ==
              'phase4_exp2b_gamma100_test_reference']
zips = list(input_root.rglob('phase4_exp2b_gamma100_test_reference.zip'))
if not references:
    assert len(zips) == 1, f'Attach the companion checkpoint ZIP; found {zips}'
    reference = Path('/kaggle/working/phase4-exp2b-reference')
    reference.mkdir(exist_ok=False)
    with zipfile.ZipFile(zips[0]) as archive:
        names = archive.namelist()
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts
                   for name in names), 'Unsafe archive paths'
        archive.extractall(reference)
else:
    assert len(references) == 1, f'Expected one reference, found {references}'
    reference = references[0]

output_root = Path('/kaggle/working/phase4-exp2b-gamma100-test')
subprocess.run([sys.executable, '/kaggle/working/phase4_exp2b_test.py',
                '--prepared', str(prepared[0]), '--reference', str(reference),
                '--output', str(output_root)], check=True)
'''))
    cells.append(cell("code", '''import pandas as pd
from IPython.display import display, Image

complete = json.loads((output_root / 'COMPLETE.json').read_text())
assert complete['status'] == 'complete' and complete['n_checkpoints'] == 15
display(pd.DataFrame([complete['test_summary']]))
if complete['missing_figure_categories']:
    print('No test example exists for:', complete['missing_figure_categories'])
for item in json.loads((output_root / 'visualization_examples.json').read_text())['examples']:
    print(item['category'], item['slide_id'], 'ensemble p(tumor)=', round(item['probability'], 4))
    display(Image(filename=str(output_root / item['figure'])))
print('Saved results:', output_root)
'''))
    cells.append(cell("code", '''import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'boto3',
                'openslide-python', 'openslide-bin'], check=True)
subprocess.run([sys.executable, '/kaggle/working/phase4_wsi_overlays.py',
                '--result-root', str(output_root), '--reference', str(reference)], check=True)
from IPython.display import FileLink
display(FileLink(str(output_root / 'three_wsi_overlays.zip')))
for jpeg in sorted((output_root / 'wsi_overlays').glob('*.jpg')):
    print(jpeg.name)
    display(Image(filename=str(jpeg)))
annotation_summary = json.loads((output_root / 'wsi_overlays/annotation_overlap_summary.json').read_text())
display(pd.DataFrame(annotation_summary['slides']))
display(pd.read_csv(output_root / 'wsi_overlays/selected_patch_annotation_overlap.csv'))
'''))
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Notebook: {NOTEBOOK} ({NOTEBOOK.stat().st_size:,} bytes)")
    print(f"Reference bundle: {BUNDLE} ({BUNDLE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
