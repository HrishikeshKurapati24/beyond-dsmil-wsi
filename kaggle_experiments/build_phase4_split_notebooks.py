"""Generate separate, two-GPU Phase-4 experiment 3 and 4 Kaggle notebooks."""
from __future__ import annotations

import base64
import hashlib
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "dataset_audits/camelyon16_uni_audit.json"
SOURCES = ("baseline_dsmil.py", "ab_validation_runner.py", "phase4_a1_runner.py",
           "phase4_exp3_4_runner.py")
STAGES = {
    "exp3": ("phase4_experiment_3.ipynb", (
        "exp3_control", "exp3_spatial_lam_000", "exp3_spatial_lam_010",
        "exp3_spatial_lam_025", "exp3_spatial_lam_050")),
    "exp4": ("phase4_experiment_4.ipynb", (
        "exp4_flat_topk_uni", "exp4_4a_feature_assignment", "exp4_4a_spatial",
        "exp4_4b_intermediate", "exp4_4c_path")),
}


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update({"execution_count": None, "outputs": []})
    return item


def embedded_audit():
    source = json.loads(AUDIT.read_text())
    train = [row for row in source["slides"] if row["split"] == "official_train"]
    assert len(train) == 270
    compact = {"source_audit_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
               "slides": [{key: row[key] for key in (
                   "slide_id", "patches", "coordinate_shape", "level_dim", "patch_size_px",
                   "patch_level", "out_of_bounds_coordinate_rows")} for row in train]}
    payload = json.dumps(compact, separators=(",", ":")).encode()
    encoded = base64.b64encode(payload).decode()
    chunks = "\n".join(f"    {piece!r}" for piece in textwrap.wrap(encoded, 100))
    return ("from pathlib import Path\nimport base64, hashlib\n"
            f"payload = base64.b64decode(\n{chunks}\n)\n"
            f"assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}\n"
            "audit_path = Path('/kaggle/working/uni_training_coordinate_metadata.json')\n"
            "audit_path.write_bytes(payload)\ndel payload\n")


def build(stage):
    filename, variants = STAGES[stage]
    intro = f"""# Phase 4 experiment {stage[-1]}: UNI validation

Attach the **prepared CAMELYON16-UNI dataset**, select **GPU T4 ×2**, and use **Save Version → Save & Run All**. This notebook runs at most two unfinished variants per version, one on each T4. It never opens official-test feature bags. The model runner and its source hashes are unchanged from the earlier combined notebook, so completed variants from that notebook can be reused.

To reuse your earlier results, attach its saved notebook **Output** as an additional input. To continue after this notebook's first pair, attach the most recent saved output to the next version of this same notebook. Attach at most one previous output; each version carries all verified completed variants forward. Change `MAX_NEW_VARIANTS` only to 1 for a smaller run, never above 2. Results appear under `/kaggle/working/phase4-{stage}/`. The last run has one remaining variant.

Experiment 3 has a no-gap multi-head control and spatial λ = 0, 0.1, 0.25, 0.5. Experiment 4 has the flat Top-K reference, feature-only assignment control, and cumulative 4(a), 4(b), 4(c) models. These families remain separate and are not tested on the UNI official-test set here.
"""
    cells = [cell("markdown", intro)]
    cells += [cell("code", f"%%writefile /kaggle/working/{name}\n" + (HERE / name).read_text())
              for name in SOURCES]
    cells.append(cell("code", embedded_audit()))
    setup = '''from pathlib import Path
import hashlib, json, shutil, sys, time
import numpy as np
import pandas as pd
import torch

STAGE = STAGE_LITERAL
VARIANTS = VARIANTS_LITERAL
MAX_NEW_VARIANTS = 2  # each new variant gets one independent T4 worker
GRID_WIDTH = 4  # experiment 4 only; use 8 only if coordinate pilot warrants it
assert MAX_NEW_VARIANTS in (1, 2) and GRID_WIDTH in (4, 8)

prepared = []
for marker in Path('/kaggle/input').rglob('PREPROCESSING_COMPLETE.json'):
    info = json.loads(marker.read_text())
    if info.get('source') == 'uni' and info.get('feature_dim') == 1024:
        prepared.append(marker.parent)
assert len(prepared) == 1, f'Attach exactly one prepared UNI dataset: {prepared}'
prepared = prepared[0]
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Select Kaggle GPU T4 ×2 before training'
print('Visible GPUs:', torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

sys.path.insert(0, '/kaggle/working')
import baseline_dsmil as base
import ab_validation_runner as ab
import phase4_a1_runner as p4
import phase4_exp3_4_runner as experiments
rows, _ = p4.load_uni_manifest(prepared)
metadata = p4.load_coordinate_metadata(audit_path, rows)

if STAGE == 'exp4':
    manifest = pd.read_csv(prepared / 'slides.csv')
    audit = {r['slide_id']: r for r in json.loads(audit_path.read_text())['slides']}
    training = manifest[manifest.official_split == 'train'].sort_values('n_patches')
    for row in (training.iloc[0], training.iloc[len(training)//2], training.iloc[-1]):
        raw = np.load(prepared / row['coords_path'], mmap_mode='r')
        groups = np.floor(np.asarray(raw, dtype=np.float32) /
                          audit[row['slide_id']]['patch_size_px'] / GRID_WIDTH).astype(np.int32)
        _, sizes = np.unique(groups, axis=0, return_counts=True)
        print('grid pilot', row['slide_id'], 'groups', len(sizes),
              'median patches/group', float(np.median(sizes)))

output_root = Path('/kaggle/working/phase4-' + STAGE)
output_root.mkdir(parents=True, exist_ok=True)
markers = [p for p in Path('/kaggle/input').rglob('PHASE4_EXPERIMENTS_3_4.json')]
markers += [p for p in Path('/kaggle/input').rglob('PHASE4_SPLIT_NOTEBOOK.json')
            if json.loads(p.read_text()).get('stage') == STAGE]
prior_roots = list(dict.fromkeys(p.parent for p in markers))
assert len(prior_roots) <= 1, f'Attach at most one previous output: {prior_roots}'

def verify_variant(folder, name):
    cfg = json.loads((folder / 'config.json').read_text())
    done = json.loads((folder / 'COMPLETE.json').read_text())
    cv = json.loads((folder / 'cv_summary.json').read_text())
    required = {'prepared_manifest_sha256': experiments.sha(prepared / 'slides.csv'),
                'coordinate_audit_sha256': experiments.sha(audit_path),
                'runner_sha256': experiments.sha(Path(experiments.__file__)),
                'baseline_sha256': experiments.sha(Path(base.__file__)),
                'phase4_loader_sha256': experiments.sha(Path(p4.__file__)),
                'ab_runner_sha256': experiments.sha(Path(ab.__file__))}
    assert cfg['variant'] == name and all(cfg.get(k) == v for k, v in required.items())
    assert cfg['seeds'] == [42, 43, 44] and cfg['folds'] == [0, 1, 2, 3, 4]
    if STAGE == 'exp4': assert cfg['grid_width_patch_units'] == GRID_WIDTH
    assert done['status'] == 'complete' and done['fits'] == 15 and done['oof_rows'] == 810
    assert done['test_evaluated'] is False and cv['test_evaluated'] is False and len(cv['runs']) == 15
    oof = base.read_csv(folder / 'oof_predictions_all_seeds.csv')
    assert len(oof) == 810 and len({(r['seed'], r['slide_id']) for r in oof}) == 810
    for seed in (42, 43, 44):
        for fold in range(5):
            run = folder / f'seed_{seed}' / f'fold_{fold}'
            assert all((run / name).is_file() for name in
                       ('checkpoint.pt', 'oof.csv', 'history.csv', 'diagnostics.json', 'COMPLETE.json'))

if prior_roots:
    prior = prior_roots[0]
    for name in VARIANTS:
        source = prior / name
        if (source / 'COMPLETE.json').is_file():
            verify_variant(source, name)
            target = output_root / name
            assert not target.exists(), f'Output already exists: {target}'
            shutil.copytree(source, target)
            print('Reused verified variant:', name)

pending = [name for name in VARIANTS if not (output_root / name / 'COMPLETE.json').is_file()]
selected = pending[:MAX_NEW_VARIANTS]
print('Completed:', [name for name in VARIANTS if name not in pending])
print('Training this version:', selected)
'''
    cells.append(cell("code", setup.replace("STAGE_LITERAL", repr(stage))
                      .replace("VARIANTS_LITERAL", repr(variants))))
    launch = '''import os, subprocess

workers = []
for gpu, name in enumerate(selected):
    worker_root = Path('/kaggle/working') / f'phase4-{STAGE}-worker-{gpu}'
    assert not worker_root.exists(), worker_root
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = env['OPENBLAS_NUM_THREADS'] = '1'
    log_path = output_root / f'{name}.log'
    log = log_path.open('w')
    cmd = [sys.executable, '-u', '/kaggle/working/phase4_exp3_4_runner.py',
           '--prepared', str(prepared), '--audit', str(audit_path),
           '--output', str(worker_root), '--stage', name, '--grid-width', str(GRID_WIDTH)]
    process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    workers.append((name, worker_root, process, log, log_path))

failures = []
for name, worker_root, process, log, log_path in workers:
    code = process.wait()
    log.close()
    print(name, 'exit:', code, 'log:', log_path)
    if code:
        print(log_path.read_text()[-6000:])
        failures.append(name)
    else:
        source = worker_root / name
        verify_variant(source, name)
        shutil.move(str(source), str(output_root / name))
        shutil.rmtree(worker_root)
assert not failures, f'Failed variants: {failures}'

experiments.write_comparisons(output_root)
complete = [name for name in VARIANTS if (output_root / name / 'COMPLETE.json').is_file()]
if len(complete) == len(VARIANTS):
    base.write_json(output_root / f'{STAGE}_COMPLETE.json',
                    {'status': 'complete', 'variants': complete, 'test_evaluated': False})
base.write_json(output_root / 'PHASE4_SPLIT_NOTEBOOK.json',
                {'stage': STAGE, 'completed_variants': complete,
                 'remaining_variants': [n for n in VARIANTS if n not in complete],
                 'test_evaluated': False})
summary = []
for name in complete:
    verify_variant(output_root / name, name)
    cv = json.loads((output_root / name / 'cv_summary.json').read_text())
    summary.append({'variant': name, 'mean_validation_auc': cv['mean_validation_auc'],
                    'mean_validation_loss': cv['mean_validation_loss'],
                    'oof_ensemble_auc': cv['oof_ensemble_auc']})
display(pd.DataFrame(summary))
print('Remaining:', [n for n in VARIANTS if n not in complete])
print('Save this notebook version with output. Attach its Output to the next version if any remain.')
'''
    cells.append(cell("code", launch))
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                            "name": "python3"}, "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = HERE / filename
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    if path.stat().st_size >= 1_000_000:
        raise ValueError(f"Notebook exceeds Kaggle 1 MB source limit: {path}")
    print(path, path.stat().st_size)


if __name__ == "__main__":
    for family in STAGES:
        build(family)
