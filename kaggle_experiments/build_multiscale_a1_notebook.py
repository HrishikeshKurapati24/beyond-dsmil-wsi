"""Build the single self-contained Kaggle multiscale A1 notebook."""
from __future__ import annotations

import hashlib
import json
import base64
import io
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "obtained_results/a1_alpha025_official_test"
OUTPUT = HERE / "06_a1_multiscale_ablation_and_test.ipynb"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build():
    names = ("baseline_dsmil.py", "ab_validation_runner.py", "phase4_a1_runner.py",
             "multiscale_a1_runner.py")
    paths = sorted(p for p in PRIOR.rglob("*") if p.is_file() and p.name not in ("README.md", "SHA256SUMS.txt"))
    assert len(paths) == 8
    manifest = {str(p.relative_to(PRIOR)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, str(path.relative_to(PRIOR)))
    encoded = base64.b64encode(payload.getvalue()).decode()
    cells = [cell("markdown", """# A1 multiscale ablation and official test

Attach only the prepared **multiscale `cat`** Kaggle dataset. The already obtained A1 alpha=0.25 official-test predictions and training OOF scores are embedded as a small checksummed bundle; this notebook does **not** run A1 test inference or need its checkpoints or uniscale feature bags. Select **GPU T4 ×2** and **Save Version → Save & Run All**. The notebook trains fixed and adaptive multiscale A1 concurrently (15 fold/seed fits each), verifies both, then evaluates those two on the 129 official test slides and compares them with the archived A1 test results. All outputs are written to `/kaggle/working/multiscale-a1/`.

The earlier project has already viewed CAMELYON16 official-test results; this comparison is exploratory on the existing holdout. The test data is not used for fitting or tuning this study.
""")]
    cells += [cell("code", f"%%writefile /kaggle/working/{name}\n" + (HERE / name).read_text())
              for name in names]
    cells.append(cell("code", """from pathlib import Path
import base64, hashlib, io, json, torch, zipfile

root = Path('/kaggle/working/multiscale-a1')
root.mkdir(parents=True, exist_ok=True)
manifest = MANIFEST_LITERAL
prior = root / 'prior-a1-test'
prior.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_LITERAL))) as archive:
    assert set(archive.namelist()) == set(manifest)
    for name, expected in manifest.items():
        rel = Path(name)
        assert not rel.is_absolute() and '..' not in rel.parts
        data = archive.read(name)
        assert hashlib.sha256(data).hexdigest() == expected, name
        target = prior / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

sources = {}
for marker in Path('/kaggle/input').rglob('PREPROCESSING_COMPLETE.json'):
    info = json.loads(marker.read_text())
    if info.get('source') == 'multiscale':
        sources.setdefault(info['source'], []).append(marker.parent)
assert len(sources.get('multiscale', [])) == 1, sources
multiscale = sources['multiscale'][0]

import sys
sys.path.insert(0, '/kaggle/working')
import multiscale_a1_runner as runner
mrows, _ = runner.load_multi(multiscale)
runner.verify_a1_results(prior, mrows)
assert torch.cuda.is_available(), 'Enable a GPU accelerator in Kaggle'
print('Inputs verified:', multiscale, 'and embedded prior A1 test results')
print('Visible CUDA devices:', torch.cuda.device_count())
""".replace("MANIFEST_LITERAL", repr(manifest)).replace("BUNDLE_LITERAL", repr(encoded))))
    cells.append(cell("code", """# Verify the adaptive extension starts from the exact fixed A1 model.
import torch
sample = torch.randn(11, 1024)
torch.manual_seed(123); fixed_probe = runner.MultiA1(False)
torch.manual_seed(123); adaptive_probe = runner.MultiA1(True)
fixed_result, adaptive_result = fixed_probe(sample), adaptive_probe(sample)
torch.testing.assert_close(fixed_result[1], adaptive_result[1], rtol=0, atol=0)
assert fixed_result[3]['indices'].tolist() == adaptive_result[3]['indices'].tolist()
torch.testing.assert_close(adaptive_result[4], torch.full((11, 2), 0.5))
loss, _, _, _ = runner.step(adaptive_probe, sample, 1, torch.nn.BCEWithLogitsLoss())
loss.backward()
assert adaptive_probe.gate.weight.grad is not None
assert adaptive_probe.gate.weight.grad.abs().sum().item() > 0
print('Fixed/adaptive initialization and gate gradient verified.')
"""))
    cells.append(cell("code", """# Resource pilot on the largest official-training bag; no model selection.
import time
largest = max((r for r in mrows if r['official_split'] == 'train'), key=lambda r: r['n_patches'])
for adaptive, gpu in ((False, 0), (True, 1 if torch.cuda.device_count() > 1 else 0)):
    device = torch.device(f'cuda:{gpu}')
    # Cumulative peaks give a conservative bound on reruns and avoid resetting
    # an allocator that may not yet be initialized for this device.
    started = time.monotonic()
    probe = runner.MultiA1(adaptive).to(device)
    x = runner.bag(multiscale, largest, device)
    loss, _, _, _ = runner.step(probe, x, largest['label'], torch.nn.BCEWithLogitsLoss())
    loss.backward()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    capacity = torch.cuda.get_device_properties(device).total_memory
    print(('adaptive' if adaptive else 'fixed'), 'GPU', gpu, 'slide', largest['slide_id'],
          'patches', largest['n_patches'], 'peak GiB', round(peak / 2**30, 2),
          'capacity GiB', round(capacity / 2**30, 2),
          'seconds', round(time.monotonic() - started, 2))
    assert peak < .8 * capacity, 'Large-bag pilot leaves insufficient GPU margin'
    del x, loss, probe
    torch.cuda.empty_cache()
"""))
    cells.append(cell("code", """# One independent 15-fit worker per T4; automatically fall back to one GPU.
import os, subprocess, sys, time
from pathlib import Path

jobs = [('M-fixed', 'validate_fixed', 0), ('M-adaptive', 'validate_adaptive', 1)]
parallel = torch.cuda.device_count() >= 2
print('Parallel two-GPU validation:', parallel)
script = '/kaggle/working/multiscale_a1_runner.py'

def start_job(name, stage, gpu):
    destination = root / name
    if (destination / 'COMPLETE.json').is_file():
        runner.verify_variant(name, multiscale, destination)
        print('Verified completed output:', name)
        return None
    assert not destination.exists(), f'Partial output exists; inspect it before retrying: {destination}'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = env['OPENBLAS_NUM_THREADS'] = '1'
    log = (root / (name + '.log')).open('w')
    process = subprocess.Popen([sys.executable, '-u', script, '--stage', stage,
        '--multi', str(multiscale), '--output', str(destination)],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    return process, log

active = []
if parallel:
    for name, stage, gpu in jobs:
        job = start_job(name, stage, gpu)
        if job: active.append((name, *job))
    for name, process, log in active:
        returncode = process.wait()
        log.close()
        print(name, 'exit code', returncode, 'log:', root / (name + '.log'))
        if returncode:
            print((root / (name + '.log')).read_text()[-5000:])
    assert all(process.returncode == 0 for _, process, _ in active), 'Validation worker failed'
else:
    for name, stage, _ in jobs:
        job = start_job(name, stage, 0)
        if job:
            process, log = job
            returncode = process.wait()
            log.close()
            print(name, 'exit code', returncode)
            if returncode:
                print((root / (name + '.log')).read_text()[-5000:])
                raise RuntimeError(f'{name} validation failed')
for name, _, _ in jobs:
    runner.verify_variant(name, multiscale, root / name)
print('Both multiscale validation variants completed and verified.')
"""))
    cells.append(cell("code", """# Final test begins only after both validation variants have completed.
import os, subprocess, sys, json
test_out = root / 'official-test'
assert not test_out.exists(), 'Test output already exists; inspect it before rerunning'
subprocess.run([sys.executable, '/kaggle/working/multiscale_a1_runner.py',
    '--stage', 'final_test', '--multi', str(multiscale), '--prior-a1', str(prior),
    '--fixed', str(root / 'M-fixed'), '--adaptive', str(root / 'M-adaptive'),
    '--output', str(test_out)], check=True)
complete = json.loads((test_out / 'COMPLETE.json').read_text())
assert complete['status'] == 'complete' and complete['models_per_variant'] == 15
import pandas as pd
display(pd.read_csv(test_out / 'test_comparison.csv'))
print('Save this completed notebook version with its /kaggle/working output.')
"""))
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
        "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(notebook, indent=1).encode()
    if len(data) >= 1_000_000:
        raise ValueError(f"Notebook source exceeds 1 MB: {len(data)} bytes")
    OUTPUT.write_bytes(data)
    print(OUTPUT, len(data), "bytes")


if __name__ == '__main__':
    build()
