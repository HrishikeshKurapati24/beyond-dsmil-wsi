"""Freeze the Phase 2/3 validation outputs needed by the combined A/B notebook."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "obtained_results" / "three_seed_test"
OUTPUT = HERE / "ab_phase2_phase3_reference.zip"
BASELINE_SHA = "32dc2a316278566e4c41dbe4fc01ca04e003c726beffcc8b7311f59f13cd2385"
PREPARED_SHA = "12d13547a03ce93dacf2087fe237f16f9a69d1ac2611c6a6b88dadab945ca604"


def main() -> None:
    members = {}
    phases = {}
    for name, folder, weight in (
        ("phase2", "phase_2_four_anchors", 0.0),
        ("phase3", "phase_3_diversity", 0.1),
    ):
        source = ARCHIVE / folder
        code = source / "baseline_dsmil.py"
        assert hashlib.sha256(code.read_bytes()).hexdigest() == BASELINE_SHA
        cv = next(p for p in source.iterdir() if p.is_dir() and not p.name.endswith("-test"))
        config = json.loads((cv / "config.json").read_text())
        assert config["seeds"] == [42, 43, 44]
        assert config["folds"] == 5 and config["anchors"] == 4
        assert config["diversity_weight"] == weight
        assert config["manifest_sha256"] == PREPARED_SHA
        phases[name] = {"source_folder": folder, "code_sha256": BASELINE_SHA,
                        "prepared_manifest_sha256": PREPARED_SHA,
                        "anchors": 4, "diversity_weight": weight}
        paths = [cv / "config.json", cv / "cv_summary.json", cv / "oof_predictions_all_seeds.csv"]
        for seed in (42, 43, 44):
            seed_root = cv / f"seed_{seed}"
            paths.append(seed_root / "config.json")
            for fold in range(5):
                paths += [seed_root / f"fold_{fold}_summary.json",
                          seed_root / f"fold_{fold}_anchor_diagnostics.json"]
        assert len(paths) == 36
        for path in paths:
            rel = Path(name) / path.relative_to(cv)
            members[rel.as_posix()] = path.read_bytes()
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    manifest = {"kind": "ab_phase2_phase3_validation_reference", "phases": phases,
                "member_sha256": checksums, "member_count": len(members)}
    members["reference_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data, compresslevel=9)
    print(OUTPUT)
    print("SHA-256", hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    print("files", len(members), "bytes", OUTPUT.stat().st_size)


if __name__ == "__main__":
    main()
