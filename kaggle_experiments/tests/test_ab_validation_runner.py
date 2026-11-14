"""Focused checks for the expensive A/B Kaggle experiment contract."""

from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ab_validation_runner as ab  # noqa: E402
import baseline_dsmil as base  # noqa: E402


def test_alpha_zero_matches_phase2_forward_and_gradients():
    base.seed_all(42)
    original = base.MILNet(anchors=4)
    changed = ab.ABMILNet(ab.PHASE2)
    changed.load_state_dict(original.state_dict())
    x = torch.randn(8, 512)
    old_instance, old_bag, old_attention, _ = original(x)
    new_instance, new_bag, new_attention, _, _ = changed(x)
    for old, new in ((old_instance, new_instance), (old_bag, new_bag),
                     (old_attention, new_attention)):
        torch.testing.assert_close(old, new)
    old_bag.sum().backward()
    new_bag.sum().backward()
    for old, new in zip(original.parameters(), changed.parameters()):
        torch.testing.assert_close(old.grad, new.grad)


def test_diverse_selector_is_deterministic_and_can_change_anchors():
    x = torch.zeros(5, 512)
    x[:, 0] = 1
    x[3, 0], x[3, 1] = 0, 1
    scores = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0])
    row_ids = torch.arange(5)
    original, _ = ab.select_diverse(x, scores, row_ids, 0.0, 4)
    diverse, _ = ab.select_diverse(x, scores, row_ids, 0.75, 4)
    assert original.tolist() == [0, 1, 2, 3]
    assert diverse[0].item() == 0 and diverse[1].item() == 3
    assert len(set(diverse.tolist())) == 4
    tied, _ = ab.select_diverse(x, torch.ones(5), torch.arange(4, -1, -1), 0.75, 4)
    assert tied[0].item() == 4


def test_zero_initialized_fusion_matches_phase2():
    base.seed_all(43)
    original = base.MILNet(anchors=4)
    fused = ab.ABMILNet(ab.NEW_VARIANTS[3])
    fused.load_state_dict(original.state_dict(), strict=False)
    x = torch.randn(7, 512)
    _, old_bag, _, _ = original(x)
    _, new_bag, _, _, details = fused(x)
    torch.testing.assert_close(old_bag, new_bag)
    torch.testing.assert_close(details["fusion_weights"], torch.full((4,), 0.25))


def test_archived_reference_has_same_270_slide_fold_map():
    bundle = HERE / "ab_phase2_phase3_reference.zip"
    with zipfile.ZipFile(bundle) as archive:
        rows = [{"slide_id": r["slide_id"], "label": int(r["label"]),
                 "validation_fold": int(r["fold"]), "official_split": "train"}
                for r in csv.DictReader(archive.read("phase2/oof_predictions_all_seeds.csv").decode().splitlines())
                if int(r["seed"]) == 42]
    references = ab.read_reference(bundle, ab.sha256(bundle), rows, ab.PREPARED_SHA256)
    assert len(references) == 2
    assert all(len(value["summary"]["runs"]) == 15 and len(value["oof"]) == 810
               for value in references.values())
