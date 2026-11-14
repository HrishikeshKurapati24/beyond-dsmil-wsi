"""Focused checks for the A1 reference and UNI spatial extension."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ab_validation_runner as ab  # noqa: E402
import phase4_a1_runner as p4  # noqa: E402


def test_archived_a1_checkpoints_load_and_predict(tmp_path):
    reference = p4.checked_reference(HERE / "a1_alpha025_test_reference.zip", tmp_path / "a1")
    x = torch.randn(7, 512)
    for seed in p4.SEEDS:
        for fold in p4.FOLDS:
            model = ab.ABMILNet(ab.Variant("A1_alpha_025", 4, alpha=0.25))
            state = torch.load(reference / f"seed_{seed}" / f"fold_{fold}.pt",
                               map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            with torch.no_grad():
                _, logit, _, _, details = model(x)
            assert torch.isfinite(logit).all() and len(details["indices"]) == 4
    assert json.loads((reference / "config.json").read_text())["variant"]["alpha"] == 0.25


def test_archived_a1_directory_matches_embedded_checksums(tmp_path):
    import zipfile

    with zipfile.ZipFile(HERE / "a1_alpha025_test_reference.zip") as archive:
        manifest = archive.read("reference_manifest.json")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest)
    source = HERE / "obtained_results/ab_validation_single_run/ab-validation/A1_alpha_025"
    reference = p4.checked_reference_directory(source, manifest_path, tmp_path / "verified")
    assert len(list(reference.glob("seed_*/fold_*.pt"))) == 15
    assert (reference / "oof_predictions_ensemble.csv").is_file()


def test_a1_notebook_source_is_below_kaggle_limit():
    notebook = HERE / "A1_alpha025_official_test.ipynb"
    assert notebook.stat().st_size < 1_000_000
    cells = json.loads(notebook.read_text())["cells"]
    code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    assert "--a1-reference-dir" in code
    assert "--a1-reference-manifest" in code
    assert "a1_bundle = base64.b64decode" not in code


def test_variants_share_initial_model_weights():
    device = torch.device("cpu")
    torch.manual_seed(43)
    ordinary, _, _ = p4.make_uni_model("none", 0.0, device)
    torch.manual_seed(43)
    spatial, _, _ = p4.make_uni_model("relative_all", 0.5, device)
    torch.manual_seed(43)
    capped, _, _ = p4.make_uni_model("relative_top32", 0.5, device)
    for name, weight in ordinary.state_dict().items():
        torch.testing.assert_close(weight, spatial.state_dict()[name])
        torch.testing.assert_close(weight, capped.state_dict()[name])


def test_spatial_selector_balances_score_feature_and_distance():
    features = torch.zeros(6, 1024)
    features[:, 0] = 1
    features[3, 0], features[3, 1] = 0, 1
    scores = torch.tensor([10., 9., 8., 7., 6., 5.])
    coords = torch.tensor([[0., 0.], [.1, 0.], [.2, 0.], [100., 0.], [.3, 0.], [.4, 0.]])
    indices, records = p4.select_spatial_anchors(features, scores, torch.arange(6), coords)
    assert indices[0].item() == 0
    assert indices[1].item() == 3
    assert records[1] == {"score_rank": 3, "feature_rank": 1, "spatial_rank": 1}


def test_top_c_uses_distance_biased_scores_before_softmax():
    visual = torch.tensor([[0.9], [0.8], [0.7]])
    biased = torch.tensor([[0.1], [0.8], [0.7]])
    mask = p4.top_c_mask(biased, torch.arange(3), limit=2)
    assert mask[:, 0].tolist() == [False, True, True]
    assert p4.top_c_mask(visual, torch.arange(3), limit=2)[:, 0].tolist() == [True, True, False]
    attention = torch.softmax(biased.masked_fill(~mask, -torch.inf), dim=0)
    assert attention[0, 0] == 0 and torch.isclose(attention[:, 0].sum(), torch.tensor(1.0))


def test_top32_pooling_has_exactly_32_nonzero_weights_and_gradients():
    model, _, _ = p4.make_uni_model("relative_top32", 0.5, torch.device("cpu"))
    full_model, _, _ = p4.make_uni_model("relative_all", 0.5, torch.device("cpu"))
    full_model.load_state_dict(model.state_dict())
    features = torch.randn(40, 1024)
    coords = torch.arange(40, dtype=torch.float32)[:, None].repeat(1, 2)
    with torch.no_grad():
        _, _, full_attention, full_details = full_model(features, coords=coords)
    _, bag, attention, details = model(features, coords=coords)
    torch.testing.assert_close(details["indices"], full_details["indices"])
    for column in range(4):
        expected = set(torch.topk(full_attention[:, column], 32).indices.tolist())
        actual = set(torch.nonzero(attention[:, column]).view(-1).tolist())
        assert actual == expected
    expected_bags = attention.T @ features
    torch.testing.assert_close(details["anchor_bags"], expected_bags)
    assert (attention > 0).sum(dim=0).tolist() == [32] * 4
    torch.testing.assert_close(attention.sum(dim=0), torch.ones(4))
    assert details["support_count"].tolist() == [32] * 4
    bag.sum().backward()
    assert model.b_classifier.q[0].weight.grad.abs().sum() > 0


def test_top32_pooling_uses_all_patches_when_the_bag_is_smaller():
    model, _, _ = p4.make_uni_model("relative_top32", 0.5, torch.device("cpu"))
    features = torch.randn(7, 1024)
    coords = torch.randn(7, 2)
    _, _, attention, details = model(features, coords=coords)
    assert details["support_count"].tolist() == [7] * 4
    torch.testing.assert_close(attention.sum(dim=0), torch.ones(4))


def test_selected_anchor_indices_are_invariant_to_patch_permutation():
    model, _, _ = p4.make_uni_model("relative_top32", 0.5, torch.device("cpu"))
    model.eval()
    features = torch.randn(13, 1024)
    coords = torch.rand(13, 2)
    permutation = torch.tensor([12, 5, 1, 9, 0, 3, 8, 2, 6, 10, 11, 7, 4])
    with torch.no_grad():
        _, bag_a, _, detail_a = model(features, coords=coords)
        _, bag_b, _, detail_b = model(features[permutation], original_ids=permutation,
                                     coords=coords[permutation])
    selected_a = detail_a["indices"]
    selected_b = permutation[detail_b["indices"]]
    assert selected_a.tolist() == selected_b.tolist()
    torch.testing.assert_close(bag_a, bag_b, atol=1e-5, rtol=1e-5)


def test_coordinates_use_source_patch_size_not_patch_extrema(tmp_path):
    (tmp_path / "bags").mkdir()
    (tmp_path / "coords").mkdir()
    np.save(tmp_path / "bags/s.npy", np.ones((2, 1024), dtype=np.float32))
    np.save(tmp_path / "coords/s.npy", np.array([[5, 10], [8, 15]], dtype=np.int64))
    row = {"slide_id": "s", "n_patches": 2, "bag_path": "bags/s.npy",
           "coords_path": "coords/s.npy"}
    metadata = {"s": {"level_dim": [10, 20], "patch_size_px": 5}}
    features, coords = p4.load_uni(tmp_path, row, torch.device("cpu"), metadata, True)
    assert features.shape == (2, 1024)
    torch.testing.assert_close(coords, torch.tensor([[1., 2.], [1.6, 3.]]))


def test_uni_manifest_does_not_open_test_feature_files(tmp_path):
    (tmp_path / "bags").mkdir()
    (tmp_path / "coords").mkdir()
    rows = []
    for index in range(270):
        sid = f"normal_{index + 1:03d}"
        np.save(tmp_path / f"bags/{sid}.npy", np.zeros((1, 1024), dtype=np.float32))
        np.save(tmp_path / f"coords/{sid}.npy", np.zeros((1, 2), dtype=np.int64))
        rows.append({"slide_id": sid, "label": index % 2,
                     "official_split": "train", "validation_fold": index // 54,
                     "n_patches": 1, "feature_dim": 1024,
                     "bag_path": f"bags/{sid}.npy", "coords_path": f"coords/{sid}.npy"})
    for index in range(1, 130):
        sid = f"test_{index:03d}"
        rows.append({"slide_id": sid, "label": 0,
                     "official_split": "test", "validation_fold": "",
                     "n_patches": 1, "feature_dim": 1024,
                     "bag_path": f"bags/{sid}.npy", "coords_path": f"coords/{sid}.npy"})
    import baseline_dsmil as base
    base.write_csv(tmp_path / "slides.csv", rows, list(rows[0]))
    (tmp_path / "PREPROCESSING_COMPLETE.json").write_text(json.dumps({
        "source": "uni", "feature_dim": 1024, "seed": 42, "folds": 5,
        "slides": 399, "train_slides": 270, "test_slides": 129}))
    loaded, _ = p4.load_uni_manifest(tmp_path)
    assert len(loaded) == 399
    assert not any((tmp_path / row["bag_path"]).exists()
                   for row in loaded if row["official_split"] == "test")
