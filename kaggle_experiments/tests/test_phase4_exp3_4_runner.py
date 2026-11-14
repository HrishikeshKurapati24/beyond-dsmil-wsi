"""Small-bag checks for the combined Phase 4 experiment notebook."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import phase4_exp3_4_runner as run  # noqa: E402
import baseline_dsmil as base  # noqa: E402


def test_beam_recovers_alternative_anchor_and_hard_gap():
    scores = torch.tensor([[9., 10., 0.], [10., 0., 0.],
                           [0., 0., 10.], [9.5, 0., 0.]])
    coords = torch.tensor([[0., 0.], [5., 0.], [12., 0.], [24., 0.]])
    selected, trace = run.spaced_beam(scores, coords, torch.arange(4), width=64,
                                      pool_sizes=(4,))
    assert len(selected) == 3 and trace[0]["selected_heads"] == 3
    assert selected[0][1] == 3  # Head 0 gives up its top-1 patch at x=5.
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            assert torch.linalg.vector_norm(coords[selected[i][1]] -
                                            coords[selected[j][1]]) >= 8


def test_variable_head_count_keeps_distance_and_gradients():
    model = run.MultiHeadSpatial(True, 0.25)
    x = torch.randn(6, 1024)
    coords = torch.zeros(6, 2)
    bag, loss, info = model(x, coords, label=1)
    assert info["active_count"] == 1
    assert info["pairwise_distances_patch_widths"] == []
    assert torch.isfinite(loss) and torch.isfinite(bag)
    loss.backward()
    assert model.scorer.weight.grad.abs().sum() > 0


def test_bag_loss_reaches_scoring_heads_through_straight_through_selection():
    model = run.MultiHeadSpatial(True, 0.0)
    x = torch.randn(8, 1024)
    coords = torch.stack((torch.arange(8.) * 10, torch.zeros(8)), dim=1)
    bag, _, _ = model(x, coords, label=1)
    bag.backward()
    assert model.scorer.weight.grad is not None
    assert model.scorer.weight.grad.abs().sum() > 0


def test_group_split_cover_and_all_hierarchy_losses_backpropagate():
    x = torch.randn(12, 1024)
    coords = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.],
                           [6., 0.], [6., 1.], [12., 0.], [12., 1.],
                           [18., 0.], [18., 1.], [24., 0.], [24., 1.]])
    groups, branches, regions, _, parent, _ = run.hierarchy_indices(
        coords, x, [0, 1, 6, 10], 0.5, 4)
    assert sorted(i for _, members, _ in groups for i in members) == list(range(12))
    assert all(branches) and len(parent) == len(groups)
    assert len({gi for members in branches for gi in members}) == len(groups)
    for level in (1, 2, 3):
        model = run.HierarchicalTopK(level, 0.5)
        bag, loss, info = model(x, coords, label=1)
        assert torch.isfinite(bag) and torch.isfinite(loss)
        assert sum(info["group_sizes"]) == 12
        assert info["active_count"] == 4
        if level == 3:
            assert 0 <= info["path_probability"] <= 1
            path = info["strongest_path"]
            assert info["group_parent_region"][path["group"]] == path["region"]
            assert info["region_branch"][path["region"]] == path["branch"]
        loss.backward()
        assert model.i_classifier.fc[0].weight.grad.abs().sum() > 0


def test_same_coordinate_anchors_keep_distinct_nonempty_groups():
    x = torch.randn(5, 1024)
    coords = torch.zeros(5, 2)
    groups, branches, _, _, _, _ = run.hierarchy_indices(coords, x, [0, 1, 2, 3], 0.5)
    assert all(branches)
    for anchor in range(4):
        assert any(anchor in groups[gi][1] for gi in branches[anchor])


def test_flat_uni_matches_released_topk_forward_for_unique_scores():
    flat = run.FlatTopKUNI()
    reference = base.MILNet(feature_dim=1024, anchors=4)
    flat.load_state_dict(reference.state_dict())
    x = torch.randn(20, 1024)
    with torch.no_grad():
        _, original_bag, original_attention, _ = reference(x)
        bag, _, info = flat(x, torch.zeros(20, 2))
    torch.testing.assert_close(bag, original_bag.view(()), atol=1e-6, rtol=1e-6)
    assert info["attention_columns"] == original_attention.shape[1]


def test_hierarchy_handles_single_patch_bag():
    x = torch.randn(1, 1024)
    coords = torch.zeros(1, 2)
    model = run.HierarchicalTopK(3, 0.5)
    bag, loss, info = model(x, coords, label=0)
    assert info["active_count"] == 1 and info["microgroups"] == 1
    assert torch.isfinite(bag) and torch.isfinite(loss)


def test_notebook_is_current_and_under_source_limit():
    notebook = HERE / "05_phase4_experiments_3_and_4.ipynb"
    assert notebook.stat().st_size < 1_000_000
    cells = json.loads(notebook.read_text())["cells"]
    for cell in cells:
        source = "".join(cell["source"])
        if source.startswith("%%writefile "):
            name = source.splitlines()[0].split("/")[-1]
            assert source.split("\n", 1)[1] == (HERE / name).read_text()
        elif cell["cell_type"] == "code":
            compile(source, str(notebook), "exec")
