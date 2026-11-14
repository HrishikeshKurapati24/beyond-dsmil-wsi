# Designated code author: Hrishikesh Kurapati
# Scope/provenance: Phase 4 experiments 3 and 4; completed validation variants and planned cumulative variants.
# Attribution header added to this review copy; original source remains unchanged.

"""Validation-only Phase 4 experiments 3 and 4 on prepared CAMELYON16-UNI."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F

import phase1_to_3_dsmil_runner as base
import pre_phase4_ab_runner as ab
import phase4_uni_spatial_runner as p4

SEEDS = (42, 43, 44)
FOLDS = range(5)
K = 4
GAP = 8.0
SEARCH_WIDTH = 64
POOL_SIZES = (32, 64, 128, 256)
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5)
EPOCHS, PATIENCE = 50, 10
LR, WEIGHT_DECAY = 1e-4, 1e-3
GRID_WIDTH = 4
FEATURE_WEIGHT, SPACE_WEIGHT = 0.5, 0.5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_rows(values: np.ndarray, ids: np.ndarray, count: int) -> list[int]:
    return np.lexsort((ids, -values))[:min(count, len(values))].tolist()


def spaced_beam(scores: torch.Tensor, coords: torch.Tensor, ids: torch.Tensor,
                width: int = SEARCH_WIDTH, pool_sizes=POOL_SIZES):
    """One anchor per head; maximize count, then minimize within-head rank sum."""
    n, heads = scores.shape
    if n < 1 or coords.shape != (n, 2) or heads < 1:
        raise ValueError("Invalid multi-head selection input")
    values = scores.detach().cpu().numpy()
    xy = coords.detach().cpu().numpy()
    original = ids.detach().cpu().numpy()
    best, trace = None, []
    for requested in pool_sizes:
        size = min(requested, n)
        lists = [top_rows(values[:, k], original, size) for k in range(heads)]
        # State: selected row per head (-1 means inactive), active count, rank cost.
        beams = {0: [((), 0, 0)]}
        for k in range(heads):
            next_beams: dict[int, list[tuple]] = {}
            for states in beams.values():
                for selected, count, rank_cost in states:
                    next_beams.setdefault(count, []).append((selected + (-1,), count, rank_cost))
                    active = [j for j in selected if j >= 0]
                    for rank, row in enumerate(lists[k], 1):
                        if all(np.linalg.norm(xy[row] - xy[j]) >= GAP - 1e-6 for j in active):
                            next_beams.setdefault(count + 1, []).append(
                                (selected + (row,), count + 1, rank_cost + rank))
            beams = {count: sorted(states, key=lambda st: (
                st[2], tuple(int(original[i]) if i >= 0 else n + k for i in st[0])))[:width]
                     for count, states in next_beams.items()}
        count = max(beams)
        candidate = beams[count][0]
        trace.append({"pool_size": size, "selected_heads": count})
        if best is None or (candidate[1], -candidate[2]) > (best[1], -best[2]):
            best = candidate
        if count == heads or size == n:
            break
    selected = best[0]
    return [(k, row) for k, row in enumerate(selected) if row >= 0], trace


class MultiHeadSpatial(nn.Module):
    def __init__(self, spatial: bool, lam: float, heads: int = K):
        super().__init__()
        self.scorer = nn.Linear(1024, heads)
        self.q = base.BClassifier(1024, 1).q
        self.fcc = nn.Linear(1024, 1)
        self.spatial, self.lam, self.heads = spatial, lam, heads

    def forward(self, x, coords, ids=None, label=None, diagnostics=True):
        n = len(x)
        if ids is None:
            ids = torch.arange(n, device=x.device)
        scores = self.scorer(x)
        if self.spatial:
            chosen, search = spaced_beam(scores, coords, ids)
        else:
            values, original = scores.detach().cpu().numpy(), ids.detach().cpu().numpy()
            chosen = [(k, top_rows(values[:, k], original, 1)[0]) for k in range(self.heads)]
            search = [{"pool_size": 1, "selected_heads": self.heads}]
        m = len(chosen)
        if m < 1:
            raise ValueError("Selector must retain at least one head")
        q = self.q(x)
        anchors, candidate_distributions = [], []
        for head, row in chosen:
            candidates = top_rows(scores[:, head].detach().cpu().numpy(),
                                  ids.detach().cpu().numpy(), 32)
            if row not in candidates:
                candidates.append(row)
            candidate = torch.as_tensor(candidates, device=x.device, dtype=torch.long)
            r = torch.softmax(scores[candidate, head], dim=0)
            candidate_distributions.append((candidate, r))
            if self.training:
                eligible = [i for i in candidates if (not self.spatial or all(
                    torch.linalg.vector_norm(coords[i] - coords[other]).item() >= GAP - 1e-6
                    for h, other in chosen if h != head))]
                if row not in eligible:
                    raise ValueError("Selected anchor missing from eligible soft support")
                index = torch.as_tensor(eligible, device=x.device, dtype=torch.long)
                probability = torch.softmax(scores[index, head], dim=0)
                hard = torch.as_tensor([float(i == row) for i in eligible], device=x.device)
                surrogate = hard + probability - probability.detach()
                anchor = surrogate @ x[index]
            else:
                anchor = x[row]
            anchors.append(anchor)
        anchor_queries = self.q(torch.stack(anchors))
        attention = torch.softmax(q @ anchor_queries.T / math.sqrt(128), dim=0)
        bags = attention.T @ x
        bag_logit = self.fcc(bags.mean(dim=0, keepdim=True)).view(())
        selected_logits = torch.stack([scores[row, head] for head, row in chosen])
        smooth_max = torch.logsumexp(selected_logits, dim=0) - math.log(m)
        repulsion = bag_logit.new_zeros(())
        if self.spatial and m > 1:
            terms = []
            for left in range(m):
                a, pa = candidate_distributions[left]
                for right in range(left + 1, m):
                    b, pb = candidate_distributions[right]
                    distances = torch.cdist(coords[a], coords[b])
                    violation = torch.clamp(1 - distances / GAP, min=0).square()
                    terms.append((pa[:, None] * pb[None, :] * violation).sum())
            repulsion = torch.stack(terms).mean()
        aux = bag_logit.new_zeros(())
        if label is not None:
            target = torch.as_tensor(float(label), device=x.device)
            instance = (F.softplus(-smooth_max) if label == 1 else
                        F.softplus(selected_logits).mean())
            aux = 0.5 * F.binary_cross_entropy_with_logits(bag_logit, target)
            aux = aux + 0.5 * instance + self.lam * repulsion
        if not diagnostics:
            return bag_logit, aux, {}
        selected_rows = [int(ids[row]) for _, row in chosen]
        pair_distances = [float(torch.linalg.vector_norm(coords[chosen[i][1]] -
                                coords[chosen[j][1]]).detach())
                          for i in range(m) for j in range(i + 1, m)]
        normalized = F.normalize(x.detach(), dim=1)
        dissimilarities = [float(1 - torch.dot(normalized[chosen[i][1]],
                                               normalized[chosen[j][1]]))
                           for i in range(m) for j in range(i + 1, m)]
        bag_norm = F.normalize(bags.detach(), dim=1)
        bag_cosines = [float(torch.dot(bag_norm[i], bag_norm[j]))
                       for i in range(m) for j in range(i + 1, m)]
        info = {"heads": [head for head, _ in chosen], "anchor_rows": selected_rows,
                "active_count": m, "pairwise_distances_patch_widths": pair_distances,
                "pairwise_feature_cosine_dissimilarity": dissimilarities,
                "pairwise_bag_cosine": bag_cosines,
                "anchor_coordinates_patch_widths": coords[[row for _, row in chosen]].detach().cpu().tolist(),
                "anchor_selected_scores": selected_logits.detach().cpu().tolist(),
                "anchor_within_head_ranks": [int((scores[:, head] > scores[row, head]).sum()) + 1
                                              for head, row in chosen],
                "search": search, "attention_columns": m,
                "attention_sums": attention.sum(dim=0).detach().cpu().tolist(),
                "attention_entropy": [float(-(attention[:, k] *
                    torch.log(attention[:, k].clamp_min(1e-12))).sum().detach()) for k in range(m)],
                "spatial_penalty": float(repulsion.detach())}
        return bag_logit, aux, info


class FlatTopKUNI(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_classifier = base.FCLayer(1024, 1)
        self.b_classifier = base.BClassifier(1024, 1, anchors=K)

    def forward(self, x, coords, ids=None, label=None, diagnostics=True):
        _, scores = self.i_classifier(x)
        order = top_rows(scores[:, 0].detach().cpu().numpy(),
                         np.arange(len(x)) if ids is None else ids.detach().cpu().numpy(), K)
        q = self.b_classifier.q(x)
        attention = torch.softmax(q @ q[order].T / math.sqrt(128), dim=0)
        anchor_bags = attention.T @ self.b_classifier.v(x)
        bag = self.b_classifier.fcc(anchor_bags.mean(dim=0).view(1, 1, -1)).view(())
        loss = bag.new_zeros(())
        if label is not None:
            target = torch.as_tensor(float(label), device=x.device)
            loss = 0.5 * (F.binary_cross_entropy_with_logits(bag, target) +
                          F.binary_cross_entropy_with_logits(scores.max(), target))
        info = ({"anchor_rows": order if ids is None else ids[order].cpu().tolist(),
                 "active_count": min(K, len(x)), "attention_columns": attention.shape[1],
                 "attention_sums": attention.sum(dim=0).detach().cpu().tolist()}
                if diagnostics else {})
        return bag, loss, info


def hierarchy_indices(coords: torch.Tensor, x: torch.Tensor, anchors: list[int],
                      space_weight: float, grid: int = GRID_WIDTH):
    """Anchor-dependent patch partition and parent maps, without N² distances."""
    xy = coords.detach().cpu().numpy()
    features = x.detach().cpu().numpy()
    cell = np.floor(xy / grid).astype(np.int64)
    per_cell: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(cell):
        per_cell.setdefault(tuple(key), []).append(i)
    groups = []
    for key, members in sorted(per_cell.items()):
        present = [a for a in anchors if a in members]
        if len(present) <= 1:
            groups.append((key, members, present[0] if present else None))
        else:
            partitions = {a: [] for a in present}
            for i in members:
                owner = (i if i in partitions else
                         min(present, key=lambda a: (np.linalg.norm(xy[i]-xy[a]), anchors.index(a))))
                partitions[owner].append(i)
            groups.extend((key, rows, a) for a, rows in partitions.items())
    branch_groups: list[list[int]] = [[] for _ in anchors]
    for group_index, (key, members, own_anchor) in enumerate(groups):
        if own_anchor is not None:
            branch = anchors.index(own_anchor)
        else:
            mean_feature = features[members].mean(axis=0)
            mean_feature = mean_feature / max(np.linalg.norm(mean_feature), 1e-12)
            centroid = xy[members].mean(axis=0)
            ratings = []
            for a in anchors:
                delta = float(np.linalg.norm(centroid-xy[a]))
                anchor_feature = features[a] / max(np.linalg.norm(features[a]), 1e-12)
                ratings.append(FEATURE_WEIGHT*float(mean_feature @ anchor_feature) -
                               space_weight*delta/(delta+GAP))
            branch = int(np.argmax(ratings))
        branch_groups[branch].append(group_index)
    if any(not group for group in branch_groups):
        raise ValueError("Anchor branch has no microgroup")
    regions: dict[tuple[int, int, int], list[int]] = {}
    group_branch = {}
    for branch, members in enumerate(branch_groups):
        for gi in members:
            group_branch[gi] = branch
            key = groups[gi][0]
            regions.setdefault((branch, key[0]//2, key[1]//2), []).append(gi)
    region_list = [regions[key] for key in sorted(regions)]
    region_branch = [key[0] for key in sorted(regions)]
    parent = {gi: region for region, child in enumerate(region_list) for gi in child}
    assert sorted(i for _, members, _ in groups for i in members) == list(range(len(x)))
    return groups, branch_groups, region_list, region_branch, parent, group_branch


class HierarchicalTopK(nn.Module):
    def __init__(self, level: int, space_weight: float, grid: int = GRID_WIDTH):
        super().__init__()
        self.i_classifier = base.FCLayer(1024, 1)
        self.b_classifier = base.BClassifier(1024, 1, anchors=K)
        self.heads = nn.ModuleList(nn.Linear(1024, 1) for _ in range(3)) if level >= 2 else None
        self.level, self.space_weight, self.grid = level, space_weight, grid

    def pool(self, children: torch.Tensor, segment: torch.Tensor, anchors: torch.Tensor):
        """Anchor-conditioned local softmax with linear memory in child count."""
        count = len(anchors)
        query = self.b_classifier.q(children)
        anchor_query = self.b_classifier.q(anchors)
        logits = (query * anchor_query[segment]).sum(dim=1) / math.sqrt(128)
        maxima = logits.new_full((count,), -torch.inf).scatter_reduce(
            0, segment, logits, reduce="amax", include_self=True)
        exp = torch.exp(logits - maxima[segment])
        denominator = logits.new_zeros(count).index_add(0, segment, exp)
        weights = exp / denominator[segment]
        return children.new_zeros((count, children.shape[1])).index_add(
            0, segment, children * weights[:, None])

    def forward(self, x, coords, ids=None, label=None, diagnostics=True):
        _, scores = self.i_classifier(x)
        order = top_rows(scores[:, 0].detach().cpu().numpy(),
                         np.arange(len(x)) if ids is None else ids.detach().cpu().numpy(), K)
        anchors = x[order]
        groups, branch_groups, regions, region_branch, parent, group_branch = hierarchy_indices(
            coords, x, order, self.space_weight, self.grid)
        device = x.device
        patch_group = torch.empty(len(x), dtype=torch.long, device=device)
        for gi, (_, members, _) in enumerate(groups):
            patch_group[torch.as_tensor(members, device=device)] = gi
        group_branch_t = torch.as_tensor([group_branch[gi] for gi in range(len(groups))],
                                         device=device, dtype=torch.long)
        parent_t = torch.as_tensor([parent[gi] for gi in range(len(groups))],
                                   device=device, dtype=torch.long)
        region_branch_t = torch.as_tensor(region_branch, device=device, dtype=torch.long)
        group_vectors = self.pool(x, patch_group, anchors[group_branch_t])
        region_vectors = self.pool(group_vectors, parent_t, anchors[region_branch_t])
        root_vectors = self.pool(region_vectors, region_branch_t, anchors)
        bag_logit = self.b_classifier.fcc(root_vectors.mean(dim=0).view(1, 1, -1)).view(())
        objective = bag_logit.new_zeros(())
        components = {}
        if label is not None:
            target = torch.as_tensor(float(label), device=x.device)
            components["bag_loss"] = 0.5 * F.binary_cross_entropy_with_logits(bag_logit, target)
            components["instance_loss"] = 0.5 * F.binary_cross_entropy_with_logits(scores.max(), target)
            objective = components["bag_loss"] + components["instance_loss"]
        level_vectors = [group_vectors, region_vectors, root_vectors]
        level_branches = [group_branch_t, region_branch_t,
                          torch.arange(len(order), device=device)]
        if self.level >= 2:
            level_logits = []
            level_probabilities = []
            level_noisy_or = []
            for ell, vectors in enumerate(level_vectors):
                summaries = self.pool(vectors, level_branches[ell], anchors)
                logits = self.heads[ell](summaries).view(-1)
                level_logits.append(logits)
                if diagnostics:
                    level_probabilities.append(torch.sigmoid(logits).detach().cpu().tolist())
                    level_noisy_or.append(float((-torch.expm1(-F.softplus(logits).sum())).detach()))
            if label is not None:
                losses = []
                for logits in level_logits:
                    log_q = -F.softplus(logits).sum()
                    losses.append(-torch.log(torch.clamp(-torch.expm1(log_q), min=1e-7))
                                  if label == 1 else -log_q)
                components["hier_loss"] = torch.stack(losses).mean()
                objective = objective + 0.1 * components["hier_loss"]
        if self.level >= 3:
            node_logits = [head(vectors).view(-1) for head, vectors in zip(self.heads, level_vectors)]
            paths = (F.logsigmoid(node_logits[0]) +
                     F.logsigmoid(node_logits[1][parent_t]) +
                     F.logsigmoid(node_logits[2][group_branch_t])) / 3
            soft_log = 0.1 * (torch.logsumexp(paths / 0.1, dim=0) - math.log(len(paths)))
            if diagnostics:
                strongest = int(paths.argmax().detach())
            if label is not None:
                components["path_loss"] = (-soft_log if label == 1 else
                    -torch.log1p(-torch.exp(soft_log).clamp(max=1-1e-7)))
                objective = objective + 0.1 * components["path_loss"]
        if not diagnostics:
            return bag_logit, objective, {}
        info = {"anchor_rows": order if ids is None else ids[order].cpu().tolist(),
                "anchor_coordinates_patch_widths": coords[order].detach().cpu().tolist(),
                "active_count": len(order), "microgroups": len(groups),
                "regions": len(regions), "branch_group_counts": [len(a) for a in branch_groups],
                "group_sizes": [len(members) for _, members, _ in groups],
                "region_group_counts": [len(child) for child in regions],
                "group_parent_region": [parent[gi] for gi in range(len(groups))],
                "group_branch": [group_branch[gi] for gi in range(len(groups))],
                "region_branch": region_branch,
                "loss_components": {k: float(v.detach()) for k, v in components.items()}}
        if self.level >= 2:
            info["level_branch_probabilities"] = level_probabilities
            info["level_noisy_or_probabilities"] = level_noisy_or
        if self.level >= 3:
            info["path_probability"] = float(torch.exp(soft_log).detach())
            info["strongest_path"] = {"group": strongest, "region": parent[strongest],
                                       "branch": group_branch[strongest],
                                       "node_probabilities": [
                                           float(torch.sigmoid(node_logits[0][strongest]).detach()),
                                           float(torch.sigmoid(node_logits[1][parent[strongest]]).detach()),
                                           float(torch.sigmoid(node_logits[2][group_branch[strongest]]).detach())]}
        return bag_logit, objective, info


VARIANTS = {
    "exp3_control": ("exp3", lambda: MultiHeadSpatial(False, 0.0)),
    **{f"exp3_spatial_lam_{int(lam*100):03d}": ("exp3", lambda lam=lam: MultiHeadSpatial(True, lam))
       for lam in LAMBDA_GRID},
    "exp4_flat_topk_uni": ("exp4", FlatTopKUNI),
    "exp4_4a_feature_assignment": ("exp4", lambda: HierarchicalTopK(1, 0.0, GRID_WIDTH)),
    "exp4_4a_spatial": ("exp4", lambda: HierarchicalTopK(1, SPACE_WEIGHT, GRID_WIDTH)),
    "exp4_4b_intermediate": ("exp4", lambda: HierarchicalTopK(2, SPACE_WEIGHT, GRID_WIDTH)),
    "exp4_4c_path": ("exp4", lambda: HierarchicalTopK(3, SPACE_WEIGHT, GRID_WIDTH)),
}


def make_model(name: str, device: torch.device):
    model = VARIANTS[name][1]().to(device)
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.9),
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=5e-6)
    return model, optimizer, scheduler


def evaluate(model, rows, prepared, metadata, device, diagnostics=False):
    model.eval()
    records, details, losses = [], [], []
    with torch.no_grad():
        for row in rows:
            x, coords = p4.load_uni(prepared, row, device, metadata, spatial=True)
            logit, loss, info = model(x, coords, label=row["label"], diagnostics=diagnostics)
            losses.append(float(loss))
            records.append({"slide_id": row["slide_id"], "label": row["label"],
                            "probability": float(torch.sigmoid(logit))})
            if diagnostics:
                details.append({"slide_id": row["slide_id"], "label": row["label"], **info})
    return float(np.mean(losses)), records, details


def train_variant(name: str, rows: list[dict], prepared: Path, metadata: dict,
                  output: Path, device: torch.device):
    output.mkdir(parents=True, exist_ok=True)
    config = {"variant": name, "seeds": list(SEEDS), "folds": list(FOLDS),
              "grid_width_patch_units": GRID_WIDTH if name.startswith("exp4_") else None,
              "assignment_feature_weight": FEATURE_WEIGHT if name.startswith("exp4_") else None,
              "assignment_spatial_weight": SPACE_WEIGHT if name.startswith("exp4_") else None,
              "prepared_manifest_sha256": sha(prepared / "slides.csv"),
              "coordinate_audit_sha256": sha(Path(metadata["_audit_path"])),
              "runner_sha256": sha(Path(__file__)),
              "baseline_sha256": sha(Path(base.__file__)),
              "phase4_loader_sha256": sha(Path(p4.__file__)),
              "ab_runner_sha256": sha(Path(ab.__file__)),
              "test_evaluated": False}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Existing output has incompatible configuration: {output}")
    base.write_json(config_path, config)
    all_oof, results = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            folder = output / f"seed_{seed}" / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            train = [r for r in rows if r["official_split"] == "train" and r["validation_fold"] != fold]
            valid = [r for r in rows if r["official_split"] == "train" and r["validation_fold"] == fold]
            if len(train) != 216 or len(valid) != 54:
                raise ValueError("Incorrect UNI fold size")
            complete = folder / "COMPLETE.json"
            if complete.exists():
                summary = json.loads(complete.read_text())
                predictions = base.read_csv(folder / "oof.csv")
                results.append(summary)
                all_oof.extend({"seed": seed, "fold": fold, "slide_id": r["slide_id"],
                                "label": int(r["label"]), "probability": float(r["probability"])}
                               for r in predictions)
                continue
            base.seed_all(seed)
            model, optimizer, scheduler = make_model(name, device)
            history, best_auc, best_loss, best_epoch, stale = [], -1.0, float("inf"), 0, 0
            start = time.monotonic()
            for epoch in range(1, EPOCHS + 1):
                model.train()
                order = np.random.default_rng(seed + fold * 100000 + epoch).permutation(len(train))
                training_losses = []
                for index in order:
                    row = train[int(index)]
                    x, coords = p4.load_uni(prepared, row, device, metadata, spatial=True)
                    permutation = torch.randperm(len(x), device=device)
                    x, coords = x[permutation], coords[permutation]
                    optimizer.zero_grad(set_to_none=True)
                    _, loss, _ = model(x, coords, ids=permutation, label=row["label"],
                                       diagnostics=False)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite loss: {name} {row['slide_id']}")
                    loss.backward()
                    optimizer.step()
                    training_losses.append(float(loss.detach()))
                val_loss, predicted, _ = evaluate(model, valid, prepared, metadata, device)
                val_auc = base.auc(predicted)
                history.append({"epoch": epoch, "train_loss": float(np.mean(training_losses)),
                                "val_loss": val_loss, "val_auc": val_auc,
                                "val_accuracy_0.5": float(accuracy_score(
                                    [r["label"] for r in predicted],
                                    [r["probability"] >= 0.5 for r in predicted])),
                                "lr": optimizer.param_groups[0]["lr"]})
                improved = (val_auc > best_auc + 1e-12 or
                            (abs(val_auc-best_auc) <= 1e-12 and val_loss < best_loss))
                if improved:
                    best_auc, best_loss, best_epoch, stale = val_auc, val_loss, epoch, 0
                    torch.save(model.state_dict(), folder / "checkpoint.pt")
                    base.write_csv(folder / "oof.csv", predicted,
                                   ["slide_id", "label", "probability"])
                else:
                    stale += 1
                scheduler.step()
                print(f"{name} seed={seed} fold={fold} epoch={epoch} "
                      f"train={history[-1]['train_loss']:.4f} val={val_loss:.4f} auc={val_auc:.4f}",
                      flush=True)
                if stale >= PATIENCE:
                    break
            base.write_csv(folder / "history.csv", history, list(history[0]))
            model.load_state_dict(torch.load(folder / "checkpoint.pt", map_location=device,
                                             weights_only=True))
            train_loss, train_pred, train_diag = evaluate(model, train, prepared, metadata,
                                                          device, diagnostics=True)
            _, _, valid_diag = evaluate(model, valid, prepared, metadata,
                                        device, diagnostics=True)
            base.write_json(folder / "diagnostics.json", {"train": train_diag, "validation": valid_diag})
            selected = base.read_csv(folder / "oof.csv")
            all_oof.extend({"seed": seed, "fold": fold, "slide_id": r["slide_id"],
                            "label": int(r["label"]), "probability": float(r["probability"])}
                           for r in selected)
            summary = {"seed": seed, "fold": fold, "best_epoch": best_epoch,
                       "validation_auc": best_auc, "validation_loss": best_loss,
                       "training_loss": train_loss, "training_auc": base.auc(train_pred),
                       "elapsed_seconds": round(time.monotonic()-start, 1),
                       "diagnostic_slide_count": len(train_diag)+len(valid_diag)}
            base.write_json(complete, summary)
            results.append(summary)
            del model, optimizer, scheduler
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(results) != 15 or len(all_oof) != 810:
        raise ValueError(f"Incomplete {name}")
    base.write_csv(output / "oof_predictions_all_seeds.csv", all_oof,
                   ["seed", "fold", "slide_id", "label", "probability"])
    ensemble = ab.ensemble_oof(all_oof)
    base.write_csv(output / "oof_predictions_ensemble.csv", ensemble,
                   ["slide_id", "fold", "label", "probability"])
    cv = {"variant": name, "runs": results, "mean_validation_auc": float(np.mean(
          [r["validation_auc"] for r in results])),
          "mean_validation_loss": float(np.mean([r["validation_loss"] for r in results])),
          "oof_ensemble_auc": base.auc(ensemble), "test_evaluated": False}
    base.write_json(output / "cv_summary.json", cv)
    base.write_json(output / "COMPLETE.json", {"status": "complete", "fits": 15,
                                                 "oof_rows": 810, "test_evaluated": False})
    return cv


def write_comparisons(output: Path):
    available = {name: json.loads((output / name / "cv_summary.json").read_text())
                 for name in VARIANTS if (output / name / "COMPLETE.json").is_file()}
    rows = [{"variant": name, "mean_validation_auc": cv["mean_validation_auc"],
             "mean_validation_loss": cv["mean_validation_loss"],
             "oof_ensemble_auc": cv["oof_ensemble_auc"]}
            for name, cv in available.items()]
    if rows:
        base.write_csv(output / "comparison_table.csv", rows, list(rows[0]))
    pairs = [("exp3_control", name) for name in VARIANTS if name.startswith("exp3_spatial_")]
    pairs += [("exp4_flat_topk_uni", "exp4_4a_spatial"),
              ("exp4_4a_feature_assignment", "exp4_4a_spatial"),
              ("exp4_4a_spatial", "exp4_4b_intermediate"),
              ("exp4_4b_intermediate", "exp4_4c_path")]
    differences = []
    for reference, candidate in pairs:
        if reference not in available or candidate not in available:
            continue
        left = {(r["seed"], r["fold"]): r for r in available[reference]["runs"]}
        right = {(r["seed"], r["fold"]): r for r in available[candidate]["runs"]}
        if set(left) != set(right) or len(left) != 15:
            raise ValueError(f"Mismatched fold/seed comparison: {reference}, {candidate}")
        differences.extend({"reference": reference, "candidate": candidate,
                            "seed": seed, "fold": fold,
                            "validation_auc_difference": right[(seed, fold)]["validation_auc"] -
                            left[(seed, fold)]["validation_auc"],
                            "validation_loss_difference": right[(seed, fold)]["validation_loss"] -
                            left[(seed, fold)]["validation_loss"]}
                           for seed, fold in sorted(left))
    if differences:
        base.write_csv(output / "paired_fold_differences.csv", differences,
                       list(differences[0]))
    if len(available) == len(VARIANTS):
        base.write_json(output / "COMPLETE.json", {"status": "complete", "fits": 150,
                                                  "variants": list(VARIANTS),
                                                  "test_evaluated": False})


def main():
    global GRID_WIDTH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("exp3", "exp4", "all", *VARIANTS), default="all")
    parser.add_argument("--grid-width", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    GRID_WIDTH = args.grid_width
    rows, _ = p4.load_uni_manifest(args.prepared)
    metadata = p4.load_coordinate_metadata(args.audit, rows)
    metadata["_audit_path"] = str(args.audit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [name for name, (stage, _) in VARIANTS.items()
                if args.stage == "all" or stage == args.stage or name == args.stage]
    summaries = {}
    for name in selected:
        summaries[name] = train_variant(name, rows, args.prepared, metadata,
                                        args.output / name, device)
    for stage in ("exp3", "exp4"):
        names = [name for name, (family, _) in VARIANTS.items() if family == stage]
        if all((args.output / name / "COMPLETE.json").exists() for name in names):
            candidates = [(name, json.loads((args.output / name / "cv_summary.json").read_text()))
                          for name in names if name.startswith("exp3_spatial_lam_")]
            selection = (max(candidates, key=lambda item: (item[1]["mean_validation_auc"],
                          -item[1]["mean_validation_loss"], -float(item[0].split("_")[-1])))
                         [0] if candidates else None)
            base.write_json(args.output / f"{stage}_COMPLETE.json",
                            {"status": "complete", "variants": names,
                             "selected_spatial_variant": selection, "test_evaluated": False})
    write_comparisons(args.output)
    base.write_json(args.output / "PHASE4_EXPERIMENTS_3_4.json",
                    {"stage_run": args.stage, "completed_variants": list(summaries),
                     "test_evaluated": False})


if __name__ == "__main__":
    main()
