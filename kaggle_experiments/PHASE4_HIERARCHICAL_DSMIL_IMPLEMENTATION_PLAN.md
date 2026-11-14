# Phase 4 experiment 4: spatial hierarchical Top-K DSMIL

**Status:** implemented in `phase4_exp3_4_runner.py` and the combined Kaggle notebook `05_phase4_experiments_3_and_4.ipynb`; the Kaggle run has not been executed. This adapts the user-supplied spatially guided hierarchical DSMIL proposal to the established Kaggle Phase 4 protocol. Experiment 4 has three **cumulative, independently trained** validation subruns: **4(a)** spatial hierarchy, **4(b)** 4(a) plus intermediate MIL supervision, and **4(c)** 4(b) plus nested positive-path supervision. The attached proposal's names B2/B3/B4 correspond to 4(a)/4(b)/4(c). No Phase 4 test-set evaluation belongs in these runs.

## Fixed decisions and matched reference

- **Feature source:** prepared CAMELYON16-UNI 1024-dimensional embeddings and row-aligned level-0 patch coordinates. Do not train UNI or process raw whole-slide images in the training notebook. Normalize each slide's coordinate differences by its audited source `patch_size_px`; one unit is one UNI patch width, approximately 128 µm.
- **Starting architecture:** the existing **flat Top-K DSMIL rule**, with **one** linear instance scorer and `K=4` highest-scoring anchors. This is the Phase 2 `K=4, attention-overlap weight=0` mechanism adapted from 512- to 1024-dimensional UNI input and trained afresh. It is **not** Phase 4 experiment 1's A1 score–feature selector, experiments 2(a)/2(b)'s spatial selector/attention bias, or experiment 3's independent multi-head scorers. No A1 alpha, 2(a)/2(b) gamma, or experiment 3 repulsion term carries into experiment 4.
- **Matched flat reference:** train flat Top-K DSMIL on UNI using the same fold/seed protocol. No existing 1024-dimensional flat Top-K checkpoints can be substituted. Its instance stream, shared `q:R^1024→R^128`, identity values `v(x)=x`, four per-anchor full-slide attention maps, mean of four bag vectors, and affine bag classifier define the comparison architecture. Flat reference loss is `0.5 BCEWithLogits(max_i s_i,y)+0.5 BCEWithLogits(g_b,y)`.
- **Study protocol:** official 270 CAMELYON16 training slides, existing five stratified folds (54 validation slides each), seeds 42/43/44, freshly initialized models for every fold/seed/variant, 50-epoch maximum, patience 10, Adam LR `1e-4`, betas `(0.5,0.9)`, weight decay `1e-3`, cosine schedule to `5e-6`, and checkpoint by highest validation ROC-AUC with validation-loss tie-break. Use the same classification threshold procedure if reporting thresholded OOF metrics. All variants use bag-logit sigmoid for slide probabilities. The 129 official-test bags are never opened for fitting, tuning, diagnostics, or evaluation in this experiment.
- **Platform:** Kaggle. Profile the actual GPU and notebook time allowance instead of assuming a Colab T4 or a ten-hour session. Keep stages separately savable/resumable if the complete comparison will not fit in one Kaggle session.

The question is whether a *partitioned spatial hierarchy* preserves cancer evidence better than the matched **flat Top-K UNI** model. Unlike flat DSMIL and Phase 4 experiments 1–3, every patch in 4(a)–4(c) belongs to exactly one branch and one parent at each local level. This architectural change must be explicit when interpreting results.

## Data validation and reusable spatial cache

Read `bags/<slide>.npy`, `coords/<slide>.npy`, and the prepared UNI manifest. Reuse the existing Phase 4 UNI loader/audit where possible. Check feature dimension, dtype, finiteness, coordinate row count/order, bounds, unique slide IDs, labels, and fixed folds. Load or embed audited `patch_size_px`, `patch_level=0`, and `level_dim` for **training slides only**. The source states embedding and coordinate rows correspond in order; retain the source audit as provenance rather than asserting independent proof of patch identity.

For each slide, precompute **anchor-independent** grid-cell IDs, neighborhood/adjacency information, patch-to-cell indices, and static coordinate summaries. Cache them by slide ID, coordinate checksum, patch width, and grouping configuration. The chosen anchors change as the scorer trains, so anchor-dependent cell splitting, branch assignment, and parent maps **must be rebuilt** when the anchors change; do not cache a stale full hierarchy across epochs.

## 4(a): spatial hierarchy, original two-stream loss

### 1. Score and select anchors

For a bag `X={x_i}_{i=1}^N`, use the single existing instance scorer `s_i=w_Iᵀx_i+b_I`. Choose `a_1,…,a_4` as the four highest `s_i` values, with original row ID as deterministic tie-break. If `N<4`, use `K_eff=min(4,N)` and record it; the real cohort is expected to have much larger bags. Do **not** add an eight-width separation rule to this experiment: it would change the attached Top-K proposal and confound its hierarchical-pooling question. Preserve `g_I=max_i s_i` and its ordinary instance loss. A selected anchor is a high-scoring candidate, not a verified cancer patch.

### 2. Partition patches into local microgroups

Convert coordinates to patch-width units `r_i=t_i/P`. For a fixed grid width `g` in patch widths, assign each patch to cell `c_i=(floor(r_ix/g),floor(r_iy/g))`. This gives exactly one initial microgroup per patch, including irregular tissue edges and singleton cells. Start with `g=4`; inspect training-slide occupancy and, if needed, choose `g=8` **before** full model comparison to target roughly 16–64 patches per occupied microgroup. Freeze the choice for the flat-reference comparison diagnostics and all 4(a)–4(c) variants; do not pick it using official-test data. For group `G_u`, retain member row IDs, arithmetic-mean feature `xbar_u`, and coordinate centroid `rbar_u`.

If a grid cell contains multiple Top-K anchors, split that cell into one sub-microgroup per anchor, assigning its other member patches to the closest anchor within the cell (deterministic row-ID tie-break). Each anchor's own patch stays in its own nonempty sub-microgroup. Splitting is recomputed when anchors change. No patch is duplicated or discarded.

### 3. Assign every microgroup to one anchor branch

For each microgroup `u` and anchor `j`, calculate a feature–space assignment score in dimensionless units:

`delta_uj=||rbar_u−r_(a_j)||₂`; `R_uj = w_feat*cos(xbar_u,x_(a_j)) − w_space*[delta_uj/(delta_uj+8)]`.

Here `delta_uj` is Euclidean distance in patch-width units and `w_feat,w_space≥0` are **branch-assignment** coefficients; they are unrelated to A1's selector alpha. Start with `w_feat=w_space=0.5`, and freeze both after coordinate-only scale checks or a small training-fold-only pilot. Assign `u` to `argmax_j R_uj`, using anchor rank as tie-break, **except** force each anchor-containing microgroup into its own anchor's branch. The four branch sets partition every microgroup and hence every input patch. Branch sizes need not match. Recompute this assignment for each forward pass if anchor identities change.

Include a matched **feature-only branch-assignment ablation** with `w_space=0` and all other hierarchy components unchanged. Its microgroups and coarsening remain spatial; therefore it isolates the *spatial term in branch assignment*, **not** all spatial information from the architecture.

### 4. Build a three-level rooted forest

- **Level 1:** learned attention-pool the patches inside each assigned spatial microgroup to one `1024`-dimensional microgroup representation.
- **Level 2:** within each branch, combine neighboring Level-1 microgroups in deterministic `2×2` coarser grid cells. Use a pass-through parent for singleton cells. No parent combines nodes from different branches. If a branch has separated islands, retain separate Level-2 regions rather than joining distant cells locally.
- **Level 3:** attention-pool all Level-2 regions in each branch to one branch-root representation `R_j`. This final branch aggregation need not imply that the entire branch is one contiguous region.

Every patch has one Level-1 group, every Level-1 group has one Level-2 parent, and every Level-2 region has one branch root. Store integer parent IDs and child lists, including the Level-1 anchor-containing group and all singleton paths. The four roots jointly cover all patches exactly once.

For a parent `u` in branch `j`, let its child representations be `r_v`. Reuse the shared DSMIL query/value structure where dimensions match, with local attention

`e_uv=q(r_v)ᵀq(x_(a_j))/sqrt(128)`; `A_uv=softmax_(v∈children(u))(e_uv)`; `r_u=Σ_v A_uv*r_v`.

The primary run uses identity values and **no extra relative-distance attention bias**, so spatial information enters through grouping and branch assignment. A configurable max-evidence residual `Project([r_u || max_v r_v])` can be implemented for a later controlled ablation, but it is **off in the primary 4(a)–4(c) comparison** to avoid changing their capacity while studying the three planned mechanisms. Avoid dense `N×N` patch attention or distance matrices.

### 5. Final prediction and objective

Average the four branch roots, `z_b=(1/K_eff)Σ_(j=1)^(K_eff) R_j`, and apply the same affine bag classifier form as flat Top-K UNI: `g_b=w_Bᵀz_b+b_B`. Keep the original instance stream. The sole 4(a) objective is

`L_4a = 0.5*BCEWithLogits(g_I,y) + 0.5*BCEWithLogits(g_b,y)`.

There is no intermediate or path supervision in 4(a). Its primary comparison is the freshly trained flat Top-K UNI reference; the `w_space=0` ablation tests branch-assignment spatial information within the same hierarchy.

## 4(b): 4(a) plus intermediate existential MIL supervision

Keep 4(a)'s anchor selection, microgroups, assignments, hierarchy, local pooling, final bag prediction, and optimizer protocol unchanged. Train a **fresh model from the matched seed initialization protocol**, rather than warm-starting 4(a)'s selected checkpoint; only the auxiliary heads/loss differ.

For each level `ell∈{1,2,3}` and branch `j`, form a branch-wide summary `R_j^(ell)` by lightweight anchor-conditioned attention over *all current nodes in that branch at that level*. At Level 3 this is its root. The summary is distinct from an individual node at Levels 1–2. Apply a small level-specific logit head `g_ell:R^1024→R` to obtain `z_j^(ell)=g_ell(R_j^(ell))` and `p_j^(ell)=sigmoid(z_j^(ell))`. Allocate these same `g_ell` modules in 4(b) so 4(c) can apply them to individual nodes without adding new classifier parameters solely in 4(c).

At each level, positive evidence is existential across the four branches:

`Q_ell=Π_j(1−p_j^(ell))`; `P_ell=1−Q_ell`; `L_hier=(1/3)Σ_(ell=1)^3 BCE(P_ell,y)`.

Thus a positive slide needs at least one positive branch per level; a negative slide pushes all branches low. Do **not** label every branch positive on a positive slide, or claim that `p_j^(ell)` is a ground-truth regional cancer probability. Compute this stably from logits: `log Q_ell=−Σ_j softplus(z_j^(ell))`, `P_ell=−expm1(log Q_ell)`; use `−log(P_ell)` for `y=1` and `−log Q_ell` for `y=0`, with a small numerical clamp only where needed. The auxiliary objective is

`L_4b=L_4a+lambda_H*L_hier`, starting with fixed `lambda_H=0.1` and changing it only through a declared validation-only sensitivity run if necessary. Averaging over levels prevents a threefold loss-scale increase. Report all auxiliary and original loss components separately.

## 4(c): 4(b) plus one consistent positive-path objective

Keep **all** 4(b) architecture and level-specific `g_ell` heads. Train a fresh matched-initialization model with only the additional path loss; do not fine-tune the selected 4(b) checkpoint for the primary comparison.

For every actual hierarchy node `u` at level `ell`, evaluate `p_u^(ell)=sigmoid(g_ell(r_u^(ell)))`. A path `pi` begins at one Level-1 microgroup, follows its recorded Level-2 parent, and ends at its Level-3 branch root. No path may switch branch or select unrelated high-scoring nodes at different levels. With `L=3`, define its log geometric-mean evidence:

`S_pi=(1/3)Σ_(ell=1)^3 log(p_(u_ell)^(ell))`, `E_pi=exp(S_pi)∈(0,1)`.

Use `logsigmoid(g_ell(r_u))` to compute log probabilities stably. There is exactly one ancestry path per Level-1 microgroup. Accumulate ancestor log scores with parent-index gathers or top-down tree dynamic programming, giving **linear** work in the number of hierarchy nodes/leaves; do not construct all patch-patch pairs or independently enumerate combinatorial paths.

The ordinary maximum over `E_pi` has sparse gradients; an unnormalized smooth maximum of probabilities can exceed 1. Use a **bounded normalized smooth maximum** with prespecified `tau_path=0.1` and `M` Level-1 paths:

`S_soft=tau_path*[logsumexp_pi(S_pi/tau_path)−log M]`; `P_path=exp(S_soft)`.

Because every `S_pi≤0`, `0<P_path≤1`. As `tau_path→0`, `P_path→max_pi E_pi`. Apply stable binary cross-entropy to `P_path`: `L_path=−log(P_path)` for `y=1`, and `L_path=−log(1−P_path)` for `y=0` using a stable log-one-minus-exp computation. The normalized aggregation uses the **global number of paths `M`**, not a separate normalization at each parent, which would bias branches according to tree shape. A positive slide is encouraged to have at least one high-evidence *nested* path; negative slides should not have one. This is an auxiliary prediction from learned heads, not observed patch truth.

The sole added objective is `L_4c=L_4b+lambda_P*L_path`, starting with `lambda_P=0.1`. If sensitivity analysis is needed, use only fold validation data and report every tried value. Keep final bag and instance predictions unchanged.

## Comparison, compute, and outputs

| Variant | New fits | Main comparison | What changes |
| --- | ---: | --- | --- |
| Flat Top-K UNI reference, `K=4` | 15 | Matched reference | Same one-scorer flat DSMIL at UNI dimension |
| 4(a), feature-only branch assignment (`w_space=0`) | 15 | 4(a) spatial | Same spatial microgroups/tree; remove only spatial assignment term |
| 4(a), spatial hierarchy | 15 | Flat Top-K UNI | Partitioned spatial hierarchy replaces four flat all-patch attention bags |
| 4(b), intermediate MIL | 15 | 4(a) spatial | Add level heads and `L_hier` |
| 4(c), path consistency | 15 | 4(b) | Add node use of existing heads and `L_path` |

This initial comparison has **75 new fold/seed fits**. If pilot runtime exceeds Kaggle's available session budget, split the five variants into separate saved Kaggle runs/notebooks, using a checksummed prepared dataset and attaching prior outputs for comparison; do not silently reduce seeds, folds, or epochs for one variant. Profile small/median/large UNI bags, forward/backward peak GPU memory, grouping/tree rebuild time, and a 3–5 epoch pilot before the full runs. Cache only static spatial structures. Use minibatch size one WSI initially, mixed precision only after finite-loss and reproducibility checks, and vectorized/segmented pooling rather than Python loops over individual patches.

Save each variant's config, source hashes, fold/seed checkpoint, epoch history, training/validation loss and metrics, OOF probabilities, runtime/memory, and a `test_evaluated=false` completion marker. At every selected fold/seed checkpoint, record all 216 fit-training and 54 held-out-validation slides' anchor row IDs, group/branch/root membership counts, compression ratio, hierarchy parent IDs, empty/imbalanced branch diagnostics, attention summaries, and predictions. 4(b) additionally saves level-wise branch logits/probabilities and noisy-OR scores; 4(c) additionally saves node/path scores, strongest path's actual ancestry and coordinates, and path-loss components. Aggregate paired validation AUC/loss/accuracy and three-seed OOF predictions; keep mean fold AUC distinct from OOF ensemble AUC. Label-derived accuracy/F1/sensitivity/specificity require a threshold fixed from training-side or OOF predictions, never the official test set.

CAMELYON16 lesion masks are **not** included in prepared UNI embeddings. If localization is later evaluated, attach a separate verified annotation source, align it to patch coordinates, and treat it as held-out qualitative/quantitative analysis rather than supervision in 4(a)–4(c). Do not claim regional or path scores are clinical explanations from slide-label training alone.

## Implementation order and acceptance checks

1. Build a separate reference/experiment-4 runner and small Kaggle notebooks or stages; reuse existing UNI manifest/coordinate checks, Top-K DSMIL components, fold assignments, optimizer, metrics, and output writers. Do not modify the completed Phase 4 experiment 1/2 notebook or experiment 3 plan to implement this method. Keep notebook source below Kaggle's 1 MB limit; never embed checkpoints or feature bags.
2. Verify static grid grouping and anchor-conditioned splitting/assignment with synthetic coordinates. Assert each patch has exactly one group and branch; each non-root node has exactly one parent; each selected anchor has a distinct nonempty branch even when anchors share a starting cell; singleton and irregular slides preserve coverage; true coordinate and `w_space=0` assignments are reproducible and meaningfully distinct on a constructed case.
3. Check pooling shapes and gradients, final mean of four branch vectors, `0.5/0.5` two-stream loss, checkpoint save/reload, and parity with the flat Top-K UNI reference before adding auxiliary losses. Test representative large bags for memory and runtime; avoid `N×N` allocation.
4. Add 4(b) heads and verify noisy-OR limits, stable positive/negative losses, gradients to every level head, and branch-wide (rather than per-node) summaries. Add 4(c) with exactly the same heads; test that the reported strongest path is a real parent–child chain, `P_path∈[0,1]` for varied path counts, stable gradients, and a negative slide with one strong path.
5. Run the fixed validation protocol and write a complete comparison only after all five variants finish. Verify all 15 fold/seed fits per variant, identical folds/labels, 810 OOF rows per variant, no official-test feature reads, and no test predictions. Report parameter counts, runtime, hierarchy compression, and all planned diagnostics alongside performance. Do not infer benefit or mechanism before inspecting outputs.

**Naming:** In this plan, `4(a)/4(b)/4(c)` are the Phase 4 subruns. The source attachment's `B2/B3/B4` are historical labels for these same cumulative ideas, not the project's earlier A/B experiments. The source attachment's A1/A2/A3/A4 *ablation labels* should not be confused with the project's pre-Phase-4 A-series or its separately planned semantic A3 experiment.
