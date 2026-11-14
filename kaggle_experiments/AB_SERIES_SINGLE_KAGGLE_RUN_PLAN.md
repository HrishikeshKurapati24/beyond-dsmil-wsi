# Final implementation plan: pre-Phase-4 A and B experiments

## 1. Scope and fixed experimental setup

Run A1, A2, and the two new B configurations in **one self-contained Kaggle notebook version** on the prepared CAMELYON16 **uniscale 512-dimensional patch features**. These are validation-stage experiments. The notebook must not load test bags, make test predictions, calculate test metrics, or choose a candidate using prior test results. The CAMELYON16 test results from Phases 1-3 have already been viewed, so later comparisons on that test set would be exploratory.

Keep the existing 270 official training slides, fixed five stratified folds (54 validation and 216 training slides per fold), canonical labels including `test_114 = normal`, and seeds **42, 43, and 44** for every new configuration. Retain the Phase 2/3 orthogonal weight initialization, Adam optimizer (`lr=1e-4`, `betas=(0.5,0.9)`, `weight_decay=0.001`), cosine schedule (`eta_min=5e-6`), maximum 50 epochs, patience 10, and checkpoint selection by highest validation AUC with validation loss breaking exact AUC ties. Keep the two-stream classification objective, `0.5 * BCE(bag logit, y) + 0.5 * BCE(max instance logit, y)`, and use the bag-stream sigmoid probability for evaluation. Each configuration has 15 fold/seed fits and 270 OOF predictions per seed. Never drop an unfavorable seed or fold.

Phase 2, the no-penalty four-anchor model, is the A-series control. Phase 3, the four-anchor model with attention penalty weight 0.1, is the other B-series reference. No ResNet-18 or other feature encoder is trained here.

## 2. Exact model configurations

| ID | Anchor count | Anchor selector | Fusion of anchor bag vectors | Attention penalty weight | Training in this run |
| --- | ---: | --- | --- | ---: | --- |
| A1-alpha0 / B-K4-L0 / Phase 2 | 4 | Original top-four instance scores | Equal mean | 0 | Reuse archive |
| A1-alpha025 | 4 | Rank-based diversity-aware selector, `alpha=0.25` | Equal mean | 0 | 15 fits |
| A1-alpha050 | 4 | Rank-based diversity-aware selector, `alpha=0.50` | Equal mean | 0 | 15 fits |
| A1-alpha075 | 4 | Rank-based diversity-aware selector, `alpha=0.75` | Equal mean | 0 | 15 fits |
| A2-learned-fusion | 4 | Original top-four instance scores | Learned softmax fusion | 0 | 15 fits |
| B-K2-L0 | 2 | Original top-two instance scores | Equal mean | 0 | 15 fits |
| B-K2-L01 | 2 | Original top-two instance scores | Equal mean | 0.1 | 15 fits |
| B-K4-L01 / Phase 3 | 4 | Original top-four instance scores | Equal mean | 0.1 | Reuse archive |

This is **90 new fold/seed fits**: six new configurations times 15. There are four B comparison cells, but only the two `K=2` cells train. The archived `K=4` cells are never retrained. A1 and A2 have **no** `L_div`. Do not combine A1 selection with A2 fusion, and do not add learned fusion to B.

### A1: diversity-aware selection, with no diversity loss

For each bag, let `c_i` be the existing scalar instance logit for patch feature `x_i`. Make a candidate pool of the top `M=min(32,N)` patches by `c_i`. Select the highest-scoring patch first. For each next anchor, calculate its dissimilarity to the already selected set `S`:

`d(i,S) = 1 - max_{j in S} cosine(x_i, x_j)`.

Within the remaining candidates, rank patches by `c_i` descending (`r_score=1` is best) and by `d(i,S)` descending (`r_div=1` is most dissimilar). Select the candidate with the smallest

`(1-alpha) * r_score(i) + alpha * r_div(i)`.

Break combined-rank ties by higher instance logit, then by original bag-row index. Preserve that original row index when the training loop shuffles patches. Normalize feature vectors with a numerical epsilon before cosine calculation. Do not select a row twice. Define `K_eff=min(4,N)` and use `M=min(32,N)` so small bags remain valid. When `alpha=0`, dispatch directly to the original Phase 2 top-four operation; this guarantees exact baseline behavior even if score ties occur and permits reuse of the Phase 2 archive. For `alpha>0`, the rule is a deterministic, non-differentiable selector; the ordinary classification loss still trains the scorer and attention network, but no gradient is claimed through the discrete selection decision. Record selected indices, selected scores/ranks, selected-feature pairwise cosine, attention pairwise cosine, and probabilities. Lower feature cosine alone does not establish cancer relevance.

### A2: learned fusion, with no diversity loss

Keep the original top-four selector and attention maps. The existing attention path makes four bag vectors `b_k = sum_i a_ik x_i`. Replace the Phase 2 equal mean with

`w_k = softmax_k(u^T b_k)`, `b = sum_k w_k b_k`,

where `u` is one shared 512-dimensional trainable vector. Omit a shared bias because it cancels in the softmax. Initialize `u=0` **after** the runner's general orthogonal initialization, so the four weights start at exactly `1/4`; check that the initial bag logit equals Phase 2's after copying identical values into every shared model parameter. Train `u` with the same classification objective and optimizer as the rest of the model. Record each weight, weight entropy, bag-vector pairwise cosine, attention pairwise cosine, and probabilities. Unequal weights do not prove that anchors represent different histology.

### B: anchor count by attention-penalty factorial

Keep the original top-K selector and equal fusion `b=(1/K) sum_k b_k`. For `K=2`, train once with `lambda=0` and once with `lambda=0.1`. The nonzero-weight training objective is

`L_train = L_classification + lambda * mean_{k != l} cosine(a_:k, a_:l)`.

`lambda` changes training only; inference always uses the bag-stream probability. For `K=2`, the mean off-diagonal cosine is the cosine of the two attention columns. Compare against the archived `K=4, lambda=0` Phase 2 and `K=4, lambda=0.1` Phase 3 results. Record the same attention and selected-feature diagnostics as for the archived cells.

## 3. Archive-reuse contract

Before preparing the notebook, freeze a compact, checksummed reference bundle containing the archived Phase 2 and Phase 3 `config.json`, 15 fold summaries per setting, 810 OOF rows per setting, and the diagnostics needed for comparisons. Include this bundle inside the standalone notebook or as an explicitly attached small input, so the single Kaggle run can produce its full comparison without relying on local files. **Do not include or retrain the old checkpoints.**

The existing archives have already been checked for source SHA-256 `32dc2a316278566e4c41dbe4fc01ca04e003c726beffcc8b7311f59f13cd2385`, prepared manifest SHA-256 `12d13547a03ce93dacf2087fe237f16f9a69d1ac2611c6a6b88dadab945ca604`, seeds 42/43/44, five folds, 50 epochs, patience 10, `lr=1e-4`, and weight decay 0.001. The new runner must verify those identifiers against the attached prepared dataset and check that the reference OOF slide IDs, labels, and fold IDs match before comparing. If a mismatch is found, stop with an explicit error; never silently substitute a different comparison cohort.

## 4. Implementation and one-run Kaggle flow

1. Add a **new A/B runner** and a self-contained Kaggle notebook; leave the Phase 1-3 runner and old notebooks unchanged. Implement selector and fusion as explicit configuration options with defaults that reproduce Phase 2. Keep each new configuration's artifacts in its own subdirectory under `/kaggle/working/ab-validation/`. Store exact code, package versions, prepared-manifest hash, source/reference hashes, configuration, seed, and fold with the results.
2. The notebook discovers exactly one prepared uniscale input by `PREPROCESSING_COMPLETE.json`, checks source/dimension/slide count/labels and the five-fold assignment, and verifies the frozen Phase 2/3 reference bundle. The prepared data remain mounted under `/kaggle/input`; do not copy bags into `/kaggle/working`.
3. Run a brief preflight before the full grid: ordinary bag and largest observed bag forward/backward; `N<32`, `N<4`, equal-score, and zero-norm selector cases; no duplicate selections; `alpha=0` forward/loss/gradient equivalence to Phase 2; A2 zero-init uniform weights and initial-logit equivalence; finite `K=2` penalty and gradients; checkpoint save/reload. Preflight reads **training** bags only.
4. Execute the six new configurations sequentially in the same **Save Version -> Save & Run All** run. Within each configuration, train all five folds for each of seeds 42/43/44. Save histories and selected checkpoints incrementally, with deterministic output names and no overwriting of mismatched configurations. Emit `COMPLETE.json` only after all 90 new fits, all OOF files, and all comparisons pass verification.
5. Produce per-fold/per-seed and per-slide comparison tables, then a concise validation-only summary. The notebook's last cell asserts the expected checkpoints, 270 x 3 OOF predictions for each new configuration, finite losses/probabilities, and no test prediction or test summary files. Save the notebook version with output enabled, then inspect its Output tab.

The completed Phase 2 and Phase 3 archives took about **0.70 and 0.91 hours per 15 fits**, respectively; 90 new fits suggest roughly **4-5 hours before A1 selector overhead and Kaggle hardware variation**. Time the first representative and largest-bag preflight operations and the first completed fold to update the projection. Kaggle's current notebook documentation gives a **12-hour CPU/GPU Save & Run All execution limit** and **20 GB of saved `/kaggle/working` space**; keep a margin for summary generation and saving. These numbers are planning checks, not runtime guarantees.

## 5. Validation analysis and decision rules

- **A1 alpha tuning:** prespecify `alpha in {0,0.25,0.5,0.75}`. The primary selection metric is the mean of the 15 saved fold/seed validation AUC values; `alpha=0` is the archived Phase 2 control. Choose the alpha with the highest primary metric, breaking exact ties toward smaller alpha. Report its paired AUC difference against `alpha=0` in all 15 matched fold/seed cells and the three-seed averaged OOF AUC. Report every alpha even if none improves on zero. Selection is development-stage hyperparameter tuning, not evidence of a new test-set gain.
- **A2:** compare the learned-fusion model with the Phase 2 control on the same 15 fold/seed AUCs and the OOF slide predictions. Report whether weights actually depart from 1/4 and whether that relates to performance. Do not select fusion based on test labels.
- **B:** report all four cells. At each `K`, calculate the paired difference from adding the penalty. Report the descriptive interaction: `[metric(2,0.1)-metric(2,0)] - [metric(4,0.1)-metric(4,0)]`. Compare `K=2` versus `K=4` at each penalty weight. Include loss/accuracy/AUC distributions, OOF results, positive/negative slide subsets, selected-feature cosine, and attention cosine. The 15 fold scores are correlated because training sets overlap; their SD is descriptive, not an independent-sample confidence interval.
- **Threshold metrics:** show fixed-threshold 0.5 OOF results and, if useful, a separately labeled OOF-chosen threshold result. Do not describe a threshold fitted and evaluated on the same OOF predictions as unbiased performance. Do not compute official-test metrics in this notebook.

## 6. Completion gates and next step

The single Kaggle version is complete when the six new configurations have all **90 checkpoints**, 90 histories, 90 fold summaries, **4,860 new OOF rows** (six configurations x 810), diagnostics, a verified two-cell archive reference, an A1 alpha comparison, an A2 comparison, a four-cell B comparison, and `COMPLETE.json`. All outputs must be under `/kaggle/working/ab-validation/` or its versioned code files. Review the complete validation distribution before proposing a later official-test evaluation. The plan-based Phase 4 still uses the original Phase 3 architecture on UNI features, regardless of these exploratory results.
