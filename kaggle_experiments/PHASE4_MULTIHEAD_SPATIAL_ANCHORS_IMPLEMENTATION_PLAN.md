# Phase 4 experiment 3: latent multi-head spatial anchors

**Status:** implemented in `phase4_exp3_4_runner.py` and the combined Kaggle notebook `05_phase4_experiments_3_and_4.ipynb`; the Kaggle run has not been executed. This is a latent multi-head experiment, not the separately planned A3 experiment with named tumor, boundary, necrosis, and stromal/immune concepts. UNI has coordinates but no patch-level concept labels, so head identities must remain unnamed.

## Question and comparison

Test whether independently learned instance scorers can choose spatially separated anchors that improve slide classification. The primary matched reference is **Phase 4 experiment 1**, the A1 selector retrained on 1024-dimensional UNI features. The original 512-dimensional uniscale A1 experiment was pre-Phase 4 and its weights are not reused. Phase 4 experiments 2(a)/2(b) are secondary comparisons because they change attention itself.

Keep UNI embeddings, coordinates, the official 270-slide training cohort and fixed five folds, seeds 42/43/44, shared 128-dimensional query network, identity value path, equal-mean bag-vector fusion, affine bag classifier, and the existing training schedule/checkpoint rule. Set configurable maximum head count `K_max`; the first comparison uses `K_max=4`. A slide can activate `m≤K_max` heads. Replace A1's one scorer and alpha-based selection entirely: **alpha, gamma, relative-distance attention bias, top-32 attention pooling, and attention-overlap loss are absent**. Attention still covers every patch for each selected anchor.

## Data and notation

Use prepared UNI `bags/<slide>.npy` and row-aligned `coords/<slide>.npy`. For slide patches `i=1,…,N`, let `x_i∈R^1024`, source level-0 coordinate `t_i∈R²`, and audited source patch width `P` pixels. Define `d(i,j)=||t_i−t_j||₂/P` in patch widths. Enforce minimum selected-anchor distance `ell=8`. Validate patch count, feature/coordinate row alignment, finite features, source patch width, and coordinate bounds. Phase 4 remains **validation-only**: do not open or score UNI official-test bags.

## Forward algorithm

1. **Independent scores.** Create `K_max` independently initialized linear scorers, `s_ik=w_kᵀx_i+b_k`. Each head ranks all `N` patches by its own score; ties use original patch-row ID. Do not assign biological names to heads.

2. **Hard constrained anchor search.** Each head may supply at most one anchor or be inactive. Search tuples `a_k∈P_k(M)∪{null}`, where `P_k(M)` contains that head's top `M` patches, subject to `d(a_k,a_h)≥8` for every active pair. Prefer assignments lexicographically by (i) larger number `m` of active heads, (ii) smaller sum of within-head selected ranks, and (iii) original row IDs/head IDs for exact ties. Using ranks avoids comparing raw logit scales across independently trained heads. Start at `M=32`, then try `64,128,256` if the best result has `m<K_max`.

   Implement a deterministic beam search with width 64 **for each partial-assignment cardinality**, so a temporarily skipped head is not automatically pruned. At each head, branch to eligible shortlist candidates and to `null`; keep the best 64 states at every active-count level by cumulative within-head rank, with deterministic ties. Return the best result after the largest attempted pool. The beam is approximate: `m<K_max` means the **selector found** fewer anchors within its declared search budget, not that a slide is geometrically incapable of supporting `K_max`. Save the attempted pool sizes and the best `m` at each expansion. If budget-limited cases are common, increase pool/beam size before interpreting the experiment. Never relax the 8-width rule, silently drop the slide, or claim that a bounded search proved infeasibility.

3. **One attention map per active head.** Use the existing shared query `q:R^1024→R^128`. For selected anchor `a_k`, calculate

   `e_ik=q(x_i)ᵀq(x_(a_k))/sqrt(128)`, `A_ik=exp(e_ik)/Σ_(j=1)^N exp(e_jk)`.

   The actual forward selection is one real patch per active head. Every attention column softmaxes over **all N slide patches**; the anchor-search shortlist is not an attention-support cap.

4. **Bag and prediction.** Form `z_k=Σ_i A_ik x_i`, `z_b=(1/m)Σ_(k active)z_k`, and bag logit `g_b=w_bᵀz_b+b_b`. If `m=3`, use three vectors and divisor three. Because the bag classifier is affine, applying it to the mean bag vector equals averaging its per-anchor logits.

## Loss and gradient path

Let `u_k=s_(a_k,k)` for active heads. Set instance smooth-max temperature `tau_inst=1` and define

`M_tau(u)=tau_inst * log[(1/m) Σ_(k active) exp(u_k/tau_inst)]`.

Use a slide-label-aware instance loss:

`L_inst = softplus(-M_tau(u))` for `y=1`; `L_inst=(1/m)Σ_(k active)softplus(u_k)` for `y=0`.

Thus a positive slide needs at least one strong selected instance score, while a negative slide pushes all active selected scores down. Do **not** give the positive slide label independently to all anchors: that would assert that each selected patch contains cancer. For `m=1`, this is the standard selected-logit binary loss.

**Trainable spatial repulsion.** A hinge on hard-selected distances would be zero because step 2 already enforces the gap, and hard argmax indices do not differentiate through coordinates. Instead, for each active head define a small support `C_k=Top32(s_·k)∪{a_k}` and a soft distribution `r_ik=softmax_(i∈C_k)(s_ik/tau_rep)` with `tau_rep=1`. For active pairs, use

`L_sp = [1 / choose(m,2)] Σ_(k<h) Σ_(i∈C_k)Σ_(j∈C_h) r_ik r_jh [max(0,1−d(i,j)/8)]²`.

Set `L_sp=0` when `m=1`. This is the expected squared proximity violation for two heads' candidate distributions. It is bounded and differentiable in the candidate logits; top-32 membership and hard head activation remain discrete. The hard rule, rather than `L_sp`, guarantees separation in the forward pass.

**Bag-to-scorer gradient.** To avoid leaving the independent scorers trained only by instance and repulsion losses, use a straight-through surrogate at each active anchor. Let `h_ik=1[i=a_k]`. Let `E_k` contain eligible candidates from `C_k`, including `a_k`, that are at least eight widths from the *other hard-selected anchors*. Define `p_ik=softmax_(i∈E_k)(s_ik/tau_sel)` with `tau_sel=1` and

`h_tilde_ik = h_ik + p_ik − stop_gradient(p_ik)`; `x_tilde_(a_k)=Σ_(i∈E_k)h_tilde_ik x_i`.

Use `q(x_tilde_(a_k))` in the attention computation during training. Its forward value is exactly `q(x_(a_k))`; its backward derivative sends bag-loss gradients to head `k` through `p`. The surrogate conditions on the selected assignment and does not differentiate through beam decisions, candidate membership, or which heads are active. If `E_k` has one patch, this bag-to-scorer gradient is zero for that slide, though the selected instance logit can still receive a gradient. Validation/inference uses the hard anchor directly.

The total objective is `L=0.5*BCEWithLogits(g_b,y)+0.5*L_inst+lambda*L_sp`. Select `lambda` on training/validation folds only from the prespecified grid `{0,0.1,0.25,0.5}`. No test-set result enters selection. `lambda=0` isolates what the hard constraint achieves without the soft repulsion loss.

## Controls and experimental contract

| Run | Heads | Hard 8-width rule | Spatial loss | Purpose | New fits |
| --- | ---: | --- | --- | --- | ---: |
| Existing UNI A1 reference | One scorer, up to four A1-selected anchors | No | No | Matched feature/fold comparison | 0, reuse existing 15 fits |
| Multi-head control | Four independent scorers | No | No | Effect of independent heads alone | 15 |
| Multi-head spatial | Four independent scorers | Yes | `lambda∈{0,0.1,0.25,0.5}` | Hard spacing and additional soft repulsion | 60 |

The no-gap control selects each head's raw top-1 patch, permits duplicate rows, uses the same instance loss and straight-through bag-gradient mechanism, and always has `m=K_max` for nonempty bags. Its `L_sp=0`. Comparing the spatial `lambda=0` run with this control isolates the hard spacing rule; comparing positive lambda with spatial `lambda=0` isolates the soft penalty. Comparing either multi-head run with UNI A1 additionally changes scorer capacity and instance loss, so attribute gains accordingly. The first complete study uses **75 new fold/seed fits**, with the same five folds, three seeds, 50-epoch maximum, patience 10, Adam/cosine settings, and maximum-validation-AUC checkpoint rule as the Phase 4 runner. Select lambda by highest mean validation AUC across the 15 fits, then lower mean validation loss, then smaller lambda. Reuse the UNI A1 validation result only if its data manifest, folds, and seeds match exactly.

## Implementation sequence

1. Add a **separate** runner (proposed `phase4_multihead_spatial_runner.py`) and Kaggle notebook (proposed `05_phase4_multihead_spatial_validation.ipynb`); leave the existing A1 test and Phase 4 experiment 1/2 notebooks untouched. Reuse the UNI manifest/coordinate validation helpers where possible. Keep notebook source below Kaggle's 1 MB source limit; embed code and only small coordinate metadata, never checkpoints or datasets.
2. Implement and test head logits, deterministic shortlist/beam search, variable `m`, pairwise physical-distance conversion, per-anchor full-slide attention, mean fusion, instance smooth max, spatial soft penalty, and straight-through surrogate independently. Include a no-gap mode using the same network.
3. Add validation-only training and checkpoint evaluation. Run all 270 official-training slides through each selected checkpoint for diagnostics (216 fit-training + 54 held-out validation slides per fold/seed). Preserve fold histories, OOF probabilities, checkpoints, and configuration/data hashes. Make stages resumable: verify completed fold/seed artifacts before skipping them, and allow a previous saved Kaggle notebook output to be attached as input and copied into `/kaggle/working` when continuing in a new session.
4. Run the no-gap control and all four spatial lambda candidates, then write one comparison table with paired fold/seed results, OOF ensemble results, chosen lambda, and explicit `test_evaluated=false`. No final-test predictions or official-test metrics belong in this notebook.

## Verification and diagnostics

Before full training, test small bags (`N<K_max`), identical-coordinate bags (`m=1` under the hard rule), spatially separated bags (`m=K_max`), a greedy-trap example where an alternative early choice permits more heads, candidate-pool expansion, deterministic tie/permutation handling, and no-gap duplicate selection. Assert every returned active pair has `d≥8−epsilon`; `epsilon` is only floating-point tolerance. Check that each active attention column sums to one over all patches; `z_b` divides by the actual `m`; positive and negative losses are finite for `m=1,…,K_max`; repulsion is zero beyond the gap and sends gradients to head logits when nearby probabilities are nonzero; bag loss reaches head weights through the straight-through path; inactive heads receive no contribution from that slide; checkpoint reload reproduces predictions; no UNI test bags are opened.

For **every** slide and selected checkpoint, save: active head IDs and `m`, selected patch row IDs/coordinates/logits, each chosen patch's within-head rank, shortlist/beam expansion records, whether the selector stopped at its search budget, minimum and all pairwise spatial distances, pairwise embedding cosine dissimilarity, per-head attention entropy/concentration and map overlap, bag-vector similarity, and prediction. Distinguish a selector-limited `m<K_max` from verified geometric infeasibility; do not infer the latter from beam failure. Report `m` and head-use distributions alongside validation AUC/loss/accuracy and paired OOF differences. A spatially separated selection or improved AUC alone does **not** establish different histologic concepts or cancer relevance; that requires patch-level annotations or reviewed overlays in the separate semantic A3 study.
