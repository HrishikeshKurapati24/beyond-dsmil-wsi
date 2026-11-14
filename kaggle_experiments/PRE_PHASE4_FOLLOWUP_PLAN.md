# Follow-up experiments before plan-based Phase 4

**Status update (2026-11-13):** Experiment C is closed as a descriptive seed-sensitivity audit; the proposed initialization and learning-rate follow-ups below were not performed and are not currently scheduled. The combined A/B validation run is complete, with results in `obtained_results/AB_VALIDATION_RESULTS_REPORT.md`. This plan records the original rationale and design; its future-tense C and A/B steps are historical.

The finalized implementation contract for the combined A/B Kaggle validation run is in `AB_SERIES_SINGLE_KAGGLE_RUN_PLAN.md`. Its exact selector, no-penalty A-series settings, archive reuse, run count, and output gates supersede the shorter A/B outline below.

## Objective and fixed comparison

Begin with **(C) seed sensitivity of the single-anchor Phase 1 DSMIL baseline**, as now requested. After reviewing its full results, plan any one-factor initialization and learning-rate diagnostics for weak runs. The previously proposed **(A) distinct anchor evidence or fusion** and **(B) anchor count** remain later follow-ups. These use the existing prepared CAMELYON16 **uniscale 512-dimensional features**. They are exploratory extensions of the completed Phase 1-3 study, not replacements for it. The plan-based Phase 4 will still take the original Phase 3 architecture (four top-scoring anchors, equal mean fusion, attention cosine penalty weight 0.1) onto UNI features, as required by `research_brief_kaggle.md` and `PHASES_4_5_IMPLEMENTATION_PLAN.md`.

For every directly compared configuration, retain the same 270 training slides, labels (`test_114` normal), saved five folds, optimizer/schedule, checkpoint rule, and bag-path probability. Experiment C uses ten unique randomly drawn seeds recorded before any C results; A and B still use seeds 42/43/44 unless revised later. Save the individual fold/seed results and 270-slide out-of-fold (OOF) scores; the primary development comparisons are paired by slide, fold, and seed. Keep each variant runnable by an explicit configuration, with its own output directory and configuration hash.

The CAMELYON16 test partition has already been inspected in Phases 1-3. Choose follow-up designs from training/OOF evidence and freeze them before any new test evaluation. New test numbers are descriptive; a fresh generalization claim needs an independent cohort. Do not discard seed 42 or select only favorable seeds after seeing results.

## A. Test whether four anchors can contribute distinct evidence

The archived Phase 2/3 diagnostics show very high selected-feature cosine similarity and attention overlap. Two **separate ablations** address different parts of that problem. Begin from the original **Phase 2 four-anchor model without the attention-diversity loss** for each; change only one mechanism at a time. Neither A1 nor A2 uses `L_div`.

### A1. Diversity-aware anchor selection

Keep the highest-scoring patch as anchor 1. For anchors 2-4, select greedily from the top 32 instance-score candidates using a fixed combination of (i) instance-score rank and (ii) dissimilarity to anchors already chosen, measured by cosine of the existing 512-dimensional features. Convert both terms to within-candidate ranks before combining them so their scales are comparable. Fix deterministic tie-breaking by original row index. The selection hyperparameter `alpha=0` must recover ordinary top-four selection when scores are distinct; document tie behavior separately. Evaluate a prespecified grid `alpha in {0, 0.25, 0.5, 0.75}` and choose any preferred value using training/validation (OOF) evidence only, before any new test evaluation. Keep the Phase 2 query network and equal fusion; set the attention-diversity weight to zero.

This is a deterministic selection rule, not a differentiable feature penalty. With fixed precomputed embeddings and hard `topk(...).indices`, simply adding cosine loss on the selected raw features would not provide a useful gradient to change the instance scores. Check the selector on bags with fewer than 32 patches and on tied scores; do not select the same row twice. Record the chosen indices, score ranks/gaps, selected-feature cosine, attention cosine, and per-slide probabilities. A lower cosine is a manipulation check, not proof that the patches contain different histology.

### A2. Learned fusion of anchor bag vectors

Return to ordinary top-four selection. Replace only the equal mean of the four attention-weighted bag vectors `b_k` with `sum_k w_k b_k`, where `w_k = softmax(g(b_k))` and `g` is a small shared linear scorer. Initialize the scorer's weights to zero so fusion starts at uniform weights; a shared bias can be omitted because it cancels inside the softmax. Keep the Phase 2 instance classifier, query/value paths, and bag classifier unchanged; set the attention-diversity weight to zero. This tests whether allowing unequal, slide-dependent contributions is useful even if anchor selection remains redundant.

Log fusion weights, weight entropy, per-anchor bag-vector cosine, attention cosine, and per-slide predictions. Confirm that zero-initialized fusion matches the original mean before the first update. Run A1 and A2 separately; only consider their combination as a later ablation if each shows a clear OOF reason to do so. Do not infer that unequal weights imply spatially or biologically distinct evidence.

**A gate:** all 15 runs per new candidate complete with finite losses and 270 x 3 OOF predictions; the selected-anchor and fusion diagnostics show whether the intended mechanism actually changed; report each configuration's five-fold results by seed and OOF ensemble against the no-penalty Phase 2 control. If neither mechanism improves the predeclared validation criteria, retain that negative result and continue to B.

## B. Compare two and four anchors, with and without attention diversity

Use the original top-K selector and equal fusion for a clean `K x penalty` factorial comparison: `K in {2,4}` and attention cosine weight in `{0,0.1}`. These are **four comparison configurations, but only two new training configurations**: train `(K=2, weight=0)` and `(K=2, weight=0.1)`; reuse the completed Phase 2 `(K=4, weight=0)` and Phase 3 `(K=4, weight=0.1)` runs after checking exact source, prepared manifest, fold assignment, seeds, optimizer, and checkpoint settings. Do not rerun either K=4 configuration when that audit passes. Keep the same three seeds and five folds. The diversity loss already returns zero for K=1 and works for K=2, so it needs no new formula.

Compare the four cells on paired fold/seed AUC, training and validation loss, OOF ensemble AUC, and threshold-dependent OOF metrics. Also compare pairwise attention and selected-feature similarity, anchor-score gaps, and positive/negative slide subsets. A lower K may reduce redundant selection simply because fewer patches are chosen; it does not establish that the penalty works better. Report both the K effect and the K-by-penalty interaction descriptively; do not claim one from a single favorable seed.

**B gate:** complete K=2 runs; exact reuse check for archived K=4 cells; full 2 x 2 comparison and diagnostics saved. Fix any candidate or penalty choice using OOF development data only before a held-out test run.

## C. Diagnose and measure seed sensitivity

Run the **unchanged Phase 1 baseline DSMIL** (one critical instance, no diversity penalty) across the saved five folds for each of ten unique randomly selected training seeds. Draw the seeds once before training, record their values and selection method, and retain every result. Keep the archived Phase 1 optimizer, orthogonal initialization, scheduler, checkpoint rule, and 50-epoch/10-epoch patience settings. Save all 50 fold histories, checkpoints, seed-level and fold-level summaries, and OOF probabilities. The folds remain correlated partitions of the same slides. This stage only measures performance and training stability; it does not evaluate the official test set or change the optimization recipe.

After the ten-seed run, examine fold histories, losses, AUC, accuracy, and checkpoint epochs. Define a weak-run criterion from that complete descriptive audit before launching follow-up diagnostics; disclose that the criterion was chosen after looking at these results. For the identified weak seed/fold combinations, plan one-factor-at-a-time comparisons on the **same seeds and folds**: initialization first, learning rate second, each against the unchanged baseline control. Keep all other settings fixed and record early loss/gradient norms and checkpoint behavior. Those diagnostic experiments are not included in the first C notebook. A low final AUC or high loss does not by itself identify a local minimum or loss-surface conditioning problem.

**C first-stage gate:** all ten seeds and five folds complete with finite histories, 50 checkpoints, and 270 OOF predictions per seed. Review the full distribution before choosing weak runs or optimization diagnostics. Any later claim that optimization is unstable requires paired follow-up evidence; include all weak runs in the report.

## Implementation order and outputs

1. Run C first using a separate orchestrator that imports the archived Phase 1 `baseline_dsmil.py` unchanged. The C Kaggle notebook embeds that exact source and the presampled seed manifest. Do not add optimization variants or official-test evaluation to this first stage.
2. Run A1, A2, and the two new B configurations in one Kaggle notebook execution, with separate output directories and configuration records. Reuse the archived Phase 2/3 K=4 results for B after the exact-reuse audit; attach those artifacts for in-notebook comparison or compare them with the new outputs after download. Keep the earlier Phase 1-3 notebooks functional. Any C initialization/learning-rate diagnostics remain a separate later decision based on the completed C audit.
3. Save per-run configuration, fold histories, checkpoints, selected-anchor indices, relevant attention/fusion diagnostics, OOF probabilities, and a machine-readable comparison summary under `/kaggle/working`. Use Kaggle **Save Version -> Save & Run All** and verify the version's Output tab. Avoid writing duplicate prepared bags to output.
4. Once A-C are complete, write a concise audit: which mechanisms changed their intended diagnostic, which changed predictive performance, and which conclusions remain uncertain. Only then start the original Phase 4 runner on UNI. Keep Phase 4's model definition tied to the original Phase 3 configuration even if an A-C variant looks promising.

The archived Phase 3 run took about 0.9 hours total for 15 fold/seed runs; use that only as an initial Kaggle runtime reference. Experiment C has 50 fold runs, so monitor Kaggle runtime and saved outputs. Benchmark one representative and one largest bag for later selectors/fusion, then estimate each full grid before launch.
