# CAMELYON16 Kaggle experiments

This directory contains the runnable notebooks and original runners. Phase 1
reproduces the existing DSMIL baseline implementation. The remaining model
variants are our implementations. All experiments use existing precomputed
patch embeddings; no feature encoder is trained or run here. The concise
[official test table](../research_results/README.md) gives the completed
results. Large downloaded Kaggle outputs are kept locally under
`obtained_results/` and excluded from new commits; curated JSON summaries
are in `../research_results/test_summaries/`.

**Current run status:** Phases 1–3, A/B validation, Experiment C, Phase 4
experiments 1/2, Phase 4 experiment 4 flat Top-K and feature assignment, and
Phase 5 multiscale fixed/adaptive runs have completed. Experiment 3 stopped
after its control and λ=0 variants; positive-λ variants were not run. The
remaining hierarchical experiment 4 additions were not run. Official test
evaluation was completed for the nine models tabulated in the result guide.

Import each notebook into Kaggle and attach the **same prepared uniscale dataset**. Enable a GPU. Run the phases in order:

1. `01_dsmil_baseline.ipynb`: original single-anchor DSMIL.
2. `02_multi_anchor.ipynb`: four highest-scoring anchor patches.
3. `03_multi_anchor_diversity.ipynb`: four anchors with the prespecified 0.1 attention-overlap penalty.

Each notebook contains the runner source. Use **Save Version → Save & Run All**; Quick Save does not run and preserve the full result. The first execution cell finds the prepared dataset by `PREPROCESSING_COMPLETE.json` under `/kaggle/input`, regardless of the Kaggle mount depth.

Each phase trains the same five stratified folds with training seeds 42, 43, and 44. The 270 official training slides are used for CV; each validation fold has 54 slides. The official 129-slide test set is not opened until all 15 CV models finish. The next notebook cell evaluates the test set and averages the probabilities from **all 15 seed-by-fold checkpoints**. This rule is fixed in advance for all three phases. The classification threshold is Youden's J on the three-seed averaged out-of-fold training predictions; it is not fitted using the test labels. `test_114` is normal (0) under the chosen protocol.

The notebook saves two folders under `/kaggle/working`:

- `dsmil-...-seeds42-43-44/` contains `seed_42/`, `seed_43/`, and `seed_44/`, each with five checkpoints, fold histories, fold summaries, and OOF scores. The root has `cv_summary.json`, `oof_predictions_all_seeds.csv`, and `oof_predictions_ensemble.csv`. The summary reports all 15 fold AUCs, each seed's five-fold mean, and descriptive mean/SD of those three means.
- `dsmil-...-seeds42-43-44-test/` contains all 1,935 test predictions from the checkpoints, the 129-slide mean-probability predictions, and `test_summary.json` with AUC, accuracy, balanced accuracy, sensitivity, specificity, and the confusion matrix.

The 15 fold scores share slides; their SD is descriptive, not a confidence interval. These reruns diagnose initialization sensitivity while keeping the fold assignment fixed. The local audit found consistent labels, unique slide coverage, balanced fold counts, and finite OOF probabilities in the previous three runs. The raw uniscale bags are not available locally, so Kaggle's prepared-dataset validation remains the source-file check.

## Completed runs

The downloaded three-seed CV and official test outputs for all three phases
are preserved in `obtained_results/three_seed_test/`. A five-page readable
summary is in `../output/pdf/camelyon16_uniscale_three_seed_results.pdf`.

## Experiment C: baseline seed sensitivity

Import `C_baseline_seed_sensitivity.ipynb` into Kaggle, attach the **prepared uniscale** dataset, enable a GPU, and choose **Save Version → Save & Run All**. This notebook embeds the unchanged Phase 1 baseline runner and uses the ten unique random seeds fixed in `experiment_c_seed_manifest.json` before Experiment C results were observed. It trains all ten seeds across the same five folds with the original single-anchor model, orthogonal initialization, learning rate, scheduler, and checkpoint rule. It does **not** evaluate the official test set or try alternative initialization and learning-rate settings.

The notebook writes `/kaggle/working/experiment-c-baseline-seed-sensitivity/` with `experiment_config.json`, the seed manifest, 50 fold checkpoints and histories, 10 seed summaries, `fold_results.csv`, `seed_results.csv`, `stability_diagnostics.csv`, all 2,700 OOF predictions, ten-seed averaged OOF probabilities, and `experiment_c_summary.json`. Review the complete performance and stability distribution before deciding which runs need controlled optimization diagnostics. The averaged OOF AUC is descriptive and does not replace fold-level reporting.

The completed Experiment C output is copied into `obtained_results/experiment_c_baseline_seed_sensitivity/`. Its `EXPERIMENT_C_RESULTS.md` and `experiment_c_per_fold_results.csv` give every fold's selected-checkpoint training and validation metrics. The 51-page curve report is at `../../output/pdf/experiment_c_baseline_training_curves.pdf`.

Experiment C is closed as a descriptive seed-sensitivity audit. Its results do not establish why weak runs occurred; changing initialization or training settings was not tested in C.

The final validation-only implementation plan for the combined A/B Kaggle run is `AB_SERIES_SINGLE_KAGGLE_RUN_PLAN.md`. It specifies the 90 new fold/seed fits, Phase 2/3 archive reuse, exact selector/fusion rules, checks, output files, and comparison method.

## Combined A/B Kaggle validation run

Import `AB_validation_single_run.ipynb` into Kaggle, attach the **same prepared CAMELYON16 uniscale dataset** used for Phases 1-3, enable a GPU, and select **Save Version -> Save & Run All** with output enabled. The notebook embeds `baseline_dsmil.py`, `ab_validation_runner.py`, and the checksummed `ab_phase2_phase3_reference.zip`. No additional code or archived-checkpoint input is needed. The run trains 90 new fold/seed models, reuses the Phase 2/3 validation tables for the two K=4 comparison cells, and never evaluates the official test bags.

The saved output appears under `/kaggle/working/ab-validation/`. Check `COMPLETE.json` (`new_runs=90`, `new_oof_rows=4860`, `test_evaluated=false`), `comparison_table.csv`, `oof_predictions_all_variants.csv`, `paired_fold_auc_differences.csv`, `B_interaction.csv`, and `A1_alpha_selection.json`. Each new variant has its own directory with 15 checkpoints, histories, fold summaries, OOF predictions, and anchor/fusion diagnostics. `build_ab_reference.py` and `build_ab_notebook.py` regenerate the embedded reference and notebook if the runner changes; rerun the focused tests in `tests/test_ab_validation_runner.py` before uploading a regenerated notebook.

The completed Kaggle output is copied into `obtained_results/ab_validation_single_run/`. The readable analysis, including A1 selection, A2 fusion, B interaction, seed variation, and diagnostic figures, is `obtained_results/AB_VALIDATION_RESULTS_REPORT.md`.

## A1 test report and Phase-4 validation

Import `A1_alpha025_official_test.ipynb` into Kaggle with **two inputs**: the prepared uniscale dataset and the completed **A/B validation notebook output**, which contains `ab-validation/A1_alpha_025/`. Alternatively, upload the local `a1_alpha025_test_reference.zip` as a Kaggle Dataset and attach that dataset; Kaggle expands the archive, and the notebook finds its extracted files. Choose **Save Version → Save & Run All** with output enabled. The notebook is under Kaggle's 1 MB source limit because it embeds only a small checksum manifest; it verifies the 15 archived checkpoints from the attached input. It evaluates only this existing model on the 129 official test slides and saves results under `/kaggle/working/a1-alpha025-test/`.

Import `04_a1_uni_spatial_validation.ipynb` separately with the **prepared UNI** dataset and a GPU. It runs Phase 4 experiment 1 (A1 α=0.25 on 1024-dimensional UNI features), experiment 2(a) (spatially spread anchors plus relative-distance attention over all patches), and experiment 2(b) (the same spatial model, but retain the 32 highest distance-biased attention logits per anchor **before softmax and pooling**). The nearby scale is fixed at eight UNI patch widths. Experiments 2(a) and 2(b) each tune gamma over `0, 0.25, 0.5, 1.0` using five folds × seeds 42/43/44; with experiment 1, the notebook runs 135 fold/seed fits. These are **validation-only** runs: they do not open UNI test bags or write test predictions. The models train new weights because the archived A1 checkpoints use 512-dimensional inputs. The full mathematical and experimental contract is in `PHASES_4_5_IMPLEMENTATION_PLAN.md`.

Check `/kaggle/working/phase4-a1-validation/COMPLETE.json` in the Phase-4 notebook's saved Output tab. Experiment 1 and each gamma candidate in 2(a)/2(b) have checkpoints, histories, OOF predictions, and validation summaries. Anchor diagnostics cover every training and validation slide at every selected checkpoint, including all six pairwise anchor feature dissimilarities and spatial distances. Every Phase-4 completion file records `test_evaluated=false`.

The proposed four-concept A3 follow-up is specified in `A3_SEMANTIC_MULTI_HEAD_PLAN.md` and is deferred until after Phase 4 experiment 2. UNI has row-aligned embeddings and coordinates but no patch-level concept labels; the audited multiscale features have neither coordinates nor concept labels. A3 therefore needs separate annotations and a matched UNI A1 comparison.

The separate **latent multi-head spatial-anchor** experiment is specified in `PHASE4_MULTIHEAD_SPATIAL_ANCHORS_IMPLEMENTATION_PLAN.md`. It replaces A1's single scorer and alpha selector with configurable independent heads, enforces eight-patch-width separation for selected anchors, and uses a trainable soft candidate-repulsion penalty.

The **Phase 4 experiment 4** plan is `PHASE4_HIERARCHICAL_DSMIL_IMPLEMENTATION_PLAN.md`. It defines three cumulative validation-only subruns: 4(a) spatial hierarchical pooling from flat Top-K UNI with four anchors, 4(b) intermediate existential MIL supervision, and 4(c) nested positive-path supervision. A new flat Top-K UNI reference and a feature-only branch-assignment control are included. The plan uses the same five folds, seeds 42/43/44, and Kaggle setup; it does not reuse the experiment-3 multi-head model.

Use **`phase4_experiment_3.ipynb`** and **`phase4_experiment_4.ipynb`** as separate Kaggle notebooks. The earlier combined `05_phase4_experiments_3_and_4.ipynb` is superseded for new runs. Attach the prepared UNI dataset and select **T4 ×2**. Each notebook trains at most two unfinished variants per **Save Version → Save & Run All** version, using one independent process per GPU. Attach the previous saved notebook output to the next version; the notebook verifies hashes, folds, and all 15 fold/seed checkpoints before carrying completed variants forward. The experiment-3 notebook can also reuse completed variants from the older combined notebook output, including the already run control and lambda=0 variants if that output was saved. Each family has five variants, so a run starting from scratch takes three saved notebook versions (2 + 2 + 1 variants). Both notebooks are validation-only and do not open UNI official-test feature bags. Experiment 4 includes a coordinate-occupancy check; fix `GRID_WIDTH` at 4 or 8 before training and keep it unchanged across versions. The model runner is unchanged to preserve compatibility with earlier completed results. Outputs are `/kaggle/working/phase4-exp3/` and `/kaggle/working/phase4-exp4/`. The completed subsets were run on Kaggle; no full local Kaggle training run was performed.

Experiment 3 was stopped after its first saved pair (`exp3_control` and `exp3_spatial_lam_000`); do not launch another Exp 3 version. The three positive-lambda variants remain unrun. See `obtained_results/phase-4 exp3/EXP3_STOP_DECISION.md` for the validation evidence and interpretation limits. Experiment 4 is unaffected.

## Phase 4 experiment 2(b): official test and spatial figures

Import `phase4_exp2b_gamma100_test_and_visualize.ipynb` into Kaggle. Attach the prepared UNI dataset and upload `phase4_exp2b_gamma100_test_reference.zip` as a separate Kaggle Dataset, then attach it. Enable a GPU and Internet, and select **Save Version → Save & Run All**. The ZIP contains the 15 frozen experiment 2(b), γ=1.0 checkpoints, its OOF predictions, and coordinate metadata. `build_phase4_exp2b_test_notebook.py` regenerates both files from the archived validation results.

The notebook verifies the checkpoint and prepared-dataset hashes, evaluates all 129 official test slides, averages 15 bag probabilities, and applies the training OOF Youden threshold. Results are under `/kaggle/working/phase4-exp2b-gamma100-test/`: `test_summary.json`, per-checkpoint and ensemble prediction CSVs, `visualization_examples.json`, and three coordinate figures in `figures/` when the test set includes a false-negative tumor. It then downloads the selected original CAMELYON16 TIFFs and available XML lesion annotations from the public AWS archive, checks slide dimensions, overlays anchor and top-eight attended patch positions with tumor/exclusion boundaries, deletes each TIFF, and saves `three_wsi_overlays.zip`. The ZIP includes `annotation_overlap_summary.json` and `selected_patch_annotation_overlap.csv`, which compare anchor and attended patch squares with annotated tumor using patch-center hits and a sampled tumor-area fraction. The annotations come from a single checkpoint chosen using validation metrics; each figure's prediction comes from the 15-checkpoint ensemble. Missing XML is recorded as unavailable; `test_114` is flagged as incompletely annotated. If there is no false-negative tumor, the notebook records that fact.

## A1 multiscale ablation and official test

Import **`06_a1_multiscale_ablation_and_test.ipynb`**. Attach only the prepared multiscale `cat` dataset. The already obtained A1 alpha=0.25 official-test results from `obtained_results/a1_alpha025_official_test/` are embedded with checksums, so this notebook does not need the uniscale bags or A1 checkpoints and does not repeat A1 test inference. Select Kaggle **T4 ×2** and **Save Version → Save & Run All**. The notebook trains fixed-concatenation and learned-gate multiscale A1 variants in separate GPU-bound subprocesses, 15 fold/seed fits each; if only one GPU is visible it runs them sequentially. It verifies both sets of checkpoints and OOF predictions, evaluates the two new variants on the official test set, and compares them with the archived A1 results. Outputs are under `/kaggle/working/multiscale-a1/`. The design and evaluation rules are in `MULTISCALE_A1_ABLATION_IMPLEMENTATION_PLAN.md`. The fixed and adaptive variants were run on Kaggle and their official test summaries were archived locally; no full local Kaggle training run was performed.

## Phase 4 experiment 4: frozen official-test evaluations

`flat_topk_uni_official_test.ipynb` requires the prepared UNI bags and the saved
`exp4_flat_topk_uni/` output with all 15 checkpoints. The notebook averages
their official-test probabilities and uses its training OOF threshold.

`feature_assignment_uni_official_test.ipynb` requires the prepared UNI bags
and coordinates plus the saved `exp4_4a_feature_assignment/` output. It embeds
audited test coordinate-scale metadata. The hash-verified training runner used
feature-only assignment weight **0.0**, although an archived `config.json`
incorrectly lists 0.5. The test notebook reproduces the actual 0.0 rule.

Both notebooks evaluate frozen models and save summary JSON and slide-level
predictions under `/kaggle/working/`. Their completed summary JSON files are
published in `../research_results/test_summaries/`; the full outputs remain in
the local `obtained_results/phase4-exp4/` archive.
