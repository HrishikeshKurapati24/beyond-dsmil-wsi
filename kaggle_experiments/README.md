# CAMELYON16 uniscale Kaggle experiments

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
