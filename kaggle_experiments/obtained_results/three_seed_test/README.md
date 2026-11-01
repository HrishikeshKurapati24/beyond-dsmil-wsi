# CAMELYON16 uniscale: three-seed CV and final test results

These are the completed Kaggle runs with training seeds 42, 43, and 44 on the same five folds. Each phase contains 15 CV checkpoints and an official 129-slide test evaluation that averages probabilities across all 15 models.

- `phase_1_baseline/` was copied from `~/Downloads/results-14/`.
- `phase_2_four_anchors/` was copied from `~/Downloads/results-13/`.
- `phase_3_diversity/` was copied from `~/Downloads/results/`.

Every source file except Finder `.DS_Store` was copied. `SHA256SUMS.txt` records the SHA-256 checksum of all 261 copied source files. The original download folders were not changed. The `cv_summary.json` field `test_evaluated: false` describes the CV stage only; each adjacent `*-test/test_summary.json` contains the completed official test evaluation.

The readable summary is at `../../../output/pdf/camelyon16_uniscale_three_seed_results.pdf` relative to this folder.
