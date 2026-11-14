# DSMIL research project: core code

This folder contains the main experiment runners and dataset preprocessing scripts. File names follow the project's **original phase progression**. Each Python file begins with its designated author and a scope note. These are annotated copies; the runnable originals remain in this repository's `kaggle_experiments/` and `kaggle_preprocessing/` directories.

## Project and data

The project extends DSMIL for slide-level tumor classification on CAMELYON16. It uses **existing precomputed patch embeddings**; the ResNet-18 feature encoder was neither trained nor run in these experiments. The official dataset split used here is 270 training slides and 129 test slides. Training and validation use five fixed folds, usually with seeds 42, 43, and 44. `test_114` is treated as normal under this project's label protocol. Experiment C audits ten additional recorded random seeds.

**Project direction:** Hrishikesh Kurapati oversaw the entire experimentation flow and proposed the research ideas and follow-up experiments. He directed the progression from baseline reproduction through anchor-selection, stability, UNI, spatial, hierarchical, and multiscale studies, including the comparisons and evaluation protocol. The file-level authorship below identifies implementation responsibility; it does not replace this project-level contribution.

Final test evaluation averages bag probabilities from 15 seed-by-fold checkpoints. The classification threshold comes from training out-of-fold predictions. Official test slides were not used for fitting or threshold selection. Some earlier test results had been viewed before later follow-up experiments were planned, so comparisons among those later experiments are descriptive.

## Phase progression and designated authors

| Phase and code file | Designated author | Purpose |
| --- | --- | --- |
| Preprocessing: `kaggle_preprocessing/dataset_preprocessing.py` | Hrishikesh Kurapati | Prepare uniscale, multiscale, and UNI feature bags, labels, coordinates, and folds on Kaggle. |
| Preprocessing: `kaggle_preprocessing/multiscale_dataset_overlay.py` | Srimanreddy | Combine a partial multiscale input with supplemental slide files. |
| Phases 1–3: `kaggle_experiments/phase1_to_3_dsmil_runner.py` | Srimanreddy | Baseline DSMIL, four-anchor Top-K, and the attention diversity penalty. |
| Before Phase 4: `kaggle_experiments/pre_phase4_ab_runner.py` | Hrishikesh Kurapati | A1 weighted ranking, A2 learned fusion, and K×λ ablations. |
| Before Phase 4: `kaggle_experiments/pre_phase4_experiment_c_runner.py` | Hrishikesh Kurapati | Ten-seed baseline stability audit across five folds. |
| Phase 4: `kaggle_experiments/phase4_uni_spatial_runner.py` | Hrishikesh Kurapati | UNI feature experiment and spatial experiments 2(a) and 2(b). |
| Phase 4: `kaggle_experiments/phase4_exp3_exp4_runner.py` | Hrishikesh Kurapati | Multi-projection and hierarchical anchor experiments. |
| Phase 4 test: `kaggle_experiments/phase4_exp2b_test_runner.py` | Srimanreddy | Evaluate frozen experiment 2(b), γ=1.0, on the official test set and select representative slides. |
| Phase 4 test: `kaggle_experiments/phase4_exp4_flat_topk_uni_test_runner.py` | Srimanreddy and Hrishikesh Kurapati | Evaluate the 15 frozen flat Top-K UNI checkpoints on the official test set; Srimanreddy's DSMIL baseline and Hrishikesh's Phase 4 experiment define the model. |
| Phase 4 test: `kaggle_experiments/phase4_exp4_feature_assignment_uni_test_runner.py` | Srimanreddy and Hrishikesh Kurapati | Evaluate the 15 frozen feature-assignment UNI checkpoints using audited test coordinates; builds on the same baseline and Phase 4 hierarchy. |
| Phase 5: `kaggle_experiments/phase5_multiscale_a1_runner.py` | Hrishikesh Kurapati | Fixed and adaptive multiscale A1 ablations. |

The project's progress presentation credits Srimanreddy with the baseline reproduction and Hrishikesh with the overall experimental direction and subsequent model studies. The file designations above are the assignments requested for this code package. The DSMIL model design builds on the original work of Bin Li and collaborators. The source repository's Git history is separate from these project responsibility labels.

## Using the code

Preprocessing is performed on Kaggle, with raw uploaded feature datasets mounted under `/kaggle/input` and prepared outputs saved under `/kaggle/working`. The preparation script accepts `--source {uniscale,multiscale,uni}`, `--input-root`, `--reference`, and `--output-root`. Prepared datasets include `PREPROCESSING_COMPLETE.json` and `slides.csv`; UNI also has per-patch coordinate arrays. Full datasets, notebook wrappers, reference CSVs, archived checkpoints, and result files remain in the original project or on Kaggle and are not duplicated in this narrow code folder.

The experiment runners import shared code from `kaggle_experiments/`. Use `python <script> --help` for each runner's stages and required paths. The unchanged Kaggle notebooks in `../kaggle_experiments/` provide the original execution order and input discovery. The main dependencies are PyTorch, NumPy, and scikit-learn; the Phase 4 test runner also uses Matplotlib.

The two Exp4 test runners are used by the corresponding standalone Kaggle notebooks. The feature-assignment test notebook supplies audited test-coordinate metadata. The visual overlay utility that downloads the three original whole-slide TIFFs is outside this folder at `../kaggle_experiments/phase4_wsi_overlays.py`. It runs after final test prediction and saves compressed image overlays.

## Reproducibility

Author comments and phase names were added **only to the copies in this folder**. Some archived runs validate exact source-file SHA-256 hashes. Use the unchanged runners and notebooks in this repository when reproducing those archived checkpoints. The renamed copies here are for consolidated review and project handoff.
