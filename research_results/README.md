# Official CAMELYON16 test results

Each row is a 15-checkpoint ensemble (five folds × three seeds) on the same 129 official test slides. Probabilities are averaged; the classification threshold is obtained from training out-of-fold predictions. These are descriptive results from one held-out set. Some later ideas were developed after earlier test outcomes had been viewed.

| Model | AUROC | Accuracy | Balanced accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing DSMIL baseline, uniscale | 0.9514 | 0.8992 | 0.8858 | 0.8333 | 0.9383 |
| Our Top-K (4), uniscale | 0.9581 | 0.9070 | 0.8877 | 0.8125 | 0.9630 |
| Our Top-K (4) + attention diversity, uniscale | 0.9496 | 0.8915 | 0.8881 | 0.8750 | 0.9012 |
| Our A1 weighted ranking (α=0.25), uniscale | 0.9552 | 0.8760 | 0.8503 | 0.7500 | 0.9506 |
| Our spatial Top-32 (γ=1.0), UNI | 0.9717 | 0.9380 | 0.9252 | 0.8750 | 0.9753 |
| Our flat Top-K, UNI | 0.9871 | 0.9690 | 0.9626 | 0.9375 | 0.9877 |
| Our feature assignment, UNI | 0.9866 | 0.9767 | 0.9730 | 0.9583 | 0.9877 |
| Our fixed multiscale A1 | 0.7006 | 0.7674 | 0.6960 | 0.4167 | 0.9753 |
| Our adaptive multiscale A1 | 0.7685 | 0.7984 | 0.7419 | 0.5208 | 0.9630 |

The feature-assignment UNI model has the highest observed accuracy (126/129), improving by 10 correctly classified slides over the existing baseline (116/129). The flat Top-K UNI model has the highest AUROC by a small margin (0.0005 over feature assignment). The two multiscale A1 variants did not improve on their uniscale counterpart.

The CSV contains the exact metrics and paths to the local archive. `test_summaries/` preserves the original JSON summaries, including threshold and provenance fields where available. Large checkpoint files, feature bags, and downloaded visualizations are kept outside the published result subset.

## Inputs and reproducibility

- Existing [uniscale](https://www.kaggle.com/datasets/hrishikeshkurapati/camelyon16-precomputedfeatures-uniscale), [multiscale](https://www.kaggle.com/datasets/hrishikeshkurapati/camelyon16-precomputedfeatures-multiscale), and [UNI](https://www.kaggle.com/datasets/hrishikeshkurapati/camelyon16-uni-uniscale-preprocessed-features) feature datasets are hosted on Kaggle.
- The [experiment guide](../kaggle_experiments/README.md) describes how to prepare inputs and run the Kaggle notebooks.
- `test_114` is treated as normal under the project label protocol.
- The [final paper](../docs/final_paper.pdf) contains the full methodological and qualitative analysis.
