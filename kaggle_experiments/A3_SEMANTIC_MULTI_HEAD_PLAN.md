# A3 implementation plan: four concept-specific DSMIL instance heads

**Sequence update (2026-11-13):** Defer semantic A3 until Phase 4 experiment 1 (UNI features) and experiments 2(a)/2(b) (relative spatial attention, with and without top-32 pooling) are complete. A3 is the next UNI-based follow-up. Compare A3 only with A1 and other controls **run on the same UNI patch set and folds**. Do not compare its raw AUC directly with the earlier 512-dimensional uniscale A1 run as though only the anchor method changed.

**Verified source inventory:** the [UNI source repository](https://huggingface.co/datasets/kaczmarj/camelyon16-uni) supplies 1024-dimensional embeddings and matching, row-ordered patch coordinates; the project audit found a coordinate row for every embedding row. It supplies no patch-level histology or concept labels. The audited multiscale source contains feature CSVs only (804 files across `cat` and `fuse`); it has neither the patch coordinates nor the concept annotations needed for this validation. Therefore UNI is the usable feature source for A3 localization. Coordinates permit overlays but do not themselves identify tumor, boundary, necrosis, or stromal/immune tissue. Original WSIs and annotation sources must be attached separately on Kaggle for qualitative or mask-based checks.

## Research question

A1 showed that changing which anchors are chosen can change attention and modestly improve average checkpoint metrics, but it still selects four anchors using **one cancer-score question** and feature dissimilarity. A3 tests the user's intended alternative: learn **four different instance-score questions**, select one anchor from each, and learn how their evidence should contribute to a slide decision. The four intended concepts are (1) tumor-rich patch, (2) tumor-boundary patch, (3) necrosis, and (4) tumor-associated stromal or immune-rich patch. The fourth must mean a cancer-associated microenvironment, not generic lymphocytes that can also appear in normal lymph nodes.

The aim is distinct *diagnostic evidence*, not merely low embedding cosine. A high slide AUC or four different selected rows cannot, by itself, establish that the heads learned the named histology.

## Data gate: how to make the names real

The old prepared CAMELYON16 **uniscale** bags contain 512-dimensional feature rows and slide labels but **no patch coordinates, patch images, or concept labels**. The selected UNI source has coordinates and 1024-dimensional embeddings, but still has no concept labels. Four scorers trained only on the same slide-level tumor/normal label are exchangeable: they may all learn to select tumor patches, regardless of the names assigned to them. Therefore semantic A3 requires an annotation and row-alignment step before training.

1. On Kaggle, use the UNI dataset's row-aligned embedding/coordinate pairs and verify the prepared `coords_path` and `bag_path` row counts and checksums. Do **not** join UNI patch coordinates to old uniscale or multiscale feature rows by slide ID or row number: these sources have different patch sets. Record patch size, WSI level, and slide dimensions needed for overlays. Attach original WSIs separately only where needed for sampled patch review; avoid copying the full images into notebook output.
2. Use the [CAMELYON16 training-slide metastasis contours and masks](https://camelyon16.grand-challenge.org/Data/) to derive candidate tumor-interior and tumor-boundary annotations after verifying patch alignment and defining boundary width/overlap rules in physical units. These masks do **not** identify necrosis or stromal/immune tissue.
3. Obtain reviewed patch-level labels for necrosis and tumor-associated stromal/immune regions from training slides. Define whether a mixed patch can carry multiple labels; use masked multi-label targets if it can. Record `slide_id`, exact feature-row ID, patch coordinates, concept labels, annotation source, and uncertainty. Treat unannotated patches as **unknown**, not as negative examples for every concept. Confirm that each concept has sufficient positive and negative examples across the five training folds before freezing the experiment.
4. Keep annotation preparation on Kaggle. In each CV fold, use only training-slide concept annotations to fit or tune A3. Held-out-fold annotations are for evaluation only. Do not inspect official-test annotations or labels during development.

If this gate cannot be met, a four-head **latent** pilot can still test whether multiple score functions improve slide prediction, but its heads must be called head 1–4; it cannot support claims about necrosis, boundaries, or immune/stromal evidence.

## Architecture and objective

For UNI patch embeddings `x_i ∈ R^1024`, replace A1's one instance-score layer with four small, independent score heads `s_ih = f_h(x_i)`, `h=1,…,4`. Start with one linear layer per head so the capacity change is transparent. Each head's anchor is its highest-scoring patch, `j_h=argmax_i s_ih` (deterministic row-index tie-break). Do not apply A1's score–cosine rank selector or an attention-similarity penalty in the primary A3 run; the head identity, rather than feature distance, determines which question selects each anchor.

Keep A1's shared query projection `q`, identity value path `v(x_i)=x_i`, **equal-mean bag-vector aggregation**, and linear bag classifier architecture (adapted to UNI's 1024-dimensional input). For each head, compute

`a_ih = softmax_i(q(x_i)ᵀq(x_{j_h})/√128)`, and `b_h = Σ_i a_ih x_i`.

Thus each of the four anchors still attends over **all patches**. Keep the A1 bag aggregation unchanged: `b=(1/4)Σ_h b_h`, followed by a 1024-to-1 linear bag classifier `z_b=W_b b+c_b`. Reuse A1's architecture and optimization settings adapted to 1024-dimensional UNI input, **not its fitted weights**. A1's selected `α=0.25` has no role after its selector is replaced.

Fuse the four instance classifiers in the **instance stream** with a low-capacity, slide-dependent gate `π=softmax(G(m_1,…,m_4))`, where `m_h=max_i s_ih`. Initialize `G` to zero so all four weights begin at 1/4. Form the slide logit `z_inst=c+Σ_h π_h m_h`, with a learned bias `c`. The concept loss below helps keep the four score scales interpretable. Keep the original two-stream classification objective, `0.5 BCE(z_b,y)+0.5 BCE(z_inst,y)`. Add a **masked, class-balanced concept loss** on annotated training patches to teach each `f_h` its named question. A tumor-positive slide need not contain necrosis or a visible boundary, so do not apply the slide's positive label independently to all four heads. Fix the concept-loss weight from training-fold-only scale checks or a declared validation grid, and report all tested values. Check that a head can indicate concept absence rather than always presenting its forced argmax as a true finding.

This design changes anchor selection and **instance-score fusion** relative to A1, while holding its bag aggregation and classifier form fixed. The controls below separate the contribution of concept supervision and learned instance fusion. A shared `q` is retained initially so A3 does not also introduce four unrelated attention networks.

## Controls and execution

Use the same 270 training slides, labels, five folds, three seeds (42–44), optimizer, cosine schedule, maximum 50 epochs, early stopping, checkpoint rule, and bag-stream probability as the A/B run, adapted to UNI embeddings. Never drop a weak seed or fold. Keep official-test bags out of this validation run.

| Configuration | Concept supervision | Instance-score fusion | New fits |
| --- | --- | --- | ---: |
| UNI A1 α=0.25, four anchors (matched rerun) | None; one score head | Original max-instance score | 15 |
| A3 latent four-head control | None; four unnamed heads | Learned gate | 15 |
| A3 semantic, uniform-fusion ablation | Four named heads | Equal mean | 15 |
| A3 semantic, primary | Four named heads | Learned gate | 15 |

Run the three A3 configurations only after the data gate passes (**45 A3 fits**), plus a **15-fit matched UNI A1 reference** unless an exactly matching UNI A1 result already exists. The latent control tests whether more heads/capacity and learned instance fusion help without concept supervision. The uniform semantic ablation tests whether the learned gate adds value after semantic supervision. The UNI A1 reference tests the full proposed change on the same features and folds. Because semantic A3 also receives extra patch labels, report that resource difference explicitly; a performance gain is not attributable to architecture alone.

Before the full Kaggle run, check ordinary, smallest, and largest bags; one anchor per head; `N×4` attention with each column summing to one; exactly uniform initial gate; finite loss and gradients; masked concept loss; and checkpoint save/reload. Freeze the annotation manifest and configuration hashes. Save each fold's checkpoint, history, OOF scores, anchor row IDs, head scores, gate weights, per-slide attention-rank/top-patch summaries, and concept predictions.

## Analysis and decision rule

- **Slide prediction:** compare all 15 matched fold/seed validation AUCs with **UNI A1 α=0.25**; report the paired differences, training/validation loss and accuracy, and three-seed OOF ensemble AUC and fixed-threshold metrics. Keep mean fold AUC distinct from ensemble OOF AUC. This is a development-stage comparison, not a new official-test result.
- **Concept validity:** on held-out annotated patches, report per-head precision–recall and AUROC where defined, selected-anchor concept precision, and a blinded pathology review of top-ranked patches. Verify that boundary and stromal/immune heads identify their defined cancer-associated contexts. If concept validity fails, describe the model as latent/mixed evidence even if slide AUC rises.
- **Noncollapse and mechanism:** compare duplicate anchor selections, head score correlation, top-attended-patch overlap and rank correlation, attention cosine, selected-feature cosine, bag-vector cosine, gate-weight entropy, and how gate weights change when concepts are absent. Save enough per-patch information to verify attention ordering; A1's summary-only attention cosine could not answer that question.
- **Success criterion:** a credible slide-level improvement over A1, accompanied by independently supported concept specialization and no persistent four-head or gate collapse. Improvement only in the training-fold mean, without corroboration in OOF predictions or concept evidence, is exploratory.

The previous Phase 1–3 official test results have already been viewed. A later comparison of A3 on that same test set is exploratory; a fresh independent cohort would support a stronger generalization claim. Phase 4's UNI A1 and relative-spatial experiments should remain separate, unchanged comparisons unless the research plan is explicitly revised.

## Literature context

[DSMIL](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.pdf) supplies the two-stream architecture. [CLAM](https://pmc.ncbi.nlm.nih.gov/articles/PMC8711640/) shows that multiple attention branches and instance constraints can be trained with slide labels, but branch names do not establish histologic meaning. A [ranking-aware histopathology MIL study](https://medinform.jmir.org/2026/1/e84417) uses expert patch annotations to give an attention/ranking signal a defined pathology target. These sources motivate A3's separate concept-validation gate; none validates the four-head design in advance.
