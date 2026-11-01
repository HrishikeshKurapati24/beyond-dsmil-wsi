# CAMELYON16 Kaggle preprocessing

Prepare **one** feature source per Kaggle notebook run. The script reads the
public raw feature dataset under `/kaggle/input` and writes compact NumPy bags,
one shared five fold assignment, and a slide manifest under `/kaggle/working`.
It does not alter raw Kaggle inputs or preprocess large files on the laptop.

## Run

1. Upload `preprocessing_bundle.zip` as a small Kaggle dataset. It contains
   `prepare.py` and `reference_slides.csv`. This bundle was generated from the
   three completed local audit reports.
2. Import `prepare_camelyon16.ipynb` into Kaggle. Attach the bundle dataset
   and **one** raw dataset. Set `SOURCE` in the first code cell to `uniscale`,
   `multiscale`, or `uni`. The multiscale Kaggle URL is an assumed future
   location; run it only after its upload is available.
3. Run all cells. A successful run ends with
   `PREPROCESSING_COMPLETE.json` and 399 files in `bags/`.
4. Use **Save Version → Save & Run All** and wait for that version to finish
   successfully. **Quick Save does not run the notebook in the saved version
   and should not be used to persist the prepared files.** Confirm the saved
   version's Output tab lists `PREPROCESSING_COMPLETE.json`, `folds.csv`,
   `slides.csv`, and the 399 `.npy` bags. Publish or create a
   versioned Kaggle Dataset from that output if later notebooks need it as
   `/kaggle/input`. Do this before the session expires. Record the output
   version used by each experiment.

The notebook locates the mounted raw dataset by its feature-folder layout,
without relying on Kaggle's mount name. It accepts the preprocessing bundle either as a ZIP or
as the `prepare.py` and `reference_slides.csv` files extracted by Kaggle.

Expected prepared feature data is approximately 10.01 GB for uniscale,
16.96 GB for multiscale, or 16.82 GB for UNI (decimal GB). Check available
Kaggle working space before each run. These are separate runs and outputs.

The prepared output is `/kaggle/working/camelyon16-SOURCE-prepared-v1/`:

```text
bags/<slide_id>.npy       float32 [patches, 512 or 1024]
coords/<slide_id>.npy     int64 [patches, 2]; UNI only
folds.csv                 canonical label, official split, validation fold
slides.csv                feature path and row count per slide
PREPROCESSING_COMPLETE.json
```

Only the 270 official training slides receive a validation fold (0–4). The
129 `test_*` slides receive a blank fold and remain held out. The seed is 42;
the assignments are generated from the same canonical reference for all
three feature sources. The script checks all 399 input slide IDs before
conversion, then verifies every bag's expected dimensions, row count, and
finite values. UNI also checks coordinate row alignment and bounds.

`test_114` is labeled **normal** in the shared reference, following the user's
explicit decision. The multiscale source agrees; the uniscale source places it
in the tumor folder. The script records the uniscale conflict and the user's
label decision in the completion JSON, and stops on any new disagreement.
Keep this provenance visible when reporting held-out results.
For the multiscale concatenation, columns 0–511 are
20× features and 512–1023 are 5× features, as established by the source
feature extraction code and local audit.

The run can be restarted in the same output directory. Complete `.npy` files
are reused; temporary files are replaced atomically. A missing completion
JSON means the output is incomplete and must not be used for training.

No remote SHA256 check is performed. The Kaggle listing checks and conversion
checks establish coverage and readable feature arrays, not byte identity
with the local audit copies.
