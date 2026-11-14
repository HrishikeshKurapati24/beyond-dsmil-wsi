"""Audit the copied Experiment C run and plot all 50 fold histories.

Usage:
  python analyze_experiment_c.py RAW_RESULT_DIR OUTPUT_PDF

The raw files are read only. Per-fold tables are written beside RAW_RESULT_DIR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def load_and_check(root: Path):
    summary = json.loads((root / "experiment_c_summary.json").read_text())
    seeds = json.loads((root / "seed_manifest.json").read_text())["seeds"]
    fold = pd.read_csv(root / "fold_results.csv")
    stability = pd.read_csv(root / "stability_diagnostics.csv")
    assert len(seeds) == len(set(seeds)) == 10
    assert summary["completed_runs"] == 50 and summary["test_evaluated"] is False
    assert len(fold) == len(stability) == 50
    assert not fold.duplicated(["seed", "fold"]).any()
    assert set(fold.seed) == set(seeds)
    assert all(set(fold.loc[fold.seed == seed, "fold"]) == set(range(5)) for seed in seeds)
    assert fold.select_dtypes("number").notna().all().all()
    assert stability.select_dtypes("number").notna().all().all()

    histories = {}
    for seed in seeds:
        for fold_id in range(5):
            path = root / f"seed_{seed}" / f"fold_{fold_id}_history.csv"
            h = pd.read_csv(path)
            row = fold[(fold.seed == seed) & (fold.fold == fold_id)].iloc[0]
            state = stability[(stability.seed == seed) & (stability.fold == fold_id)].iloc[0]
            assert len(h) == int(state.epochs_ran)
            assert h.epoch.tolist() == list(range(1, len(h) + 1))
            assert int(row.best_epoch) in h.epoch.values
            assert h[["train_loss", "valid_loss", "valid_auc", "valid_accuracy_0.5", "lr"]].notna().all().all()
            selected = h.loc[h.epoch == int(row.best_epoch)].iloc[0]
            assert np.isclose(selected.valid_auc, row.validation_auc)
            assert np.isclose(selected.valid_loss, row.validation_loss)
            assert np.isclose(selected["valid_accuracy_0.5"], row["validation_accuracy_0.5"])
            assert (root / f"seed_{seed}" / f"fold_{fold_id}.pt").is_file()
            histories[(seed, fold_id)] = h
    return summary, seeds, fold, stability, histories


def write_tables(root: Path, summary: dict, seeds: list[int], fold: pd.DataFrame, stability: pd.DataFrame):
    table = fold.merge(stability[["seed", "fold", "epochs_ran"]], on=["seed", "fold"], validate="one_to_one")
    table["seed_order"] = table.seed.map({seed: i for i, seed in enumerate(seeds)})
    table = table.sort_values(["seed_order", "fold"]).drop(columns="seed_order")
    columns = ["seed", "fold", "epochs_ran", "best_epoch",
               "training_loss_at_best_checkpoint", "validation_loss",
               "training_accuracy_0.5", "validation_accuracy_0.5",
               "validation_balanced_accuracy_0.5", "validation_auc",
               "validation_sensitivity_0.5", "validation_specificity_0.5"]
    table[columns].to_csv(root / "experiment_c_per_fold_results.csv", index=False)

    lines = [
        "# Experiment C: baseline DSMIL seed sensitivity",
        "",
        "Ten presampled random seeds, five fixed CAMELYON16 training folds per seed; 50 runs total. The official test set was not evaluated in this experiment.",
        "",
        f"Mean of the 50 fold validation AUCs: **{summary['fold_validation_auc_mean']:.3f}** (descriptive SD {summary['fold_validation_auc_sd']:.3f}).",
        f"Ten-seed mean out-of-fold AUC: **{summary['mean_10_seed_oof_auc']:.3f}**. This is a development-set result, not a test score.",
        "",
        "## Reading the table and curves",
        "",
        "Loss and accuracy in the table are recomputed at the selected validation-AUC checkpoint. Training loss in the curves is the mean loss observed during each training epoch; validation loss is measured after that epoch. Each curve page marks the selected checkpoint epoch. Validation AUC and validation accuracy were logged every epoch. Training accuracy was logged only at the selected checkpoint, so no training accuracy curve can be reconstructed from these outputs.",
        "",
        "AUC values across folds and seeds are descriptive because the folds share much of their training data. A low validation AUC alone does not establish a local minimum, optimization pathology, or a causal seed effect.",
        "",
        "## Per-fold results",
        "",
        "| Seed | Fold | Epochs run | Best epoch | Train loss | Val loss | Train acc | Val acc | Val AUC |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {int(row.seed)} | {int(row.fold)} | {int(row.epochs_ran)} | {int(row.best_epoch)} | "
            f"{row.training_loss_at_best_checkpoint:.3f} | {row.validation_loss:.3f} | "
            f"{row['training_accuracy_0.5']:.3f} | {row['validation_accuracy_0.5']:.3f} | {row.validation_auc:.3f} |"
        )
    lines += [
        "",
        "## Initial diagnostic observations",
        "",
    ]
    weak = table.loc[(table.seed == 558403303) & (table.fold == 4)].iloc[0]
    peers = table.loc[(table.fold == 4) & (table.seed != 558403303), "validation_auc"].mean()
    fold2 = table.loc[table.fold == 2, "validation_auc"].mean()
    lines += [
        f"- Seed 558403303, fold 4 is a clear low-performing run: selected validation AUC {weak.validation_auc:.3f}, train loss {weak.training_loss_at_best_checkpoint:.3f}, validation loss {weak.validation_loss:.3f}, train accuracy {weak['training_accuracy_0.5']:.3f}, and validation accuracy {weak['validation_accuracy_0.5']:.3f}. The other nine fold-4 models average {peers:.3f} AUC. Its epoch curves warrant closer inspection.",
        f"- Fold 2 is systematically harder across these ten seeds: mean validation AUC {fold2:.3f}. This pattern should be separated from an isolated weak seed/fold run when planning follow-up experiments.",
        "- Diagnose any weak runs using the saved histories and matched fold/seed reruns. Fix a review criterion before testing alternative initialization or learning rate; these results alone do not identify the cause.",
        "",
    ]
    (root / "EXPERIMENT_C_RESULTS.md").write_text("\n".join(lines))
    return table


def summary_page(pdf: PdfPages, summary: dict, seeds: list[int], fold: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [0.9, 1.35]})
    fig.patch.set_facecolor("white")
    fig.suptitle("Experiment C: baseline DSMIL seed sensitivity", fontsize=17, fontweight="bold", y=0.96)
    means = fold.groupby("seed").validation_auc.mean().reindex(seeds)
    axes[0].barh(np.arange(10), means.values, color="#3b6c93")
    axes[0].set_yticks(np.arange(10), [str(seed) for seed in seeds], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.7, 1.0)
    axes[0].set_xlabel("Mean validation AUC across five folds")
    axes[0].set_title("Seed means")
    axes[0].grid(axis="x", alpha=0.2)
    matrix = fold.pivot(index="seed", columns="fold", values="validation_auc").reindex(seeds).values
    image = axes[1].imshow(matrix, cmap="YlGnBu", vmin=0.75, vmax=1.0, aspect="auto")
    axes[1].set_xticks(range(5), [f"Fold {fold_id}" for fold_id in range(5)])
    axes[1].set_yticks(range(10), [str(seed) for seed in seeds], fontsize=8)
    axes[1].set_title("Validation AUC by seed and fold")
    for i in range(10):
        for j in range(5):
            value = matrix[i, j]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axes[1].text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7,
                         color="white" if luminance < 0.52 else "#12202d")
    fig.colorbar(image, ax=axes[1], fraction=0.036, pad=0.03)
    fig.text(0.06, 0.08,
             f"50 fold runs | mean validation AUC {summary['fold_validation_auc_mean']:.3f} | "
             f"ten-seed mean OOF AUC {summary['mean_10_seed_oof_auc']:.3f} | no official test evaluation",
             fontsize=10)
    fig.text(0.06, 0.045, "Each following page shows one fold's epoch history and selected checkpoint.", fontsize=9,
             color="#4b5563")
    fig.subplots_adjust(left=0.14, right=0.95, top=0.87, bottom=0.16, wspace=0.42)
    pdf.savefig(fig)
    plt.close(fig)


def fold_page(pdf: PdfPages, seed: int, fold_id: int, row: pd.Series, h: pd.DataFrame, page: int):
    fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Experiment C - seed {seed}, fold {fold_id}", fontsize=17, fontweight="bold", y=0.95)
    best = int(row.best_epoch)
    epochs = h.epoch.to_numpy()
    axes[0].plot(epochs, h.train_loss, color="#27669a", linewidth=1.9, label="Training loss")
    axes[0].plot(epochs, h.valid_loss, color="#bd5b39", linewidth=1.9, label="Validation loss")
    axes[0].set_title("Classification loss")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(epochs, h.valid_auc, color="#3c8a74", linewidth=1.9)
    axes[1].set_title("Validation AUC")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0, 1.02)
    axes[2].plot(epochs, h["valid_accuracy_0.5"], color="#6b5b95", linewidth=1.9)
    axes[2].set_title("Validation accuracy")
    axes[2].set_ylabel("Accuracy at 0.5")
    axes[2].set_ylim(0, 1.02)
    for ax in axes:
        ax.axvline(best, color="#333333", linestyle="--", linewidth=1.2, alpha=0.75)
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, max(2, int(h.epoch.iloc[-1])))
        ax.grid(alpha=0.2)
    fig.text(0.065, 0.315, "Selected checkpoint", fontsize=12, fontweight="bold")
    labels = [
        ("Epochs run", f"{len(h)}"), ("Best epoch", f"{best}"),
        ("Train loss at checkpoint", f"{row.training_loss_at_best_checkpoint:.3f}"),
        ("Validation loss", f"{row.validation_loss:.3f}"),
        ("Train accuracy at 0.5", f"{row['training_accuracy_0.5']:.3f}"),
        ("Validation accuracy at 0.5", f"{row['validation_accuracy_0.5']:.3f}"),
        ("Validation AUC", f"{row.validation_auc:.3f}"),
    ]
    for index, (label, value) in enumerate(labels):
        col = index % 4
        line = index // 4
        x = 0.065 + col * 0.235
        y = 0.255 - line * 0.095
        fig.text(x, y, label, fontsize=9, color="#4b5563")
        fig.text(x, y - 0.03, value, fontsize=12, fontweight="bold")
    fig.text(0.065, 0.065,
             "Training accuracy was recorded only at the selected checkpoint; no epoch-level training accuracy curve is available.",
             fontsize=8.8, color="#4b5563")
    fig.text(0.93, 0.035, f"{page}/51", ha="right", fontsize=8, color="#6b7280")
    fig.subplots_adjust(left=0.065, right=0.965, top=0.82, bottom=0.4, wspace=0.28)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_result_dir", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()
    root = args.raw_result_dir
    summary, seeds, fold, stability, histories = load_and_check(root)
    table = write_tables(root, summary, seeds, fold, stability)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output_pdf) as pdf:
        summary_page(pdf, summary, seeds, fold)
        for i, (seed, fold_id) in enumerate(((seed, fold_id) for seed in seeds for fold_id in range(5)), 2):
            row = table[(table.seed == seed) & (table.fold == fold_id)].iloc[0]
            fold_page(pdf, seed, fold_id, row, histories[(seed, fold_id)], i)
    print(f"Wrote {root / 'experiment_c_per_fold_results.csv'}")
    print(f"Wrote {root / 'EXPERIMENT_C_RESULTS.md'}")
    print(f"Wrote {args.output_pdf}")


if __name__ == "__main__":
    main()
