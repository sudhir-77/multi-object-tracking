"""Figures written next to metrics.json."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_loss(losses: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(losses) + 1), losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title("Association model training loss")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_metric_bars(sort_metrics: dict[str, float], learned_metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["MOTA", "IDF1", "HOTA"]
    keys = ["mota", "idf1", "hota"]
    sort_vals = [100.0 * sort_metrics[k] for k in keys]
    learned_vals = [100.0 * learned_metrics[k] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    x = range(len(labels))
    width = 0.35
    axes[0].bar([i - width / 2 for i in x], sort_vals, width, label="SORT")
    axes[0].bar([i + width / 2 for i in x], learned_vals, width, label="Learned")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Held-out tracking scores")
    axes[0].legend()
    axes[1].bar(
        ["SORT", "Learned"],
        [sort_metrics["id_switches"], learned_metrics["id_switches"]],
        color=["#4c72b0", "#dd8452"],
    )
    axes[1].set_ylabel("Identity switches")
    axes[1].set_title("Held-out ID switches")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_failure_modes(counts: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["occlusion", "crossing", "short_gap", "other"]
    values = [int(counts.get(k, 0)) for k in labels]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, values, color="#55a868")
    ax.set_ylabel("Identity switches")
    ax.set_title("Remaining identity switches by situation")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
