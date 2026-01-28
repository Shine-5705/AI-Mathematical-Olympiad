"""Publication-quality plots with clear, colorful styling."""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#222222",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.8,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 11,
    "legend.framealpha": 0.95,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

PALETTE = [
    "#4C72B0",  # steel blue
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#CCB974",  # gold
    "#64B5CD",  # sky blue
    "#DD8452",  # orange
    "#DA8BC3",  # pink
]

GRADIENT_BLUE = ["#d0e1f9", "#4C72B0"]
GRADIENT_GREEN = ["#d4edda", "#55A868"]


def plot_training_loss(
    train_losses: List[Tuple[int, float]],
    eval_losses: List[Tuple[int, float]],
    output_path: str,
):
    """Training and eval loss with smoothing, annotations, and fill."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    if train_losses:
        steps, losses = zip(*train_losses)
        losses = np.array(losses)
        steps = np.array(steps)

        # Raw data (faint)
        ax.plot(steps, losses, color=PALETTE[0], alpha=0.15, linewidth=0.8)

        # Smoothed curve
        window = max(1, len(losses) // 15)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        s_steps = steps[window - 1:]
        ax.plot(s_steps, smoothed, color=PALETTE[0], linewidth=2.8, label="Train Loss (smoothed)")
        ax.fill_between(s_steps, smoothed, alpha=0.08, color=PALETTE[0])

        # Start and end annotations
        ax.annotate(
            f"{losses[0]:.3f}", xy=(steps[0], losses[0]),
            fontsize=9, fontweight="bold", color=PALETTE[0],
            xytext=(15, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=1.2),
        )
        ax.annotate(
            f"{smoothed[-1]:.3f}", xy=(s_steps[-1], smoothed[-1]),
            fontsize=9, fontweight="bold", color=PALETTE[0],
            xytext=(-40, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=1.2),
        )

    if eval_losses:
        steps, losses = zip(*eval_losses)
        ax.plot(steps, losses, color=PALETTE[2], linewidth=2.5,
                marker="D", markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                label="Eval Loss", zorder=5)

        # Best eval point
        best_idx = np.argmin(losses)
        ax.scatter(steps[best_idx], losses[best_idx], color=PALETTE[1], s=120,
                   edgecolors="white", linewidths=2, zorder=6)
        ax.annotate(
            f"Best: {losses[best_idx]:.3f}", xy=(steps[best_idx], losses[best_idx]),
            fontsize=9, fontweight="bold", color=PALETTE[1],
            xytext=(10, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=1.2),
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend(loc="upper right")
    ax.grid(True, axis="both", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path)
    plt.close()


def plot_experiment_comparison(
    results: Dict[str, Any],
    output_path: str,
):
    """Bar charts for accuracy, compute cost, and prune rate."""
    methods = list(results.keys())
    n = len(methods)
    accuracies = [results[m]["aggregate"].get("accuracy", 0) * 100 for m in methods]
    tokens = [results[m]["aggregate"].get("avg_tokens", 0) for m in methods]
    times = [results[m]["aggregate"].get("avg_time", 0) for m in methods]
    prune_rates = [results[m]["aggregate"].get("prune_rate", 0) * 100 for m in methods]

    has_prune = any(p > 0 for p in prune_rates)
    cols = 3 if has_prune else 2

    fig, axes = plt.subplots(1, cols, figsize=(6.5 * cols, 6))
    if cols == 1:
        axes = [axes]

    bar_width = 0.6
    x = np.arange(n)
    colors = PALETTE[:n]

    # --- Accuracy ---
    ax = axes[0]
    bars = ax.bar(x, accuracies, width=bar_width, color=colors,
                  edgecolor="white", linewidth=1.5, zorder=3)
    for bar, acc in zip(bars, accuracies):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 1.2,
                f"{acc:.1f}%", ha="center", fontsize=12, fontweight="bold", color="#333333")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, max(accuracies) * 1.25 if max(accuracies) > 0 else 100)
    ax.grid(True, axis="y", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Horizontal baseline
    if len(accuracies) > 1:
        ax.axhline(y=accuracies[0], color=colors[0], linestyle="--", alpha=0.4, linewidth=1, zorder=1)

    # --- Compute Cost (grouped: tokens + time) ---
    ax2 = axes[1]
    w = 0.3
    b1 = ax2.bar(x - w / 2, tokens, width=w, color=colors, edgecolor="white",
                 linewidth=1.5, zorder=3, label="Tokens")

    ax2_twin = ax2.twinx()
    b2 = ax2_twin.bar(x + w / 2, times, width=w, color=colors, edgecolor="white",
                      linewidth=1.5, zorder=3, alpha=0.45)

    for bar, tok in zip(b1, tokens):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tokens) * 0.02,
                 f"{tok:.0f}", ha="center", fontsize=10, fontweight="bold", color="#333333")
    for bar, t in zip(b2, times):
        ax2_twin.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                      f"{t:.1f}s", ha="center", fontsize=9, color="#666666")

    ax2.set_ylabel("Avg Tokens / Problem", color="#333333")
    ax2_twin.set_ylabel("Avg Time (s)", color="#888888")
    ax2.set_title("Compute Cost")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=25, ha="right", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.4, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2_twin.spines["top"].set_visible(False)

    tokens_patch = Patch(facecolor=PALETTE[0], edgecolor="white", label="Tokens")
    time_patch = Patch(facecolor=PALETTE[0], alpha=0.45, edgecolor="white", label="Time (s)")
    ax2.legend(handles=[tokens_patch, time_patch], loc="upper left", fontsize=9)

    # --- Prune Rate ---
    if has_prune:
        ax3 = axes[2]
        bars3 = ax3.bar(x, prune_rates, width=bar_width, color=colors,
                        edgecolor="white", linewidth=1.5, zorder=3)
        for bar, pr in zip(bars3, prune_rates):
            if pr > 0:
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                         f"{pr:.1f}%", ha="center", fontsize=12, fontweight="bold", color="#333333")
        ax3.set_ylabel("Branches Pruned (%)")
        ax3.set_title("Symbolic Verification Pruning")
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods, rotation=25, ha="right", fontsize=10)
        ax3.set_ylim(0, max(prune_rates) * 1.3 if max(prune_rates) > 0 else 100)
        ax3.grid(True, axis="y", alpha=0.4, zorder=0)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

    fig.suptitle("Experiment Results", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()


def plot_per_problem_heatmap(
    results: Dict[str, Any],
    output_path: str,
):
    """Heatmap: correct/wrong per problem per method with accuracy labels."""
    methods = list(results.keys())
    if not methods:
        return

    n_problems = len(results[methods[0]]["results"])
    if n_problems == 0:
        return

    matrix = np.zeros((len(methods), n_problems))
    for i, method in enumerate(methods):
        for j, result in enumerate(results[method]["results"]):
            matrix[i, j] = 1 if result.get("correct", False) else 0

    fig_width = max(10, n_problems * 0.45 + 3)
    fig_height = max(3, len(methods) * 1.0 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Custom colormap: soft red → soft green
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rg", ["#F4A6A0", "#A8D5BA"], N=2)

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11, fontweight="bold")
    ax.set_xticks(range(n_problems))
    ax.set_xticklabels(range(1, n_problems + 1), fontsize=9)
    ax.set_xlabel("Problem Number", fontsize=12)
    ax.set_title("Per-Problem Results", fontsize=15, fontweight="bold", pad=15)

    # Cell labels
    for i in range(len(methods)):
        for j in range(n_problems):
            symbol = "O" if matrix[i, j] == 1 else "X"
            color = "#2d6a4f" if matrix[i, j] == 1 else "#9b2226"
            ax.text(j, i, symbol, ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)

    # Accuracy on right side
    for i, method in enumerate(methods):
        acc = results[method]["aggregate"].get("accuracy", 0) * 100
        color = "#2d6a4f" if acc >= 50 else "#9b2226" if acc < 30 else "#b45309"
        ax.text(n_problems + 0.5, i, f"{acc:.0f}%", va="center", fontsize=12,
                fontweight="bold", color=color)

    ax.text(n_problems + 0.5, -0.8, "Acc", ha="center", fontsize=10,
            fontweight="bold", color="#666666")

    legend_elements = [
        Patch(facecolor="#A8D5BA", edgecolor="#cccccc", label="Correct"),
        Patch(facecolor="#F4A6A0", edgecolor="#cccccc", label="Wrong"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=2)

    ax.set_xlim(-0.5, n_problems + 1.5)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()


def plot_efficiency_scatter(
    results: Dict[str, Any],
    output_path: str,
):
    """Scatter: accuracy vs tokens with sized markers and connecting line."""
    fig, ax = plt.subplots(figsize=(9, 6.5))

    methods = list(results.keys())
    points = []

    for i, method in enumerate(methods):
        acc = results[method]["aggregate"].get("accuracy", 0) * 100
        tok = results[method]["aggregate"].get("avg_tokens", 0)
        points.append((tok, acc, method, i))

    # Sort by tokens for connecting line
    points.sort(key=lambda p: p[0])

    # Connecting line (efficiency frontier)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color="#cccccc", linewidth=1.5, linestyle="--", zorder=1)

    # Scatter points
    for tok, acc, method, i in points:
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(tok, acc, color=color, s=280, zorder=5,
                   edgecolors="white", linewidths=2.5)
        ax.annotate(
            method, (tok, acc),
            textcoords="offset points", xytext=(12, 10),
            fontsize=11, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.85),
        )

    # Quadrant labels
    mid_x = (min(xs) + max(xs)) / 2 if xs else 500
    mid_y = (min(ys) + max(ys)) / 2 if ys else 50
    ax.axvline(x=mid_x, color="#e0e0e0", linewidth=1, linestyle=":", zorder=0)
    ax.axhline(y=mid_y, color="#e0e0e0", linewidth=1, linestyle=":", zorder=0)

    pad = 5
    ax.text(min(xs) - pad, max(ys) + pad, "Efficient & Accurate",
            fontsize=8, color="#55A868", fontstyle="italic", alpha=0.7)
    ax.text(max(xs) + pad, min(ys) - pad, "Expensive & Inaccurate",
            fontsize=8, color="#C44E52", fontstyle="italic", alpha=0.7, ha="right")

    ax.set_xlabel("Avg Tokens / Problem", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Efficiency vs Accuracy", fontsize=15, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_margin = (max(xs) - min(xs)) * 0.15 if len(xs) > 1 else 100
    y_margin = (max(ys) - min(ys)) * 0.2 if len(ys) > 1 else 20
    ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
    ax.set_ylim(max(0, min(ys) - y_margin), min(100, max(ys) + y_margin))

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()


def generate_all_experiment_plots(results: Dict[str, Any], output_dir: str, name: str):
    """Generate all experiment plots."""
    out = Path(output_dir)
    plot_experiment_comparison(results, str(out / f"{name}_comparison.png"))
    plot_per_problem_heatmap(results, str(out / f"{name}_heatmap.png"))
    plot_efficiency_scatter(results, str(out / f"{name}_efficiency.png"))
