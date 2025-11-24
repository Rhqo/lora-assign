"""
Visualization utilities for gradient norm analysis.
"""
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(".")
from src.callbacks.gradient_norm import GradientNormLog


def plot_gradient_norms(
    logs: Union[GradientNormLog, Dict[str, GradientNormLog]],
    title: str = "Gradient Norm Analysis",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    smoothing_window: int = 5,
) -> plt.Figure:
    """
    Plot gradient norms over training steps.

    Args:
        logs: Single GradientNormLog or dict of {experiment_name: log}
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        smoothing_window: Window size for moving average smoothing

    Returns:
        Matplotlib figure
    """
    # Convert single log to dict format
    if isinstance(logs, GradientNormLog):
        logs = {"experiment": logs}

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Distinct colors for each module type
    module_colors = {
        "embed": "#4CAF50",      # Green
        "attention": "#2196F3",  # Blue
        "mlp": "#FF9800",        # Orange
        "lm_head": "#9C27B0",    # Purple
        "total": "#E53935",      # Red
    }

    # Plot 1: All module types comparison (LISA Fig.2 style)
    ax1 = axes[0]
    for exp_name, log in logs.items():
        steps = log.steps

        # Plot embed if available
        if "embed" in log.group_norms and any(v > 0 for v in log.group_norms["embed"]):
            embed_norms = _smooth(log.group_norms["embed"], smoothing_window)
            ax1.plot(steps, embed_norms, label="Embed", color=module_colors["embed"],
                     linestyle="-", marker="s", markersize=3, markevery=max(1, len(steps)//10))

        attn_norms = _smooth(log.group_norms["attention"], smoothing_window)
        mlp_norms = _smooth(log.group_norms["mlp"], smoothing_window)
        ax1.plot(steps, attn_norms, label="Attention", color=module_colors["attention"], linestyle="-")
        ax1.plot(steps, mlp_norms, label="MLP", color=module_colors["mlp"], linestyle="-")

        # Plot lm_head if available
        if "lm_head" in log.group_norms and any(v > 0 for v in log.group_norms["lm_head"]):
            lm_head_norms = _smooth(log.group_norms["lm_head"], smoothing_window)
            ax1.plot(steps, lm_head_norms, label="LM Head", color=module_colors["lm_head"],
                     linestyle="-", marker="^", markersize=3, markevery=max(1, len(steps)//10))

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Gradient Norm (log scale)")
    ax1.set_yscale("log")
    ax1.set_title("Module-wise Gradient Norms")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Total gradient norm
    ax2 = axes[1]
    for exp_name, log in logs.items():
        steps = log.steps
        total_norms = _smooth(log.group_norms["total"], smoothing_window)
        mean_norm = np.mean(log.group_norms["total"])

        ax2.plot(steps, total_norms, label=f"Total (mean: {mean_norm:.1f})", color=module_colors["total"])

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Gradient Norm (log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Total Gradient Norm")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_layer_heatmap(
    log: GradientNormLog,
    module_type: str = "both",
    title: str = "Layer-wise Gradient Norm Heatmap",
    save_path: Optional[str] = None,
    figsize: tuple = (16, 8),
    aggregation: str = "mean",
) -> plt.Figure:
    """
    Plot heatmap of gradient norms across layers and time.
    Includes embed and lm_head if available.

    Args:
        log: GradientNormLog with layer data
        module_type: "attention", "mlp", or "both"
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        aggregation: How to aggregate steps ("mean", "max", "last")

    Returns:
        Matplotlib figure
    """
    if not log.layer_norms:
        print("No layer-level data available")
        return None

    # Check if embed and lm_head data are available
    has_embed = "embed" in log.group_norms and any(v > 0 for v in log.group_norms["embed"])
    has_lm_head = "lm_head" in log.group_norms and any(v > 0 for v in log.group_norms["lm_head"])

    # Prepare data
    layers = sorted(log.layer_norms.keys(), key=lambda x: int(x.split("_")[1]))
    num_layers = len(layers)
    num_steps = len(log.steps)

    # Calculate extra rows for embed and lm_head
    extra_rows = (1 if has_embed else 0) + (1 if has_lm_head else 0)

    if module_type == "both":
        fig, axes = plt.subplots(1, 3 if extra_rows > 0 else 2, figsize=figsize)

        # Plot 1: All components combined (LISA style)
        if extra_rows > 0:
            ax_all = axes[0]
            total_rows = num_layers + extra_rows
            data_all = np.zeros((total_rows, num_steps))
            labels_all = []

            row_idx = 0
            if has_embed:
                data_all[row_idx, :] = log.group_norms["embed"]
                labels_all.append("Embed")
                row_idx += 1

            for layer in layers:
                # Combined attention + mlp for each layer
                attn_data = np.array(log.layer_norms[layer]["attention"])
                mlp_data = np.array(log.layer_norms[layer]["mlp"])
                data_all[row_idx, :] = attn_data + mlp_data
                labels_all.append(layer.replace("layer_", "L"))
                row_idx += 1

            if has_lm_head:
                data_all[row_idx, :] = log.group_norms["lm_head"]
                labels_all.append("LM Head")

            sns.heatmap(
                data_all,
                ax=ax_all,
                cmap="YlOrRd",
                xticklabels=50,
                yticklabels=labels_all,
            )
            ax_all.set_xlabel("Training Steps")
            ax_all.set_ylabel("Component")
            ax_all.set_title("All Components (Combined)")

            # Attention and MLP heatmaps
            for idx, mod_type in enumerate(["attention", "mlp"]):
                ax = axes[idx + 1]
                data = np.zeros((num_layers, num_steps))

                for i, layer in enumerate(layers):
                    data[i, :] = log.layer_norms[layer][mod_type]

                sns.heatmap(
                    data,
                    ax=ax,
                    cmap="YlOrRd",
                    xticklabels=50,
                    yticklabels=[l.replace("layer_", "L") for l in layers],
                )
                ax.set_xlabel("Training Steps")
                ax.set_ylabel("Layer")
                ax.set_title(f"{mod_type.capitalize()} Gradients")
        else:
            # No embed/lm_head, just show attention and mlp
            for idx, mod_type in enumerate(["attention", "mlp"]):
                ax = axes[idx]
                data = np.zeros((num_layers, num_steps))

                for i, layer in enumerate(layers):
                    data[i, :] = log.layer_norms[layer][mod_type]

                sns.heatmap(
                    data,
                    ax=ax,
                    cmap="YlOrRd",
                    xticklabels=50,
                    yticklabels=[l.replace("layer_", "L") for l in layers],
                )
                ax.set_xlabel("Training Steps")
                ax.set_ylabel("Layer")
                ax.set_title(f"{mod_type.capitalize()} Module Gradients")

    else:
        fig, ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
        data = np.zeros((num_layers, num_steps))

        for i, layer in enumerate(layers):
            data[i, :] = log.layer_norms[layer][module_type]

        sns.heatmap(
            data,
            ax=ax,
            cmap="YlOrRd",
            xticklabels=50,
            yticklabels=[l.replace("layer_", "L") for l in layers],
        )
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Layer")
        ax.set_title(f"{module_type.capitalize()} Module Gradients")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    return fig


def plot_layer_comparison(
    log: GradientNormLog,
    title: str = "Layer-wise Attention vs MLP Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Plot bar chart comparing mean gradient norms per layer.
    Includes embed and lm_head if available.

    Args:
        log: GradientNormLog with layer data
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if not log.layer_norms:
        print("No layer-level data available")
        return None

    # Check if embed and lm_head data are available
    has_embed = "embed" in log.group_norms and any(v > 0 for v in log.group_norms["embed"])
    has_lm_head = "lm_head" in log.group_norms and any(v > 0 for v in log.group_norms["lm_head"])

    layers = sorted(log.layer_norms.keys(), key=lambda x: int(x.split("_")[1]))

    # Build labels and data including embed and lm_head
    labels = []
    values = []  # Single value per component (not attention/mlp split)

    # Add embed first
    if has_embed:
        labels.append("Embed")
        values.append(np.mean(log.group_norms["embed"]))

    # Add transformer layers (attention + mlp combined for comparison with embed/lm_head)
    for layer in layers:
        layer_num = layer.replace("layer_", "")
        labels.append(f"L{layer_num}")
        attn_mean = np.mean(log.layer_norms[layer]["attention"])
        mlp_mean = np.mean(log.layer_norms[layer]["mlp"])
        values.append(attn_mean + mlp_mean)  # Combined for scale comparison

    # Add lm_head last
    if has_lm_head:
        labels.append("LM Head")
        values.append(np.mean(log.group_norms["lm_head"]))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: All components with stacked bars for layers (Attention + MLP)
    ax1 = axes[0]
    x = np.arange(len(labels))

    # Separate indices for special layers vs transformer layers
    embed_idx = labels.index("Embed") if "Embed" in labels else None
    lm_head_idx = labels.index("LM Head") if "LM Head" in labels else None
    layer_indices = [i for i, label in enumerate(labels) if label not in ["Embed", "LM Head"]]

    # Prepare data for transformer layers only (stacked)
    attn_values = np.zeros(len(labels))
    mlp_values = np.zeros(len(labels))

    for i, label in enumerate(labels):
        if label not in ["Embed", "LM Head"]:
            layer_key = f"layer_{label[1:]}"  # "L0" -> "layer_0"
            attn_values[i] = np.mean(log.layer_norms[layer_key]["attention"])
            mlp_values[i] = np.mean(log.layer_norms[layer_key]["mlp"])

    # Draw transformer layers as stacked bars (Attention bottom, MLP top)
    layer_x = [x[i] for i in layer_indices]
    layer_attn = [attn_values[i] for i in layer_indices]
    layer_mlp = [mlp_values[i] for i in layer_indices]

    ax1.bar(layer_x, layer_attn, color="#2196F3", alpha=0.8, label="Attention")
    ax1.bar(layer_x, layer_mlp, bottom=layer_attn, color="#FF9800", alpha=0.8, label="MLP")

    # Draw Embed and LM Head as separate single bars
    if embed_idx is not None:
        ax1.bar(x[embed_idx], values[embed_idx], color="#4CAF50", alpha=0.8, label="Embed")
    if lm_head_idx is not None:
        ax1.bar(x[lm_head_idx], values[lm_head_idx], color="#9C27B0", alpha=0.8, label="LM Head")

    ax1.set_xlabel("Component")
    ax1.set_ylabel("Mean Gradient Norm")
    ax1.set_title("All Components (Stacked)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Plot 2: Side-by-side bars for Attention vs MLP
    ax2 = axes[1]
    attn_means = [np.mean(log.layer_norms[l]["attention"]) for l in layers]
    mlp_means = [np.mean(log.layer_norms[l]["mlp"]) for l in layers]

    x2 = np.arange(len(layers))
    width = 0.35

    ax2.bar(x2 - width / 2, attn_means, width, label="Attention", color="#2196F3", alpha=0.8)
    ax2.bar(x2 + width / 2, mlp_means, width, label="MLP", color="#FF9800", alpha=0.8)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean Gradient Norm (log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Attention vs MLP (Layers Only)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([l.replace("layer_", "") for l in layers])
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison saved to: {save_path}")

    return fig


def generate_summary_report(
    logs: Dict[str, GradientNormLog],
    output_path: str,
    include_plots: bool = True,
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary report.

    Args:
        logs: Dictionary of experiment logs
        output_path: Base path for output files
        include_plots: Whether to generate and save plots

    Returns:
        Summary dictionary
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiments": {},
        "comparison": {},
    }

    # Per-experiment statistics
    for exp_name, log in logs.items():
        stats = log.get_summary_stats()
        summary["experiments"][exp_name] = stats

        # Generate individual plots
        if include_plots and log.layer_norms:
            plot_layer_heatmap(
                log,
                title=f"Layer Heatmap - {exp_name}",
                save_path=str(output_path / f"{exp_name}_layer_heatmap.png"),
            )
            plt.close()

            plot_layer_comparison(
                log,
                title=f"Layer Comparison - {exp_name}",
                save_path=str(output_path / f"{exp_name}_layer_comparison.png"),
            )
            plt.close()

    # Cross-experiment comparison
    if len(logs) > 1:
        # Attention vs MLP ratio comparison
        for exp_name, log in logs.items():
            attn_mean = np.mean(log.group_norms["attention"])
            mlp_mean = np.mean(log.group_norms["mlp"])
            ratio = attn_mean / mlp_mean if mlp_mean > 0 else float("inf")

            summary["comparison"][exp_name] = {
                "attention_mean": float(attn_mean),
                "mlp_mean": float(mlp_mean),
                "attention_to_mlp_ratio": float(ratio),
            }

        # Combined plot
        if include_plots:
            plot_gradient_norms(
                logs,
                title="Cross-Experiment Gradient Norm Comparison",
                save_path=str(output_path / "comparison_gradient_norms.png"),
            )
            plt.close()

    # Save summary JSON
    summary_path = output_path / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary report saved to: {summary_path}")

    return summary


def load_gradient_log_from_csv(base_path: str) -> GradientNormLog:
    """
    Load gradient norm log from CSV files.

    Args:
        base_path: Base path without extension

    Returns:
        GradientNormLog object
    """
    log = GradientNormLog()
    base_path = Path(base_path)

    # Load group norms
    group_path = base_path.with_suffix(".group_norms.csv")
    if group_path.exists():
        with open(group_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                log.steps.append(int(row["step"]))
                # Handle new columns (embed, lm_head) with backwards compatibility
                log.group_norms["embed"].append(float(row.get("embed_norm", 0)))
                log.group_norms["attention"].append(float(row["attention_norm"]))
                log.group_norms["mlp"].append(float(row["mlp_norm"]))
                log.group_norms["lm_head"].append(float(row.get("lm_head_norm", 0)))
                log.group_norms["total"].append(float(row["total_norm"]))

    # Load layer norms
    layer_path = base_path.with_suffix(".layer_norms.csv")
    if layer_path.exists():
        with open(layer_path, "r") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # Parse layer names from headers
            layer_names = set()
            for h in headers:
                if h != "step" and "_" in h:
                    layer_name = "_".join(h.split("_")[:-1])
                    layer_names.add(layer_name)

            for layer_name in layer_names:
                log.layer_norms[layer_name] = {"attention": [], "mlp": []}

            # Reset file and read data
            f.seek(0)
            next(reader)  # Skip header
            for row in reader:
                for layer_name in layer_names:
                    log.layer_norms[layer_name]["attention"].append(
                        float(row.get(f"{layer_name}_attention", 0))
                    )
                    log.layer_norms[layer_name]["mlp"].append(
                        float(row.get(f"{layer_name}_mlp", 0))
                    )

    return log


def load_gradient_log_from_json(path: str) -> GradientNormLog:
    """
    Load gradient norm log from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        GradientNormLog object
    """
    with open(path, "r") as f:
        data = json.load(f)

    log = GradientNormLog()
    log.steps = data.get("steps", [])

    # Load group norms with backwards compatibility
    default_norms = {"embed": [], "attention": [], "mlp": [], "lm_head": [], "total": []}
    loaded_norms = data.get("group_norms", {})
    for key in default_norms:
        log.group_norms[key] = loaded_norms.get(key, [])

    log.layer_norms = data.get("layer_norms", {})

    return log


def plot_gradient_evolution(
    log: GradientNormLog,
    title: str = "Gradient Norm Evolution",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Plot gradient norm evolution over training steps as line charts.
    Each line represents a different training step, with color gradient from red (early) to purple (late).

    Args:
        log: GradientNormLog with layer data
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if not log.layer_norms or len(log.steps) < 2:
        print("Need at least 2 steps of layer-level data")
        return None

    # Check if embed and lm_head data are available
    has_embed = "embed" in log.group_norms and any(v > 0 for v in log.group_norms["embed"])
    has_lm_head = "lm_head" in log.group_norms and any(v > 0 for v in log.group_norms["lm_head"])

    layers = sorted(log.layer_norms.keys(), key=lambda x: int(x.split("_")[1]))

    # Get unique steps and sample 7 for rainbow colors
    all_steps = sorted(set(log.steps))
    num_display = 7  # Rainbow: red, orange, yellow, green, blue, indigo, violet

    if len(all_steps) <= num_display:
        unique_steps = all_steps
    else:
        # Sample evenly spaced steps including first and last
        indices = np.linspace(0, len(all_steps) - 1, num_display, dtype=int)
        unique_steps = [all_steps[i] for i in indices]

    num_steps = len(unique_steps)

    # Build component labels
    labels = []
    if has_embed:
        labels.append("Embed")
    for layer in layers:
        labels.append(f"L{layer.replace('layer_', '')}")
    if has_lm_head:
        labels.append("LM Head")

    x = np.arange(len(labels))

    # Create color gradient from red to purple (rainbow order)
    colors = plt.cm.rainbow_r(np.linspace(0, 1, num_steps))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each step as a line
    for step_idx, step in enumerate(unique_steps):
        # Find indices for this step in the log
        step_indices = [i for i, s in enumerate(log.steps) if s == step]
        if not step_indices:
            continue
        idx = step_indices[0]  # Use first occurrence

        values = []

        # Add embed
        if has_embed:
            values.append(log.group_norms["embed"][idx])

        # Add transformer layers
        for layer in layers:
            attn = log.layer_norms[layer]["attention"][idx]
            mlp = log.layer_norms[layer]["mlp"][idx]
            values.append(attn + mlp)

        # Add lm_head
        if has_lm_head:
            values.append(log.group_norms["lm_head"][idx])

        # Plot line with dots
        ax.plot(x, values, marker='o', markersize=4, linewidth=1.5,
                color=colors[step_idx], label=f"Step {step}", alpha=0.8)

    ax.set_xlabel("Component", fontsize=12)
    ax.set_ylabel("Gradient Norm (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Add legend with smaller font if many steps
    if num_steps <= 10:
        ax.legend(loc="upper right", fontsize=8)
    else:
        # Show only first, middle, and last steps in legend
        handles, legend_labels = ax.get_legend_handles_labels()
        selected_indices = [0, num_steps // 2, num_steps - 1]
        ax.legend([handles[i] for i in selected_indices],
                  [legend_labels[i] for i in selected_indices],
                  loc="upper right", fontsize=8)

    # Add colorbar to show step progression (use full range)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow_r,
                                norm=plt.Normalize(vmin=all_steps[0], vmax=all_steps[-1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Training Step", fontsize=10)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Evolution plot saved to: {save_path}")

    return fig


def _smooth(data: List[float], window: int) -> np.ndarray:
    """Apply moving average smoothing."""
    if window <= 1 or len(data) < window:
        return np.array(data)

    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode="valid")

    # Pad to original length
    pad_size = len(data) - len(smoothed)
    return np.concatenate([data[:pad_size], smoothed])
