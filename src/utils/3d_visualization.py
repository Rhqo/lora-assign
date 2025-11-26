"""
3D Visualization utilities for gradient norm analysis.
Creates 3D surface plots with:
- X-axis: Layer (Embed, L0-L21, LM Head)
- Y-axis: Training Step
- Z-axis: Gradient Norm
"""
import numpy as np
import matplotlib
import sys

# Set interactive backend for 3D visualization
# Try Qt5Agg first (more reliable with uv Python)
_backend_set = False
for backend in ['Qt5Agg', 'GTK3Agg', 'WXAgg', 'MacOSX']:
    try:
        matplotlib.use(backend, force=True)
        _backend_set = True
        break
    except:
        continue

if not _backend_set:
    print("Warning: No interactive backend available.")
    print("Install Qt5: pip install PyQt5")
    print("Or use plotly for web-based interactive plots")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List

sys.path.append(".")
from src.callbacks.gradient_norm import GradientNormLog
from src.utils.visualization import load_gradient_log_from_csv


def plot_gradient_3d_surface(
    log: GradientNormLog,
    module_type: str = "combined",
    title: str = "3D Gradient Evolution - Surface",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
    view_angle: tuple = (30, -60),
    interactive: bool = True,
) -> plt.Figure:
    """
    Plot 3D surface of gradient norms.

    Args:
        log: GradientNormLog with layer data
        module_type: "combined" (attention+mlp), "attention", or "mlp"
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        view_angle: (elevation, azimuth) for 3D view

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
    num_steps = len(log.steps)

    # Build component labels
    labels = []
    if has_embed:
        labels.append("Embed")
    for layer in layers:
        labels.append(f"L{layer.replace('layer_', '')}")
    if has_lm_head:
        labels.append("LM Head")

    num_components = len(labels)

    # Create meshgrid
    X = np.arange(num_components)  # Layer index
    Y = np.array(log.steps)        # Training steps
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    # Build Z data (gradient norms)
    Z = np.zeros((num_steps, num_components))

    for step_idx in range(num_steps):
        comp_idx = 0

        # Add embed
        if has_embed:
            Z[step_idx, comp_idx] = log.group_norms["embed"][step_idx]
            comp_idx += 1

        # Add transformer layers
        for layer in layers:
            attn = log.layer_norms[layer]["attention"][step_idx]
            mlp = log.layer_norms[layer]["mlp"][step_idx]

            if module_type == "combined":
                Z[step_idx, comp_idx] = attn + mlp
            elif module_type == "attention":
                Z[step_idx, comp_idx] = attn
            elif module_type == "mlp":
                Z[step_idx, comp_idx] = mlp
            comp_idx += 1

        # Add lm_head
        if has_lm_head:
            Z[step_idx, comp_idx] = log.group_norms["lm_head"][step_idx]

    # Create figure with 3D projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z,
        cmap='rainbow',
        edgecolor='none',
        alpha=0.8,
        antialiased=True
    )

    # Set labels
    ax.set_xlabel('Layer', fontsize=12, labelpad=10)
    ax.set_ylabel('Training Step', fontsize=12, labelpad=10)
    ax.set_zlabel('Gradient Norm', fontsize=12, labelpad=10)

    # Set x-axis ticks (layer labels)
    ax.set_xticks(X)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Gradient Norm', fontsize=10)

    # Set title
    module_suffix = f" ({module_type.capitalize()})" if module_type != "combined" else ""
    plt.title(f"{title}{module_suffix}", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D surface plot saved to: {save_path}")

    if interactive and not save_path:
        print("\nInteractive 3D plot opened!")
        print("Controls:")
        print("  - Left mouse: Rotate")
        print("  - Right mouse: Zoom")
        print("  - Middle mouse: Pan")
        plt.show()

    return fig


def plot_gradient_3d_wireframe(
    log: GradientNormLog,
    module_type: str = "combined",
    title: str = "3D Gradient Evolution - Wireframe",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
    view_angle: tuple = (30, -60),
    interactive: bool = True,
) -> plt.Figure:
    """
    Plot 3D wireframe of gradient norms.
    More lightweight visualization that shows the structure clearly.

    Args:
        log: GradientNormLog with layer data
        module_type: "combined" (attention+mlp), "attention", or "mlp"
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        view_angle: (elevation, azimuth) for 3D view

    Returns:
        Matplotlib figure
    """
    if not log.layer_norms or len(log.steps) < 2:
        print("Need at least 2 steps of layer-level data")
        return None

    has_embed = "embed" in log.group_norms and any(v > 0 for v in log.group_norms["embed"])
    has_lm_head = "lm_head" in log.group_norms and any(v > 0 for v in log.group_norms["lm_head"])

    layers = sorted(log.layer_norms.keys(), key=lambda x: int(x.split("_")[1]))
    num_steps = len(log.steps)

    labels = []
    if has_embed:
        labels.append("Embed")
    for layer in layers:
        labels.append(f"L{layer.replace('layer_', '')}")
    if has_lm_head:
        labels.append("LM Head")

    num_components = len(labels)

    X = np.arange(num_components)
    Y = np.array(log.steps)
    X_mesh, Y_mesh = np.meshgrid(X, Y)

    Z = np.zeros((num_steps, num_components))

    for step_idx in range(num_steps):
        comp_idx = 0

        if has_embed:
            Z[step_idx, comp_idx] = log.group_norms["embed"][step_idx]
            comp_idx += 1

        for layer in layers:
            attn = log.layer_norms[layer]["attention"][step_idx]
            mlp = log.layer_norms[layer]["mlp"][step_idx]

            if module_type == "combined":
                Z[step_idx, comp_idx] = attn + mlp
            elif module_type == "attention":
                Z[step_idx, comp_idx] = attn
            elif module_type == "mlp":
                Z[step_idx, comp_idx] = mlp
            comp_idx += 1

        if has_lm_head:
            Z[step_idx, comp_idx] = log.group_norms["lm_head"][step_idx]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot wireframe
    ax.plot_wireframe(
        X_mesh, Y_mesh, Z,
        color='#2196F3',
        linewidth=0.5,
        alpha=0.8
    )

    ax.set_xlabel('Layer', fontsize=12, labelpad=10)
    ax.set_ylabel('Training Step', fontsize=12, labelpad=10)
    ax.set_zlabel('Gradient Norm', fontsize=12, labelpad=10)

    ax.set_xticks(X)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    module_suffix = f" ({module_type.capitalize()})" if module_type != "combined" else ""
    plt.title(f"{title}{module_suffix}", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D wireframe plot saved to: {save_path}")

    if interactive and not save_path:
        print("\nInteractive 3D wireframe opened!")
        print("Controls:")
        print("  - Left mouse: Rotate")
        print("  - Right mouse: Zoom")
        print("  - Middle mouse: Pan")
        plt.show()

    return fig


def plot_gradient_3d_bars(
    log: GradientNormLog,
    module_type: str = "stacked",
    title: str = "3D Gradient Evolution - Stacked Bars",
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10),
    view_angle: tuple = (25, -45),
    step_sample: int = 10,
    interactive: bool = True,
) -> plt.Figure:
    """
    Plot 3D stacked bar chart of gradient norms.
    For transformer layers: Attention bars (blue) at bottom, MLP bars (orange) stacked on top.
    For embed/lm_head: Single bars.

    Args:
        log: GradientNormLog with layer data
        module_type: "stacked" (attention+mlp stacked), "attention", or "mlp"
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        view_angle: (elevation, azimuth) for 3D view
        step_sample: Number of steps to sample (to avoid cluttering)

    Returns:
        Matplotlib figure
    """
    if not log.layer_norms or len(log.steps) < 2:
        print("Need at least 2 steps of layer-level data")
        return None

    has_embed = "embed" in log.group_norms and any(v > 0 for v in log.group_norms["embed"])
    has_lm_head = "lm_head" in log.group_norms and any(v > 0 for v in log.group_norms["lm_head"])

    layers = sorted(log.layer_norms.keys(), key=lambda x: int(x.split("_")[1]))

    labels = []
    if has_embed:
        labels.append("Embed")
    for layer in layers:
        labels.append(f"L{layer.replace('layer_', '')}")
    if has_lm_head:
        labels.append("LM Head")

    num_components = len(labels)

    # Sample steps evenly
    all_steps = list(range(len(log.steps)))
    if len(all_steps) > step_sample:
        indices = np.linspace(0, len(all_steps) - 1, step_sample, dtype=int)
        sampled_step_indices = [all_steps[i] for i in indices]
    else:
        sampled_step_indices = all_steps

    num_sampled_steps = len(sampled_step_indices)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Bar dimensions - make them solid rectangular prisms
    dx = 0.8  # Width along x-axis (layer axis) - wider for better visibility
    if len(log.steps) > 1:
        # Calculate depth based on step spacing
        step_range = log.steps[-1] - log.steps[0]
        step_spacing = step_range / max(1, num_sampled_steps - 1)
        dy = step_spacing * 0.7  # 70% of spacing for clear separation
    else:
        dy = 1.0

    # Colors: Attention (blue), MLP (orange), Embed (green), LM Head (purple)
    color_attn = "#2196F3"
    color_mlp = "#FF9800"
    color_embed = "#4CAF50"
    color_lm_head = "#9C27B0"

    for i, step_idx in enumerate(sampled_step_indices):
        step = log.steps[step_idx]

        comp_idx = 0

        # Embed: single bar
        if has_embed:
            z_val = log.group_norms["embed"][step_idx]
            ax.bar3d(comp_idx - dx/2, step - dy/2, 0, dx, dy, z_val,
                    color=color_embed, alpha=0.8)
            comp_idx += 1

        # Transformer layers: stacked bars
        for layer in layers:
            attn = log.layer_norms[layer]["attention"][step_idx]
            mlp = log.layer_norms[layer]["mlp"][step_idx]

            if module_type == "stacked":
                # Draw attention bar at bottom (z=0)
                ax.bar3d(comp_idx - dx/2, step - dy/2, 0, dx, dy, attn,
                        color=color_attn, alpha=0.8)
                # Draw mlp bar on top (z=attn)
                ax.bar3d(comp_idx - dx/2, step - dy/2, attn, dx, dy, mlp,
                        color=color_mlp, alpha=0.8)
            elif module_type == "attention":
                ax.bar3d(comp_idx - dx/2, step - dy/2, 0, dx, dy, attn,
                        color=color_attn, alpha=0.8)
            elif module_type == "mlp":
                ax.bar3d(comp_idx - dx/2, step - dy/2, 0, dx, dy, mlp,
                        color=color_mlp, alpha=0.8)

            comp_idx += 1

        # LM Head: single bar
        if has_lm_head:
            z_val = log.group_norms["lm_head"][step_idx]
            ax.bar3d(comp_idx - dx/2, step - dy/2, 0, dx, dy, z_val,
                    color=color_lm_head, alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12, labelpad=10)
    ax.set_ylabel('Training Step', fontsize=12, labelpad=10)
    ax.set_zlabel('Gradient Norm', fontsize=12, labelpad=10)

    ax.set_xticks(np.arange(num_components))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Add legend
    if module_type == "stacked":
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_attn, alpha=0.8, label='Attention'),
            Patch(facecolor=color_mlp, alpha=0.8, label='MLP')
        ]
        if has_embed:
            legend_elements.insert(0, Patch(facecolor=color_embed, alpha=0.8, label='Embed'))
        if has_lm_head:
            legend_elements.append(Patch(facecolor=color_lm_head, alpha=0.8, label='LM Head'))
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    module_suffix = f" ({module_type.capitalize()})" if module_type != "stacked" else ""
    plt.title(f"{title}{module_suffix}", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D bar plot saved to: {save_path}")

    if interactive and not save_path:
        print("\nInteractive 3D stacked bar plot opened!")
        print("Controls:")
        print("  - Left mouse: Rotate")
        print("  - Right mouse: Zoom")
        print("  - Middle mouse: Pan")
        plt.show()

    return fig


def plot_gradient_3d_all(
    log: GradientNormLog,
    title_prefix: str = "3D Gradient Evolution",
    save_dir: Optional[str] = None,
    figsize: tuple = (14, 10),
    interactive: bool = True,
) -> List[plt.Figure]:
    """
    Generate all 3D visualizations (surface, wireframe, bars) for gradient norms.

    Args:
        log: GradientNormLog with layer data
        title_prefix: Prefix for plot titles
        save_dir: Directory to save figures
        figsize: Figure size

    Returns:
        List of Matplotlib figures
    """
    from pathlib import Path

    figures = []

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Surface plot
    fig_surface = plot_gradient_3d_surface(
        log,
        title=f"{title_prefix} - Surface",
        save_path=str(save_dir / "gradient_3d_surface.png") if save_dir else None,
        figsize=figsize,
        interactive=interactive,
    )
    if fig_surface:
        figures.append(fig_surface)

    # Wireframe plot
    fig_wireframe = plot_gradient_3d_wireframe(
        log,
        title=f"{title_prefix} - Wireframe",
        save_path=str(save_dir / "gradient_3d_wireframe.png") if save_dir else None,
        figsize=figsize,
        interactive=interactive,
    )
    if fig_wireframe:
        figures.append(fig_wireframe)

    # Bar plot
    fig_bars = plot_gradient_3d_bars(
        log,
        title=f"{title_prefix} - Bars",
        save_path=str(save_dir / "gradient_3d_bars.png") if save_dir else None,
        figsize=figsize,
        interactive=interactive,
    )
    if fig_bars:
        figures.append(fig_bars)

    return figures


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate 3D gradient visualizations (interactive)")
    parser.add_argument("--input", "-i", required=True, help="Path to gradient_analysis CSV base (without extension)")
    parser.add_argument("--output", "-o", help="Output directory to save plots (if not provided, shows interactive plot)")
    parser.add_argument("--type", "-t", default="bars", choices=["surface", "wireframe", "bars", "all"],
                        help="Type of 3D plot (default: bars)")
    parser.add_argument("--module", "-m", default="stacked", choices=["stacked", "attention", "mlp", "combined"],
                        help="Module type to visualize (default: stacked for bars, combined for surface/wireframe)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Disable interactive mode (use when saving only)")

    args = parser.parse_args()

    # Load data
    print(f"Loading gradient data from: {args.input}")
    log = load_gradient_log_from_csv(args.input)
    print(f"Loaded {len(log.steps)} steps of gradient data")

    interactive = not args.no_interactive
    save_dir = args.output

    from pathlib import Path
    if save_dir:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to: {output_dir}")

    if args.type == "all":
        print("\nGenerating all 3D visualizations...")
        plot_gradient_3d_all(
            log,
            save_dir=save_dir,
            interactive=interactive,
        )
    elif args.type == "surface":
        print("\nGenerating 3D surface plot...")
        plot_gradient_3d_surface(
            log,
            module_type=args.module,
            save_path=str(output_dir / "gradient_3d_surface.png") if save_dir else None,
            interactive=interactive,
        )
    elif args.type == "wireframe":
        print("\nGenerating 3D wireframe plot...")
        plot_gradient_3d_wireframe(
            log,
            module_type=args.module,
            save_path=str(output_dir / "gradient_3d_wireframe.png") if save_dir else None,
            interactive=interactive,
        )
    elif args.type == "bars":
        print("\nGenerating 3D bar plot...")
        plot_gradient_3d_bars(
            log,
            module_type=args.module,
            save_path=str(output_dir / "gradient_3d_bars.png") if save_dir else None,
            interactive=interactive,
        )
