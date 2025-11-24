"""
Gradient Norm measurement callback for LoRA sensitivity analysis.
Measures gradient norms at both group level (Attention vs MLP) and layer level.
"""
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments, Trainer

import sys
sys.path.append(".")
from configs.config import GradientAnalysisConfig


# Module patterns for classification
ATTENTION_PATTERNS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PATTERNS = ["gate_proj", "up_proj", "down_proj"]

# Special layer patterns (embedding and head)
EMBED_PATTERNS = ["embed_tokens", "wte", "word_embeddings"]  # token embedding
POS_EMBED_PATTERNS = ["wpe", "position_embeddings"]  # position embedding (GPT2 style)
LM_HEAD_PATTERNS = ["lm_head", "output"]  # language model head


def measure_gradient_norms(model) -> dict:
    """
    Measure gradient norms for each module group and layer.

    Returns:
        Dictionary with gradient norms for all tracked components:
        - attention_norm, mlp_norm, total_norm (legacy)
        - embed_norm, lm_head_norm (new special layers)
        - layer_data (per-layer breakdown)
    """
    attention_grad_sq = 0.0
    mlp_grad_sq = 0.0
    embed_grad_sq = 0.0
    lm_head_grad_sq = 0.0
    total_grad_sq = 0.0

    # Layer-level tracking
    layer_grads: Dict[str, Dict[str, float]] = defaultdict(lambda: {"attention": 0.0, "mlp": 0.0})

    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue

        grad_norm_sq = param.grad.data.norm(2).item() ** 2
        total_grad_sq += grad_norm_sq

        # Check for special layers first (embedding, lm_head)
        is_special = False

        # Token embedding
        for pattern in EMBED_PATTERNS:
            if pattern in name:
                embed_grad_sq += grad_norm_sq
                is_special = True
                break

        # LM head
        if not is_special:
            for pattern in LM_HEAD_PATTERNS:
                if pattern in name:
                    lm_head_grad_sq += grad_norm_sq
                    is_special = True
                    break

        # Skip layer classification for special layers
        if is_special:
            continue

        # Classify the parameter (attention or mlp)
        module_type = None
        for pattern in ATTENTION_PATTERNS:
            if pattern in name:
                module_type = "attention"
                break
        if module_type is None:
            for pattern in MLP_PATTERNS:
                if pattern in name:
                    module_type = "mlp"
                    break

        # Extract layer index
        match = re.search(r"(?:layers?|h)\.(\d+)\.", name)
        layer_idx = int(match.group(1)) if match else None

        if module_type == "attention":
            attention_grad_sq += grad_norm_sq
            if layer_idx is not None:
                layer_grads[f"layer_{layer_idx}"]["attention"] += grad_norm_sq
        elif module_type == "mlp":
            mlp_grad_sq += grad_norm_sq
            if layer_idx is not None:
                layer_grads[f"layer_{layer_idx}"]["mlp"] += grad_norm_sq

    # Convert squared sums to L2 norms
    attention_norm = attention_grad_sq ** 0.5
    mlp_norm = mlp_grad_sq ** 0.5
    embed_norm = embed_grad_sq ** 0.5
    lm_head_norm = lm_head_grad_sq ** 0.5
    total_norm = total_grad_sq ** 0.5

    # Convert layer squared sums to norms
    layer_data = {}
    for layer_name, grads in layer_grads.items():
        layer_data[layer_name] = {
            "attention": grads["attention"] ** 0.5,
            "mlp": grads["mlp"] ** 0.5,
        }

    return {
        "attention_norm": attention_norm,
        "mlp_norm": mlp_norm,
        "embed_norm": embed_norm,
        "lm_head_norm": lm_head_norm,
        "total_norm": total_norm,
        "layer_data": layer_data,
    }


class GradientMeasuringTrainer(Trainer):
    """Custom Trainer that measures gradient norms before optimizer.step()."""

    def __init__(self, *args, gradient_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_callback = gradient_callback

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to measure gradients after backward."""
        # Call parent's training_step (does forward + backward)
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Measure gradients BEFORE optimizer.step() clears them
        # Only measure on gradient accumulation boundary
        if self.gradient_callback is not None:
            if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
                grad_norms = measure_gradient_norms(model)
                self.gradient_callback.record_gradient(
                    step=self.state.global_step + 1,
                    attention_norm=grad_norms["attention_norm"],
                    mlp_norm=grad_norms["mlp_norm"],
                    embed_norm=grad_norms["embed_norm"],
                    lm_head_norm=grad_norms["lm_head_norm"],
                    total_norm=grad_norms["total_norm"],
                    layer_data=grad_norms["layer_data"],
                )

        return loss


@dataclass
class GradientNormLog:
    """Container for gradient norm measurements."""
    # Group-level measurements (total norm per module type)
    group_norms: Dict[str, List[float]] = field(default_factory=lambda: {
        "embed": [],       # token embedding
        "attention": [],   # attention layers (q,k,v,o proj)
        "mlp": [],         # MLP layers (gate, up, down proj)
        "lm_head": [],     # language model head
        "total": [],       # total gradient norm
    })

    # Layer-level measurements (norm per layer)
    layer_norms: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # Structure: {"layer_0": {"attention": [...], "mlp": [...]}, ...}

    # Step tracking
    steps: List[int] = field(default_factory=list)

    def add_measurement(
        self,
        step: int,
        attention_norm: float,
        mlp_norm: float,
        total_norm: float,
        embed_norm: float = 0.0,
        lm_head_norm: float = 0.0,
        layer_data: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Add a single measurement."""
        self.steps.append(step)
        self.group_norms["embed"].append(embed_norm)
        self.group_norms["attention"].append(attention_norm)
        self.group_norms["mlp"].append(mlp_norm)
        self.group_norms["lm_head"].append(lm_head_norm)
        self.group_norms["total"].append(total_norm)

        if layer_data:
            for layer_name, module_norms in layer_data.items():
                if layer_name not in self.layer_norms:
                    self.layer_norms[layer_name] = {"attention": [], "mlp": []}
                self.layer_norms[layer_name]["attention"].append(module_norms.get("attention", 0.0))
                self.layer_norms[layer_name]["mlp"].append(module_norms.get("mlp", 0.0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "steps": self.steps,
            "group_norms": self.group_norms,
            "layer_norms": self.layer_norms,
        }

    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for each group."""
        import numpy as np

        stats = {}
        for group_name, norms in self.group_norms.items():
            if norms:
                stats[group_name] = {
                    "mean": float(np.mean(norms)),
                    "std": float(np.std(norms)),
                    "max": float(np.max(norms)),
                    "min": float(np.min(norms)),
                    "peak_step": int(self.steps[np.argmax(norms)]),
                }
        return stats


class GradientNormCallback(TrainerCallback):
    """
    Callback to measure gradient norms during training.

    Measures:
    1. Group-level: Total L2 norm for Attention modules vs MLP modules
    2. Layer-level: L2 norm for each layer's Attention and MLP separately

    Note: Use with GradientMeasuringTrainer which calls record_gradient()
    before optimizer.step() clears the gradients.
    """

    def __init__(
        self,
        config: GradientAnalysisConfig,
        output_dir: str,
        experiment_name: str,
    ):
        """
        Initialize callback.

        Args:
            config: Gradient analysis configuration
            output_dir: Directory to save results
            experiment_name: Name of the experiment
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.log = GradientNormLog()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_gradient(
        self,
        step: int,
        attention_norm: float,
        mlp_norm: float,
        total_norm: float,
        embed_norm: float = 0.0,
        lm_head_norm: float = 0.0,
        layer_data: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Record gradient norms. Called by GradientMeasuringTrainer.

        Args:
            step: Current training step
            attention_norm: L2 norm of attention module gradients
            mlp_norm: L2 norm of MLP module gradients
            embed_norm: L2 norm of token embedding gradients
            lm_head_norm: L2 norm of LM head gradients
            total_norm: Total L2 norm of all gradients
            layer_data: Per-layer gradient norms
        """
        # Check if we should log this step
        if step % self.config.log_frequency != 0:
            return

        # Store measurement
        self.log.add_measurement(
            step=step,
            attention_norm=attention_norm,
            mlp_norm=mlp_norm,
            total_norm=total_norm,
            embed_norm=embed_norm,
            lm_head_norm=lm_head_norm,
            layer_data=layer_data if self.config.measure_layer_norm else None,
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Save results when training ends."""
        self._save_results()

    def _measure_gradients(self, model) -> tuple:
        """
        Measure gradient norms for each module group and layer.

        Returns:
            Tuple of (attention_norm, mlp_norm, total_norm, layer_data)
        """
        attention_grad_sq = 0.0
        mlp_grad_sq = 0.0
        total_grad_sq = 0.0

        # Layer-level tracking
        layer_grads: Dict[str, Dict[str, float]] = defaultdict(lambda: {"attention": 0.0, "mlp": 0.0})

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad_norm_sq = param.grad.data.norm(2).item() ** 2
            total_grad_sq += grad_norm_sq

            # Classify the parameter
            module_type = self._classify_module(name)
            layer_idx = self._extract_layer_index(name)

            if module_type == "attention":
                attention_grad_sq += grad_norm_sq
                if layer_idx is not None:
                    layer_grads[f"layer_{layer_idx}"]["attention"] += grad_norm_sq
            elif module_type == "mlp":
                mlp_grad_sq += grad_norm_sq
                if layer_idx is not None:
                    layer_grads[f"layer_{layer_idx}"]["mlp"] += grad_norm_sq

        # Convert squared sums to L2 norms
        attention_norm = attention_grad_sq ** 0.5
        mlp_norm = mlp_grad_sq ** 0.5
        total_norm = total_grad_sq ** 0.5

        # Convert layer squared sums to norms
        layer_data = {}
        for layer_name, grads in layer_grads.items():
            layer_data[layer_name] = {
                "attention": grads["attention"] ** 0.5,
                "mlp": grads["mlp"] ** 0.5,
            }

        return attention_norm, mlp_norm, total_norm, layer_data

    def _classify_module(self, param_name: str) -> Optional[str]:
        """Classify parameter as attention or mlp based on name."""
        for pattern in ATTENTION_PATTERNS:
            if pattern in param_name:
                return "attention"
        for pattern in MLP_PATTERNS:
            if pattern in param_name:
                return "mlp"
        return None

    def _extract_layer_index(self, param_name: str) -> Optional[int]:
        """Extract layer index from parameter name."""
        # Common patterns: "layers.0.", "layer.0.", "h.0."
        match = re.search(r"(?:layers?|h)\.(\d+)\.", param_name)
        if match:
            return int(match.group(1))
        return None

    def _save_results(self):
        """Save gradient norm logs to files."""
        base_path = self.output_dir / self.experiment_name

        if self.config.save_format == "csv":
            self._save_csv(base_path)
        else:
            self._save_json(base_path)

        # Always save summary stats
        self._save_summary(base_path)

    def _save_csv(self, base_path: Path):
        """Save results as CSV files."""
        # Group-level CSV
        group_path = base_path.with_suffix(".group_norms.csv")
        with open(group_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "embed_norm", "attention_norm", "mlp_norm", "lm_head_norm", "total_norm"])
            for i, step in enumerate(self.log.steps):
                writer.writerow([
                    step,
                    self.log.group_norms["embed"][i],
                    self.log.group_norms["attention"][i],
                    self.log.group_norms["mlp"][i],
                    self.log.group_norms["lm_head"][i],
                    self.log.group_norms["total"][i],
                ])
        print(f"Group norms saved to: {group_path}")

        # Layer-level CSV (if measured)
        if self.config.measure_layer_norm and self.log.layer_norms:
            layer_path = base_path.with_suffix(".layer_norms.csv")
            with open(layer_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                layers = sorted(self.log.layer_norms.keys(), key=lambda x: int(x.split("_")[1]))
                header = ["step"]
                for layer in layers:
                    header.extend([f"{layer}_attention", f"{layer}_mlp"])
                writer.writerow(header)

                # Data rows
                for i, step in enumerate(self.log.steps):
                    row = [step]
                    for layer in layers:
                        row.append(self.log.layer_norms[layer]["attention"][i])
                        row.append(self.log.layer_norms[layer]["mlp"][i])
                    writer.writerow(row)
            print(f"Layer norms saved to: {layer_path}")

    def _save_json(self, base_path: Path):
        """Save results as JSON file."""
        json_path = base_path.with_suffix(".gradient_norms.json")
        with open(json_path, "w") as f:
            json.dump(self.log.to_dict(), f, indent=2)
        print(f"Gradient norms saved to: {json_path}")

    def _save_summary(self, base_path: Path):
        """Save summary statistics."""
        summary_path = base_path.with_suffix(".summary.json")
        summary = self.log.get_summary_stats()
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")

    def get_log(self) -> GradientNormLog:
        """Return the gradient norm log for external use."""
        return self.log
