"""
Dynamic LoRA Module Switching Callback.

Enables/disables different LoRA modules at specific training steps:
- Phase 1: Attention modules only
- Phase 2: MLP modules only
- Phase 3: Both modules active
"""
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from typing import Dict, List, Optional
import torch.nn as nn


class DynamicLoRACallback(TrainerCallback):
    """
    Callback to dynamically switch LoRA modules during training.

    Example:
        Phase 1 (steps 0-100): Only attention LoRA active
        Phase 2 (steps 100-200): Only MLP LoRA active
        Phase 3 (steps 200+): Both active
    """

    def __init__(
        self,
        phase_configs: List[Dict[str, any]],
        verbose: bool = True,
    ):
        """
        Args:
            phase_configs: List of phase configurations
                Each config: {
                    "start_step": int,
                    "end_step": int or None (None = until end),
                    "active_modules": ["attention", "mlp"],
                    "layer_range": (start, end) or None (None = all layers)
                }
            verbose: Print phase transitions
        """
        self.phase_configs = sorted(phase_configs, key=lambda x: x["start_step"])
        self.verbose = verbose
        self.current_phase_idx = -1

        # LoRA module patterns
        self.attention_patterns = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.mlp_patterns = ["gate_proj", "up_proj", "down_proj"]

    def _get_current_phase(self, step: int) -> Optional[Dict]:
        """Get the phase config for current step."""
        for phase in self.phase_configs:
            start = phase["start_step"]
            end = phase.get("end_step")
            # Handle None as infinity (until end of training)
            if end is None:
                end = float('inf')
            if start <= step < end:
                return phase
        return None

    def _is_lora_param(self, name: str) -> bool:
        """Check if parameter is a LoRA parameter."""
        return "lora_A" in name or "lora_B" in name

    def _get_module_type(self, name: str) -> Optional[str]:
        """Determine module type from parameter name."""
        # Check attention modules
        if any(pattern in name for pattern in self.attention_patterns):
            return "attention"
        # Check MLP modules
        elif any(pattern in name for pattern in self.mlp_patterns):
            return "mlp"
        return None

    def _get_layer_number(self, name: str) -> Optional[int]:
        """Extract layer number from parameter name."""
        import re
        # Match patterns like "layers.12."
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return None

    def _update_lora_modules(self, model: nn.Module, active_modules: List[str], layer_range: Optional[tuple] = None):
        """Enable/disable LoRA modules based on active_modules list and layer_range."""
        total_params = 0
        changed_to_active = 0
        changed_to_inactive = 0
        currently_active = 0
        currently_inactive = 0

        for name, param in model.named_parameters():
            if not self._is_lora_param(name):
                continue

            total_params += 1
            module_type = self._get_module_type(name)

            if module_type is None:
                continue

            # Check layer range if specified
            should_activate = module_type in active_modules

            if layer_range is not None:
                layer_num = self._get_layer_number(name)
                # If it's a layer parameter, check if it's in range
                if layer_num is not None:
                    start_layer, end_layer = layer_range
                    # end_layer is inclusive
                    if not (start_layer <= layer_num <= end_layer):
                        should_activate = False

            if should_activate and not param.requires_grad:
                param.requires_grad = True
                changed_to_active += 1
            elif not should_activate and param.requires_grad:
                param.requires_grad = False
                changed_to_inactive += 1

            # Count current state
            if param.requires_grad:
                currently_active += 1
            else:
                currently_inactive += 1

        return {
            "total_lora_params": total_params,
            "changed_to_active": changed_to_active,
            "changed_to_inactive": changed_to_inactive,
            "currently_active": currently_active,
            "currently_inactive": currently_inactive,
        }

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs,
    ):
        """Called at the beginning of each training step."""
        current_step = state.global_step

        # Get current phase
        phase = self._get_current_phase(current_step)

        if phase is None:
            return

        # Check if we need to transition to a new phase
        phase_idx = self.phase_configs.index(phase)

        if phase_idx != self.current_phase_idx:
            # Phase transition!
            self.current_phase_idx = phase_idx
            active_modules = phase["active_modules"]
            layer_range = phase.get("layer_range")

            # Update LoRA module states
            stats = self._update_lora_modules(model, active_modules, layer_range)

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"[Phase {phase_idx + 1}] Step {current_step}: Switching LoRA modules")
                print(f"  Active modules: {', '.join(active_modules)}")
                if layer_range:
                    print(f"  Layer range: L{layer_range[0]}-L{layer_range[1]}")
                print(f"  Changed to active: {stats['changed_to_active']}")
                print(f"  Changed to inactive: {stats['changed_to_inactive']}")
                print(f"  Currently active: {stats['currently_active']}/{stats['total_lora_params']}")
                print(f"  Currently inactive: {stats['currently_inactive']}/{stats['total_lora_params']}")
                print(f"{'='*60}\n")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs,
    ):
        """Initialize LoRA modules at training start."""
        # Set initial phase (step 0)
        phase = self._get_current_phase(0)
        if phase:
            active_modules = phase["active_modules"]
            layer_range = phase.get("layer_range")
            stats = self._update_lora_modules(model, active_modules, layer_range)

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"[Initial Phase] Starting with LoRA modules: {', '.join(active_modules)}")
                if layer_range:
                    print(f"  Layer range: L{layer_range[0]}-L{layer_range[1]}")
                print(f"  Total LoRA params: {stats['total_lora_params']}")
                print(f"  Currently active: {stats['currently_active']}")
                print(f"  Currently inactive: {stats['currently_inactive']}")
                print(f"{'='*60}\n")


def create_phased_lora_config(
    phase1_steps: tuple = (0, 100),
    phase2_steps: tuple = (100, 200),
    phase3_steps: tuple = (200, None),
    phase1_modules: List[str] = None,
    phase2_modules: List[str] = None,
    phase3_modules: List[str] = None,
) -> List[Dict]:
    """
    Create a flexible 3-phase LoRA configuration.

    Args:
        phase1_steps: (start, end) for phase 1
        phase2_steps: (start, end) for phase 2
        phase3_steps: (start, end) for phase 3 (end=None means until training end)
        phase1_modules: Active modules for phase 1 (default: ["attention"])
        phase2_modules: Active modules for phase 2 (default: ["mlp"])
        phase3_modules: Active modules for phase 3 (default: ["attention", "mlp"])

    Returns:
        List of phase configurations

    Examples:
        # Strategy 1: Attention → MLP → Both
        create_phased_lora_config(
            phase1_modules=["attention"],
            phase2_modules=["mlp"],
            phase3_modules=["attention", "mlp"]
        )

        # Strategy 2: Both → MLP → Attention (recommended)
        create_phased_lora_config(
            phase1_modules=["attention", "mlp"],
            phase2_modules=["mlp"],
            phase3_modules=["attention"]
        )
    """
    # Set defaults
    if phase1_modules is None:
        phase1_modules = ["attention"]
    if phase2_modules is None:
        phase2_modules = ["mlp"]
    if phase3_modules is None:
        phase3_modules = ["attention", "mlp"]

    phases = []

    if phase1_steps:
        phases.append({
            "start_step": phase1_steps[0],
            "end_step": phase1_steps[1],
            "active_modules": phase1_modules,
        })

    if phase2_steps:
        phases.append({
            "start_step": phase2_steps[0],
            "end_step": phase2_steps[1],
            "active_modules": phase2_modules,
        })

    if phase3_steps:
        phases.append({
            "start_step": phase3_steps[0],
            "end_step": phase3_steps[1],
            "active_modules": phase3_modules,
        })

    return phases


def create_layerwise_lora_config(
    num_layers: int = 22,
    phase1_steps: tuple = (0, 200),
    phase2_steps: tuple = (200, None),
    modules: List[str] = None,
) -> List[Dict]:
    """
    Create layer-wise progressive LoRA configuration.

    Args:
        num_layers: Total number of layers (default: 22 for TinyLlama)
        phase1_steps: (start, end) for early layers phase
        phase2_steps: (start, end) for late layers phase
        modules: Active module types (default: ["attention", "mlp"])

    Returns:
        List of phase configurations

    Example:
        # Early layers (L0-L10) → Late layers (L11-L21)
        create_layerwise_lora_config(
            num_layers=22,
            phase1_steps=(0, 200),
            phase2_steps=(200, None),
        )
    """
    if modules is None:
        modules = ["attention", "mlp"]

    # Split layers in half
    mid_layer = num_layers // 2
    early_layers = (0, mid_layer - 1)
    late_layers = (mid_layer, num_layers - 1)

    phases = []

    if phase1_steps:
        phases.append({
            "start_step": phase1_steps[0],
            "end_step": phase1_steps[1],
            "active_modules": modules,
            "layer_range": early_layers,
        })

    if phase2_steps:
        phases.append({
            "start_step": phase2_steps[0],
            "end_step": phase2_steps[1],
            "active_modules": modules,
            "layer_range": late_layers,
        })

    return phases
