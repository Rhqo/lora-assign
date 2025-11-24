"""
LoRA model setup utilities.
"""
from typing import Dict, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType

import sys
sys.path.append(".")
from configs.config import ModelConfig, LoRAConfig


def setup_model_with_lora(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base model and apply LoRA configuration.

    Args:
        model_config: Model configuration
        lora_config: LoRA configuration

    Returns:
        Tuple of (model with LoRA, tokenizer)
    """
    # Determine torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_config.torch_dtype, torch.float16)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        torch_dtype=torch_dtype,
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Separate target modules: LoRA vs modules_to_save
    # embed_tokens and lm_head can't use LoRA adapters, they need to be saved/trained directly
    special_modules = ["embed_tokens", "lm_head"]
    lora_targets = [m for m in lora_config.target_modules if m not in special_modules]
    modules_to_save = [m for m in lora_config.target_modules if m in special_modules]

    # Configure LoRA
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_targets if lora_targets else None,
        modules_to_save=modules_to_save if modules_to_save else None,
        bias=lora_config.bias,
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    return model, tokenizer


def count_parameters(model: PreTrainedModel) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Count by module type
    embed_params = 0
    attention_params = 0
    mlp_params = 0
    lm_head_params = 0
    other_params = 0

    embed_patterns = ["embed_tokens", "wte", "word_embeddings"]
    attention_patterns = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_patterns = ["gate_proj", "up_proj", "down_proj"]
    lm_head_patterns = ["lm_head"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        numel = param.numel()
        classified = False

        # Check embed
        for pattern in embed_patterns:
            if pattern in name:
                embed_params += numel
                classified = True
                break

        # Check attention
        if not classified:
            for pattern in attention_patterns:
                if pattern in name:
                    attention_params += numel
                    classified = True
                    break

        # Check MLP
        if not classified:
            for pattern in mlp_patterns:
                if pattern in name:
                    mlp_params += numel
                    classified = True
                    break

        # Check lm_head
        if not classified:
            for pattern in lm_head_patterns:
                if pattern in name:
                    lm_head_params += numel
                    classified = True
                    break

        if not classified:
            other_params += numel

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_embed": embed_params,
        "trainable_attention": attention_params,
        "trainable_mlp": mlp_params,
        "trainable_lm_head": lm_head_params,
        "trainable_other": other_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
    }


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """Print trainable parameter summary."""
    stats = count_parameters(model)

    print("\n" + "=" * 60)
    print("Model Parameter Summary")
    print("=" * 60)
    print(f"Total parameters:      {stats['total']:,}")
    print(f"Trainable parameters:  {stats['trainable']:,}")
    print(f"Frozen parameters:     {stats['frozen']:,}")
    print(f"Trainable ratio:       {stats['trainable_ratio']:.4%}")
    print("-" * 60)
    print("Trainable by module type:")
    print(f"  Embed (token):       {stats['trainable_embed']:,}")
    print(f"  Attention (QKV/O):   {stats['trainable_attention']:,}")
    print(f"  MLP (Gate/Up/Down):  {stats['trainable_mlp']:,}")
    print(f"  LM Head:             {stats['trainable_lm_head']:,}")
    print(f"  Other:               {stats['trainable_other']:,}")
    print("=" * 60 + "\n")


def get_lora_layer_info(model: PreTrainedModel) -> Dict[str, Dict[str, bool]]:
    """
    Get information about which layers have LoRA applied.

    Args:
        model: Model with LoRA

    Returns:
        Dictionary mapping layer index to applied module types
    """
    import re

    layer_info: Dict[str, Dict[str, bool]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Extract layer index
        match = re.search(r"(?:layers?|h)\.(\d+)\.", name)
        if not match:
            continue

        layer_idx = match.group(1)
        if layer_idx not in layer_info:
            layer_info[layer_idx] = {"attention": False, "mlp": False}

        # Check module type
        if any(p in name for p in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            layer_info[layer_idx]["attention"] = True
        elif any(p in name for p in ["gate_proj", "up_proj", "down_proj"]):
            layer_info[layer_idx]["mlp"] = True

    return layer_info


def freeze_layers(
    model: PreTrainedModel,
    layers_to_freeze: Optional[list] = None,
    freeze_attention: bool = False,
    freeze_mlp: bool = False,
) -> None:
    """
    Selectively freeze layers in the model.

    Args:
        model: Model to modify
        layers_to_freeze: List of layer indices to freeze completely
        freeze_attention: Freeze all attention LoRA modules
        freeze_mlp: Freeze all MLP LoRA modules
    """
    import re

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        should_freeze = False

        # Check layer index
        if layers_to_freeze:
            match = re.search(r"(?:layers?|h)\.(\d+)\.", name)
            if match and int(match.group(1)) in layers_to_freeze:
                should_freeze = True

        # Check module type
        if freeze_attention and any(p in name for p in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            should_freeze = True
        if freeze_mlp and any(p in name for p in ["gate_proj", "up_proj", "down_proj"]):
            should_freeze = True

        if should_freeze:
            param.requires_grad = False
