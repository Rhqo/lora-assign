"""
Hyperparameter configuration using dataclasses.
Supports easy modification and CLI override.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class DatasetType(str, Enum):
    """Supported dataset types."""
    E2E_NLG = "e2e_nlg"
    SAMSUM = "samsum"
    WIKISQL = "wikisql"
    MULTI_NLI = "multi_nli"


class TaskType(str, Enum):
    """Task types."""
    GENERATION = "generation"
    CLASSIFICATION = "classification"


class TargetModuleGroup(str, Enum):
    """LoRA target module groups."""
    ATTENTION_ONLY = "attention_only"
    MLP_ONLY = "mlp_only"
    BOTH = "both"
    ALL = "all"  # All layers including embed and lm_head


# Pre-defined target modules for each group
TARGET_MODULES = {
    TargetModuleGroup.ATTENTION_ONLY: ["q_proj", "v_proj"], # , "k_proj", "o_proj"
    TargetModuleGroup.MLP_ONLY: ["gate_proj", "up_proj", "down_proj"],
    TargetModuleGroup.BOTH: ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], # , "k_proj", "o_proj"
    TargetModuleGroup.ALL: ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"], # , "k_proj", "o_proj"
}


@dataclass
class ModelConfig:
    """Model-related configuration."""
    model_id: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    device_map: str = "auto"
    torch_dtype: str = "float16"  # "float16", "bfloat16", "float32"
    trust_remote_code: bool = False


@dataclass
class LoRAConfig:
    """LoRA-specific configuration."""
    r: int = 8  # Low-rank dimension
    lora_alpha: int = 32  # Scaling factor (alpha/r)
    lora_dropout: float = 0.1  # Dropout probability
    target_module_group: TargetModuleGroup = TargetModuleGroup.ATTENTION_ONLY
    bias: str = "none"  # "none", "all", "lora_only"

    @property
    def target_modules(self) -> List[str]:
        """Get target modules based on group selection."""
        return TARGET_MODULES[self.target_module_group]


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 100
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    warmup_ratio: float = 0.0  # Alternative to warmup_steps (0 to disable)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient clipping
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "constant"
    logging_steps: int = 10
    eval_steps: int = 10   # Evaluate every N steps
    save_steps: int = 50  # Save checkpoint every N steps (when using validation)
    save_strategy: str = "steps"  # Save based on steps
    save_total_limit: int = 2  # Keep only best and last (2 checkpoints max)
    load_best_model_at_end: bool = True  # Load best model at end
    metric_for_best_model: str = "eval_loss"  # Use validation loss
    seed: int = 42


@dataclass
class DataConfig:
    """Data processing configuration."""
    dataset_type: DatasetType = DatasetType.E2E_NLG
    task_type: TaskType = TaskType.GENERATION
    max_length: int = 128
    num_samples: int = 0  # Number of training samples (0 for full dataset)
    validation_split: float = 0.2  # Ratio for validation set (increased from 0.1)
    num_labels: int = 3  # For classification tasks (e.g., Multi-NLI has 3 classes)

    @property
    def dataset_name(self) -> str:
        """Get HuggingFace dataset name."""
        mapping = {
            DatasetType.E2E_NLG: "tuetschek/e2e_nlg",
            DatasetType.SAMSUM: "knkarthick/samsum",
            DatasetType.WIKISQL: "Salesforce/wikisql",
            DatasetType.MULTI_NLI: "nyu-mll/multi_nli",
        }
        return mapping.get(self.dataset_type, "")


@dataclass
class GradientAnalysisConfig:
    """Configuration for gradient norm analysis."""
    measure_group_norm: bool = True  # Measure total norm per module group
    measure_layer_norm: bool = True  # Measure norm per individual layer
    log_frequency: int = 1  # Log every N steps (1 = every step)
    save_format: str = "csv"  # "csv" or "json"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    enabled: bool = True  # Whether to run evaluation after training
    num_samples: int = 100  # Number of test samples to evaluate
    max_new_tokens: int = 128  # Max tokens to generate
    temperature: float = 0.7  # Sampling temperature
    top_p: float = 0.9  # Top-p sampling
    do_sample: bool = True  # Use sampling vs greedy
    batch_size: int = 32  # Batch size for generation


@dataclass
class DynamicLoRAConfig:
    """Configuration for dynamic LoRA module switching during training."""
    enabled: bool = False  # Whether to use dynamic LoRA switching
    phase_strategy: str = "both->mlp->attn"  # Phase switching strategy
    phase1_steps: tuple = (0, 100)  # Phase 1 steps
    phase2_steps: tuple = (100, 200)  # Phase 2 steps
    phase3_steps: tuple = (200, None)  # Phase 3 steps (None = until end)
    verbose: bool = True  # Print phase transitions


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gradient_analysis: GradientAnalysisConfig = field(default_factory=GradientAnalysisConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    dynamic_lora: DynamicLoRAConfig = field(default_factory=DynamicLoRAConfig)

    # Experiment metadata
    experiment_name: str = "lora_sensitivity"
    output_dir: str = "./results"

    def __post_init__(self):
        """Auto-generate experiment name if not provided."""
        if self.experiment_name == "lora_sensitivity":
            self.experiment_name = (
                f"{self.data.dataset_type.value}_"
                f"{self.lora.target_module_group.value}_"
                f"r{self.lora.r}_"
                f"lr{self.training.learning_rate}"
            )


# ============================================================
# Preset Configurations for Quick Experiments
# ============================================================

def get_e2e_attention_config() -> ExperimentConfig:
    """E2E NLG dataset with Attention-only LoRA."""
    config = ExperimentConfig()
    config.data.dataset_type = DatasetType.E2E_NLG
    config.lora.target_module_group = TargetModuleGroup.ATTENTION_ONLY
    config.experiment_name = "e2e_attention_only"
    return config


def get_e2e_mlp_config() -> ExperimentConfig:
    """E2E NLG dataset with MLP-only LoRA."""
    config = ExperimentConfig()
    config.data.dataset_type = DatasetType.E2E_NLG
    config.lora.target_module_group = TargetModuleGroup.MLP_ONLY
    config.experiment_name = "e2e_mlp_only"
    return config


def get_samsum_attention_config() -> ExperimentConfig:
    """SAMSum dataset with Attention-only LoRA."""
    config = ExperimentConfig()
    config.data.dataset_type = DatasetType.SAMSUM
    config.lora.target_module_group = TargetModuleGroup.ATTENTION_ONLY
    config.experiment_name = "samsum_attention_only"
    return config


def get_samsum_mlp_config() -> ExperimentConfig:
    """SAMSum dataset with MLP-only LoRA."""
    config = ExperimentConfig()
    config.data.dataset_type = DatasetType.SAMSUM
    config.lora.target_module_group = TargetModuleGroup.MLP_ONLY
    config.experiment_name = "samsum_mlp_only"
    return config


# All preset configurations
PRESETS = {
    "e2e_attention": get_e2e_attention_config,
    "e2e_mlp": get_e2e_mlp_config,
    "samsum_attention": get_samsum_attention_config,
    "samsum_mlp": get_samsum_mlp_config,
}


def config_to_dict(config: ExperimentConfig) -> dict:
    """Convert config to dictionary for serialization."""
    from dataclasses import asdict
    return asdict(config)


def config_from_dict(data: dict) -> ExperimentConfig:
    """Create config from dictionary."""
    model = ModelConfig(**data.get("model", {}))
    lora = LoRAConfig(**data.get("lora", {}))
    training = TrainingConfig(**data.get("training", {}))
    data_cfg = DataConfig(**data.get("data", {}))
    grad_analysis = GradientAnalysisConfig(**data.get("gradient_analysis", {}))
    eval_cfg = EvaluationConfig(**data.get("evaluation", {}))
    dynamic_lora_cfg = DynamicLoRAConfig(**data.get("dynamic_lora", {}))

    return ExperimentConfig(
        model=model,
        lora=lora,
        training=training,
        data=data_cfg,
        gradient_analysis=grad_analysis,
        evaluation=eval_cfg,
        dynamic_lora=dynamic_lora_cfg,
        experiment_name=data.get("experiment_name", "lora_sensitivity"),
        output_dir=data.get("output_dir", "./results"),
    )
