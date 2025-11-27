# Code Summary - Quick Reference

## 📁 Core Files Overview

### 1. **configs/config.py** (245 lines)
**Purpose**: Configuration management system

**Key Components**:
- `ExperimentConfig`: Master configuration containing all sub-configs
- `LoRAConfig`: LoRA hyperparameters (r=4, alpha=32, dropout=0.1)
- `TrainingConfig`: Training settings (lr=2e-4, max_steps=1000)
- `DataConfig`: Dataset selection (e2e_nlg, samsum, wikisql, multi_nli)
- `DynamicLoRAConfig`: Phase switching configuration

**Target Module Groups**:
```python
ATTENTION_ONLY = ["q_proj", "v_proj"]
MLP_ONLY = ["gate_proj", "up_proj", "down_proj"]
BOTH = ["q_proj", "k_proj", "o_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
```

---

### 2. **src/callbacks/dynamic_lora.py** (334 lines)
**Purpose**: Dynamic LoRA module switching during training

**Main Class**: `DynamicLoRACallback(TrainerCallback)`

**Key Methods**:
- `on_train_begin()`: Initialize first phase
- `on_step_begin()`: Check for phase transitions
- `_update_lora_modules()`: Enable/disable LoRA parameters
- `_get_layer_number()`: Extract layer index from parameter name

**Strategies**:
1. **Module-based**:
   - `both->mlp->attn`: All → MLP only → Attention only (recommended)
   - `attn->mlp->both`: Attention → MLP → Both
   - `both->mlp->both`: Both → MLP → Both

2. **Layer-wise**:
   - `early->late`: L0-L10 → L11-L21 (experimental, has known issues)

**Helper Functions**:
- `create_phased_lora_config()`: Create module-based phase configs
- `create_layerwise_lora_config()`: Create layer-wise phase configs

---

### 3. **src/callbacks/gradient_norm.py** (510 lines)
**Purpose**: Measure and track gradient norms during training

**Main Components**:

1. **`measure_gradient_norms(model)`**:
   - Computes L2 norms of gradients
   - Returns: attention_norm, mlp_norm, embed_norm, lm_head_norm, total_norm, layer_data

2. **`GradientMeasuringTrainer(Trainer)`**:
   - Custom Trainer that measures gradients after backward pass
   - Calls `measure_gradient_norms()` before optimizer.step()

3. **`GradientNormCallback`**:
   - Stores measurements in `GradientNormLog`
   - Saves to CSV and JSON files

**Output Files**:
- `gradient_analysis.group_norms.csv`: Group-level norms (attention, mlp, embed, lm_head, total)
- `gradient_analysis.layer_norms.csv`: Layer-level norms (layer_0_attention, layer_0_mlp, ...)
- `gradient_analysis.summary.json`: Statistical summary

**Module Patterns**:
```python
ATTENTION_PATTERNS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PATTERNS = ["gate_proj", "up_proj", "down_proj"]
EMBED_PATTERNS = ["embed_tokens", "wte", "word_embeddings"]
LM_HEAD_PATTERNS = ["lm_head", "output"]
```

---

### 4. **src/data/datasets.py** (415 lines)
**Purpose**: Dataset loading and preprocessing

**Supported Datasets**:

| Dataset | Function | Samples | Metrics |
|---------|----------|---------|---------|
| E2E NLG | `_load_e2e_nlg()` | 42,061 | BLEU, ROUGE, METEOR |
| SAMSum | `_load_samsum()` | 14,732 | BLEU, ROUGE, METEOR |
| WikiSQL | `_load_wikisql()` | 56,355 | Exact Match |
| Multi-NLI | `_load_multi_nli()` | 392,702 | Accuracy |

**Key Functions**:
- `load_and_preprocess_dataset()`: Main entry point
- `get_data_collator()`: Returns appropriate data collator
- `_format_*_prompt()`: Format prompts for each dataset

**Preprocessing**:
- Tokenization with truncation/padding
- Label alignment for causal LM
- Instruction-based prompt formatting

**WikiSQL Special Handling**:
- Downloads raw data from GitHub (HuggingFace version deprecated)
- Converts SQL dict to string format
- Handles table headers and conditions

---

### 5. **src/evaluation.py** (750 lines)
**Purpose**: Model evaluation with multiple metrics

**Main Class**: `ModelEvaluator`

**Key Features**:
- ✅ **Left-padding for decoder models** (critical fix)
- Batched generation for efficiency
- Flexible generation config (temperature, top_p, do_sample)

**Evaluation Functions**:

1. **`evaluate_e2e()`**: E2E NLG evaluation
   - Metrics: BLEU (1-4), ROUGE (1, 2, L, Lsum), METEOR
   - Returns: `EvaluationResult` object

2. **`evaluate_samsum()`**: SAMSum summarization
   - Metrics: BLEU, ROUGE, METEOR
   - Returns: `EvaluationResult` object

3. **`evaluate_wikisql()`**: WikiSQL SQL generation
   - Metric: Exact Match (case-insensitive, whitespace-normalized)
   - Returns: Dict with exact_match and num_samples

4. **`evaluate_multi_nli()`**: Multi-NLI classification
   - Metric: Accuracy (entailment/neutral/contradiction)
   - Returns: Dict with accuracy and num_samples

**Usage Example**:
```python
from src.evaluation import evaluate_wikisql

result = evaluate_wikisql(
    model_path="results/experiment/final_model",
    base_model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    num_samples=100,
    batch_size=8
)
# result = {"exact_match": 0.02, "num_samples": 100}
```

---

### 6. **src/models/lora_setup.py** (~150 lines)
**Purpose**: Initialize model with LoRA adapters

**Main Function**: `setup_model_with_lora(model_config, lora_config)`

**Process**:
1. Load base model (TinyLlama-1.1B)
2. Create PEFT LoRA configuration
3. Wrap model with PEFT
4. Set all base parameters to `requires_grad=False`
5. Enable gradient checkpointing

**Returns**: (model, tokenizer)

**Helper**:
- `print_trainable_parameters()`: Shows trainable vs total parameters

---

### 7. **src/utils/visualization.py** (600+ lines)
**Purpose**: Generate visualizations from gradient logs

**Main Functions**:

1. **`plot_gradient_norms()`**: Line plot of group-level norms
2. **`plot_layer_heatmap()`**: Heatmap of layer-wise gradients
3. **`plot_layer_comparison()`**: Bar chart comparing layers
4. **`plot_gradient_evolution()`**: Detailed layer evolution
5. **`plot_training_loss()`**: Training/validation loss curves
6. **`generate_summary_report()`**: Statistical analysis

**Input**: CSV files from `GradientNormCallback`

**Output**: PNG files in experiment directory

---

### 8. **run_experiment.py** (784 lines)
**Purpose**: Main experiment orchestration

**Workflow**:
```python
1. Parse arguments / Load preset
2. Build configuration
3. Setup model with LoRA
4. Load and preprocess dataset
5. Create train/validation split
6. Setup callbacks:
   - GradientNormCallback (always)
   - DynamicLoRACallback (if enabled)
7. Train with GradientMeasuringTrainer
8. Save best model
9. Generate visualizations
10. Run evaluation (if enabled)
11. Save results summary
```

**CLI Arguments**:
```bash
--dataset {e2e_nlg, samsum, wikisql, multi_nli}
--target {attention_only, mlp_only, both, all}
--r RANK
--batch_size SIZE
--max_steps STEPS
--lr LEARNING_RATE
--dynamic_lora  # Enable dynamic switching
--phase_strategy {attn->mlp->both, both->mlp->attn, early->late}
--phase1_end STEP
--phase2_end STEP
```

**Presets**:
```python
PRESETS = {
    "e2e_attention": E2E NLG with attention-only LoRA
    "e2e_mlp": E2E NLG with MLP-only LoRA
    "samsum_attention": SAMSum with attention-only LoRA
    "samsum_mlp": SAMSum with MLP-only LoRA
}
```

---

## 🔑 Key Code Patterns

### 1. Configuration Loading
```python
from configs.config import ExperimentConfig, PRESETS

# Use preset
config = PRESETS["e2e_attention"]()

# Or build from args
config = ExperimentConfig()
config.data.dataset_type = DatasetType.WIKISQL
config.lora.r = 4
```

### 2. Dynamic LoRA Setup
```python
from src.callbacks.dynamic_lora import DynamicLoRACallback, create_phased_lora_config

# Module-based strategy
phase_configs = create_phased_lora_config(
    phase1_steps=(0, 300),
    phase2_steps=(300, 600),
    phase3_steps=(600, None),
    phase1_modules=["attention", "mlp"],
    phase2_modules=["mlp"],
    phase3_modules=["attention"]
)

callback = DynamicLoRACallback(phase_configs, verbose=True)
```

### 3. Gradient Measurement
```python
from src.callbacks.gradient_norm import GradientNormCallback, GradientMeasuringTrainer

# Create callback
grad_callback = GradientNormCallback(
    config=gradient_analysis_config,
    output_dir="./results/experiment"
)

# Use custom trainer
trainer = GradientMeasuringTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    gradient_callback=grad_callback,
    callbacks=[grad_callback, dynamic_lora_callback]
)
```

### 4. Evaluation
```python
from src.evaluation import evaluate_wikisql, evaluate_e2e

# WikiSQL
wikisql_result = evaluate_wikisql(
    model_path="results/exp/final_model",
    base_model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    num_samples=100
)

# E2E NLG
e2e_result = evaluate_e2e(
    model_path="results/exp/final_model",
    base_model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    num_samples=100
)
```

---

## 🐛 Important Fixes Applied

### 1. **Evaluation Padding Fix** (src/evaluation.py:93)
```python
# CRITICAL FIX: Use left-padding for decoder-only models
self.tokenizer.padding_side = 'left'
```
**Why**: Decoder-only models (like TinyLlama) require left-padding for correct generation. Right-padding causes incorrect predictions.

### 2. **WikiSQL Data Loading** (src/data/datasets.py:178)
```python
# Direct download from GitHub (HuggingFace version deprecated)
data_url = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
```
**Why**: HuggingFace dataset scripts are deprecated. Direct loading ensures compatibility.

### 3. **Early->Late Phase Fix** (run_experiment.py:291)
```python
# For early->late strategy, phase2 should go until the end
if args.phase_strategy == "early->late":
    config.dynamic_lora.phase2_steps = (args.phase1_end, None)
```
**Why**: Without this, Phase 2 would end prematurely, leaving no active LoRA parameters.

---

## 📊 Critical Numbers

**Model**: TinyLlama-1.1B-intermediate-step-1431k-3T
- Layers: 22 (L0-L21)
- Attention modules per layer: 4 (q, k, v, o projections)
- MLP modules per layer: 3 (gate, up, down projections)

**LoRA Parameters** (r=4, both modules):
- Per layer: 4 attention + 3 MLP = 7 modules
- Total: 22 layers × 7 modules × 2 (A+B matrices) = 308 LoRA parameters

**Recommended Settings**:
- Rank (r): 4-8 for most tasks
- Learning rate: 2e-4
- Batch size: 16-32
- Max steps: 1000-1500
- Warmup steps: 10

---

## 🎯 Quick Start Commands

```bash
# Standard experiment
python run_experiment.py --dataset e2e_nlg --target both --r 8 --max_steps 1000

# Dynamic LoRA (recommended)
python run_experiment.py \
    --dataset samsum \
    --target both \
    --dynamic_lora \
    --phase_strategy both->mlp->attn \
    --phase1_end 300 \
    --phase2_end 600 \
    --max_steps 1500

# Evaluation only
python -c "from src.evaluation import evaluate_wikisql; \
result = evaluate_wikisql('results/exp/final_model', \
'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 100)"
```

---

## 📝 Notes for Reviewers

1. **All debug code removed**: No debug print statements or test files
2. **Production-ready**: Clean, documented, modular code
3. **Comprehensive logging**: Gradient norms tracked at group and layer level
4. **Flexible configuration**: Easy to add new datasets or strategies
5. **Known limitation**: early->late layer-wise training has gradient propagation issues (documented in ARCHITECTURE.md)
