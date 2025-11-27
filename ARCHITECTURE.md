# LoRA Module Sensitivity Analysis - Architecture Documentation

## 📂 Project Structure

```
lora/
├── configs/
│   └── config.py                    # Configuration management
├── src/
│   ├── callbacks/
│   │   ├── dynamic_lora.py         # Dynamic LoRA module switching
│   │   └── gradient_norm.py        # Gradient norm measurement
│   ├── data/
│   │   └── datasets.py             # Dataset loading and preprocessing
│   ├── models/
│   │   └── lora_setup.py           # LoRA model initialization
│   ├── utils/
│   │   └── visualization.py        # Visualization utilities
│   └── evaluation.py               # Model evaluation
├── run_experiment.py               # Main experiment runner
└── GUIDE.md                        # User guide

```

---

## 🎯 Core Components

### 1. **Configuration System** (`configs/config.py`)

**Purpose**: Centralized configuration management using dataclasses

**Key Classes**:
- `ModelConfig`: Base model settings (model_id, dtype, device_map)
- `LoRAConfig`: LoRA hyperparameters (rank, alpha, dropout, target modules)
- `TrainingConfig`: Training hyperparameters (learning rate, batch size, etc.)
- `DataConfig`: Dataset selection and preprocessing options
- `GradientAnalysisConfig`: Gradient measurement settings
- `EvaluationConfig`: Post-training evaluation settings
- `DynamicLoRAConfig`: Dynamic module switching configuration
- `ExperimentConfig`: Complete experiment configuration

**Target Module Groups**:
- `ATTENTION_ONLY`: q_proj, v_proj
- `MLP_ONLY`: gate_proj, up_proj, down_proj
- `BOTH`: All attention and MLP modules
- `ALL`: Includes embed_tokens and lm_head

---

### 2. **Dynamic LoRA Callback** (`src/callbacks/dynamic_lora.py`)

**Purpose**: Enable/disable specific LoRA modules during training based on predefined phases

**Key Features**:
- **Module-based strategies**: Switch between attention/MLP modules
  - `attn->mlp->both`: Attention → MLP → Both
  - `both->mlp->attn`: Both → MLP → Attention (recommended)
  - `both->mlp->both`: Both → MLP → Both

- **Layer-wise strategies**: Progressive layer training
  - `early->late`: Train early layers (L0-L10) then late layers (L11-L21)

**Implementation**:
```python
class DynamicLoRACallback(TrainerCallback):
    def on_step_begin(self, ...):
        # Check if phase transition is needed
        # Update requires_grad for LoRA parameters

    def _update_lora_modules(self, model, active_modules, layer_range):
        # Enable/disable modules based on:
        # 1. Module type (attention/mlp)
        # 2. Layer range (for layer-wise training)
```

**Phase Configuration**:
```python
phase_configs = [
    {
        "start_step": 0,
        "end_step": 100,
        "active_modules": ["attention", "mlp"],
        "layer_range": (0, 10)  # Optional: for layer-wise training
    },
    ...
]
```

---

### 3. **Gradient Norm Measurement** (`src/callbacks/gradient_norm.py`)

**Purpose**: Track gradient norms during training for sensitivity analysis

**Measurement Levels**:
1. **Group-level**: Total norm per module type
   - Attention modules (q/k/v/o_proj)
   - MLP modules (gate/up/down_proj)
   - Embedding (embed_tokens)
   - LM head (lm_head)

2. **Layer-level**: Norm per individual layer
   - `layer_0_attention`, `layer_0_mlp`
   - `layer_1_attention`, `layer_1_mlp`
   - ... (for all 22 layers in TinyLlama)

**Key Classes**:
- `GradientMeasuringTrainer`: Custom Trainer that measures gradients after backward pass
- `GradientNormCallback`: Stores and saves gradient measurements
- `GradientNormLog`: Container for measurements

**Output Files**:
- `gradient_analysis.group_norms.csv`: Group-level norms per step
- `gradient_analysis.layer_norms.csv`: Layer-level norms per step
- `gradient_analysis.summary.json`: Statistical summary

---

### 4. **Dataset Support** (`src/data/datasets.py`)

**Purpose**: Load and preprocess datasets for LoRA fine-tuning

**Supported Datasets**:

| Dataset | Task | Samples | Format |
|---------|------|---------|--------|
| **E2E NLG** | Text Generation | 42,061 train | MR → Description |
| **SAMSum** | Summarization | 14,732 train | Dialogue → Summary |
| **WikiSQL** | SQL Generation | 56,355 train | Question + Table → SQL |
| **Multi-NLI** | Classification | 392,702 train | Premise + Hypothesis → Label |

**Preprocessing**:
- Prompt formatting with clear instruction/input/output structure
- Tokenization with padding and truncation
- Label alignment for causal LM training

**Example Prompt** (WikiSQL):
```
### Instruction: Generate SQL from question and table.

### Table:
name, area, country

### Question:
What is the area of France?

### SQL:
SELECT area FROM table WHERE country = France
```

---

### 5. **Model Evaluation** (`src/evaluation.py`)

**Purpose**: Comprehensive evaluation of fine-tuned models

**Evaluation Metrics**:

| Dataset | Metrics | Description |
|---------|---------|-------------|
| E2E NLG | BLEU, ROUGE, METEOR | Text quality |
| SAMSum | BLEU, ROUGE, METEOR | Summary quality |
| WikiSQL | Exact Match | SQL correctness |
| Multi-NLI | Accuracy | Classification accuracy |

**Key Features**:
- **Batched generation** for efficiency
- **Left-padding** for decoder-only models (critical fix)
- **Flexible sampling** (temperature, top_p, do_sample)
- **Results persistence** (JSON format)

**ModelEvaluator Class**:
```python
evaluator = ModelEvaluator(
    model_path="results/experiment/final_model",
    base_model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)

predictions = evaluator.generate(
    prompts,
    max_new_tokens=128,
    temperature=0.7,
    batch_size=8
)
```

---

### 6. **Visualization** (`src/utils/visualization.py`)

**Purpose**: Generate visualizations for gradient analysis

**Generated Plots**:
1. **Gradient Norms** (`gradient_norms.png`)
   - Line plot of group-level gradient norms over training
   - Shows attention vs MLP gradient evolution

2. **Layer Heatmap** (`layer_heatmap.png`)
   - Heatmap of layer-wise gradient norms
   - Reveals which layers learn most actively

3. **Layer Comparison** (`layer_comparison.png`)
   - Bar chart comparing early vs late layers
   - Shows layer-wise learning patterns

4. **Gradient Evolution** (`gradient_evolution.png`)
   - Detailed evolution of individual layer gradients
   - Useful for understanding dynamic LoRA effects

5. **Training Loss** (`training_loss.png`)
   - Training and validation loss curves

---

### 7. **Main Experiment Runner** (`run_experiment.py`)

**Purpose**: Orchestrate complete experiments from training to evaluation

**Workflow**:
```
1. Parse CLI arguments / Load preset configuration
2. Setup model with LoRA adapters
3. Load and preprocess dataset
4. Create train/validation split
5. Setup callbacks (gradient measurement, dynamic LoRA)
6. Train model with HuggingFace Trainer
7. Save best model (based on validation loss)
8. Generate visualizations
9. Run evaluation (if enabled)
10. Save comprehensive results
```

**CLI Usage**:
```bash
# Quick preset
python run_experiment.py --preset e2e_attention

# Custom configuration
python run_experiment.py \
    --dataset wikisql \
    --target both \
    --r 4 \
    --batch_size 32 \
    --max_steps 1000 \
    --lr 2e-4

# Dynamic LoRA
python run_experiment.py \
    --dataset wikisql \
    --target both \
    --dynamic_lora \
    --phase_strategy both->mlp->attn \
    --phase1_end 300 \
    --phase2_end 600 \
    --max_steps 1500
```

---

## 🔬 Experiment Outputs

Each experiment creates a directory: `results/{experiment_name}/`

**Directory Contents**:
```
results/experiment_name/
├── config.json                          # Full experiment configuration
├── final_model/                         # Best LoRA adapter (PEFT format)
├── checkpoints/                         # Training checkpoints
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── ...
├── gradient_analysis.group_norms.csv    # Group-level gradient data
├── gradient_analysis.layer_norms.csv    # Layer-level gradient data
├── gradient_analysis.summary.json       # Statistical summary
├── gradient_norms.png                   # Gradient norm plot
├── layer_heatmap.png                    # Layer-wise heatmap
├── layer_comparison.png                 # Layer comparison
├── gradient_evolution.png               # Detailed evolution
├── training_loss.png                    # Loss curves
├── results_summary.json                 # Training metrics
└── {dataset}_evaluation.json            # Evaluation results (if enabled)
```

---

## 🚀 Key Design Decisions

### 1. **Modular Architecture**
- Clear separation of concerns (data, models, callbacks, visualization)
- Easy to extend with new datasets or evaluation metrics

### 2. **Configuration-Driven**
- All hyperparameters in centralized config
- Supports both presets and CLI overrides
- Reproducible experiments

### 3. **Comprehensive Logging**
- Group-level AND layer-level gradient tracking
- CSV format for easy analysis in external tools
- Automatic visualization generation

### 4. **Dynamic LoRA Support**
- Flexible phase configuration
- Both module-based and layer-wise strategies
- Real-time parameter switching during training

### 5. **Production-Ready Evaluation**
- Batched generation for efficiency
- Proper padding handling (left-padding for decoder models)
- Multiple metrics for comprehensive assessment

---

## 🐛 Known Issues & Limitations

### 1. **Early->Late Strategy (Layer-wise Training)**
**Issue**: Gradients stop propagating after phase transition
**Root Cause**: PyTorch optimizer only tracks parameters with `requires_grad=True` at initialization. Changing `requires_grad` mid-training doesn't update optimizer's param_groups.
**Status**: Not recommended for production use
**Workaround**: Use module-based strategies (both->mlp->attn) instead

### 2. **Validation Sample Count**
**Behavior**: Training validation uses `num_samples * validation_split`, while post-training evaluation uses `EvaluationConfig.num_samples`
**Impact**: User may see different sample counts
**Solution**: Clear documentation in config comments

### 3. **WikiSQL Exact Match**
**Issue**: Very low scores (2-5%) even after training
**Reason**: SQL generation is extremely difficult for small models; requires exact syntax
**Recommendation**: Use as research benchmark, not production metric

---

## 📊 Recommended Experiment Configurations

### **1. Standard LoRA Comparison**
Compare attention-only vs MLP-only vs both:

```bash
# Attention only
python run_experiment.py --dataset e2e_nlg --target attention_only --r 8 --max_steps 1000

# MLP only
python run_experiment.py --dataset e2e_nlg --target mlp_only --r 8 --max_steps 1000

# Both
python run_experiment.py --dataset e2e_nlg --target both --r 8 --max_steps 1000
```

### **2. Dynamic LoRA Sensitivity** (Recommended)
Test if progressive specialization improves results:

```bash
python run_experiment.py \
    --dataset samsum \
    --target both \
    --dynamic_lora \
    --phase_strategy both->mlp->attn \
    --phase1_end 300 \
    --phase2_end 600 \
    --max_steps 1500 \
    --r 4 \
    --batch_size 32
```

### **3. Rank Comparison**
Find optimal LoRA rank:

```bash
for rank in 4 8 16 32; do
    python run_experiment.py --dataset wikisql --target both --r $rank --max_steps 1000
done
```

---

## 🔧 Extension Guide

### Adding a New Dataset

1. **Update `configs/config.py`**:
```python
class DatasetType(str, Enum):
    YOUR_DATASET = "your_dataset"
```

2. **Implement loader in `src/data/datasets.py`**:
```python
def _load_your_dataset(num_samples: int) -> Dataset:
    dataset = load_dataset("org/dataset", split="train")
    # Preprocess...
    return dataset
```

3. **Add evaluation in `src/evaluation.py`**:
```python
def evaluate_your_dataset(model_path, base_model_id, num_samples, output_path):
    # Load test data
    # Generate predictions
    # Compute metrics
    return results
```

---

## 📝 Citation

If you use this codebase, please cite:

```
@software{lora_sensitivity_analysis,
  title={LoRA Module Sensitivity Analysis Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/lora}
}
```

---

## 📞 Support

For questions or issues:
- Check `GUIDE.md` for usage instructions
- Review this architecture document for implementation details
- Open an issue on GitHub
