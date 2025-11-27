# LoRA Module Sensitivity Analysis

A comprehensive framework for analyzing LoRA (Low-Rank Adaptation) module sensitivity in fine-tuning large language models.

## 🚀 Features

- **Dynamic LoRA Module Switching**: Enable/disable specific LoRA modules during training
- **Comprehensive Gradient Analysis**: Track gradient norms at both group and layer levels
- **Multi-Dataset Support**: E2E NLG, SAMSum, WikiSQL, Multi-NLI
- **Flexible Configuration**: Dataclass-based config system with CLI overrides
- **Automatic Evaluation**: Built-in evaluation with multiple metrics (BLEU, ROUGE, METEOR, Exact Match, Accuracy)
- **Rich Visualizations**: Automatic generation of gradient heatmaps, evolution plots, and layer comparisons

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/lora.git
cd lora

# Create virtual environment
uv venv -p 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic LoRA Fine-tuning

```bash
# Train with default settings (E2E NLG, attention-only)
python run_experiment.py --preset e2e_attention

# Custom configuration
python run_experiment.py \
    --dataset wikisql \
    --target both \
    --r 8 \
    --batch_size 32 \
    --max_steps 1000 \
    --lr 2e-4
```

### Dynamic LoRA (Recommended)

```bash
# Progressive module specialization: Both → MLP → Attention
python run_experiment.py \
    --dataset samsum \
    --target both \
    --dynamic_lora \
    --phase_strategy both->mlp->attn \
    --phase1_end 300 \
    --phase2_end 600 \
    --max_steps 1500
```

### Evaluation Only

```bash
python -c "
from src.evaluation import evaluate_wikisql

result = evaluate_wikisql(
    model_path='results/experiment/final_model',
    base_model_id='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
    num_samples=100,
    batch_size=8
)
print(f'Exact Match: {result[\"exact_match\"]:.4f}')
"
```

## 📊 Supported Datasets

| Dataset | Task | Samples | Metrics |
|---------|------|---------|---------|
| **E2E NLG** | Text Generation | 42,061 | BLEU, ROUGE, METEOR |
| **SAMSum** | Summarization | 14,732 | BLEU, ROUGE, METEOR |
| **WikiSQL** | SQL Generation | 56,355 | Exact Match |
| **Multi-NLI** | Classification | 392,702 | Accuracy |

## 🔬 Dynamic LoRA Strategies

### Module-based Strategies

1. **both->mlp->attn** (Recommended)
   - Phase 1: Train both attention and MLP
   - Phase 2: Train MLP only
   - Phase 3: Train attention only

2. **attn->mlp->both**
   - Phase 1: Train attention only
   - Phase 2: Train MLP only
   - Phase 3: Train both

3. **both->mlp->both**
   - Phase 1: Train both
   - Phase 2: Train MLP only
   - Phase 3: Train both again

### Layer-wise Strategy (Experimental)

- **early->late**: Train early layers (L0-L10) then late layers (L11-L21)
  - ⚠️ Known issue: Gradient propagation problems after phase transition
  - Not recommended for production use

## 📁 Project Structure

```
lora/
├── configs/
│   └── config.py                 # Configuration system
├── src/
│   ├── callbacks/
│   │   ├── dynamic_lora.py      # Dynamic LoRA switching
│   │   └── gradient_norm.py     # Gradient measurement
│   ├── data/
│   │   └── datasets.py          # Dataset loading
│   ├── models/
│   │   └── lora_setup.py        # LoRA initialization
│   ├── utils/
│   │   └── visualization.py     # Plotting utilities
│   └── evaluation.py            # Model evaluation
├── run_experiment.py            # Main script
├── GUIDE.md                     # User guide
├── ARCHITECTURE.md              # Architecture documentation
└── CODE_SUMMARY.md              # Code reference
```

## 📈 Experiment Outputs

Each experiment creates:

```
results/{experiment_name}/
├── config.json                       # Full configuration
├── final_model/                      # Best LoRA adapter
├── gradient_analysis.group_norms.csv # Group-level gradients
├── gradient_analysis.layer_norms.csv # Layer-level gradients
├── gradient_norms.png                # Visualization
├── layer_heatmap.png                 # Layer heatmap
├── layer_comparison.png              # Layer comparison
├── gradient_evolution.png            # Evolution plot
├── training_loss.png                 # Loss curves
└── {dataset}_evaluation.json         # Evaluation results
```

## 📚 Documentation

- **[GUIDE.md](GUIDE.md)**: Detailed user guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and design decisions
- **[CODE_SUMMARY.md](CODE_SUMMARY.md)**: Quick code reference

## 🐛 Known Issues

1. **Early->Late Layer-wise Strategy**: Gradient propagation issues after phase transition
   - **Workaround**: Use module-based strategies instead
   - **Details**: See ARCHITECTURE.md

2. **WikiSQL Low Scores**: Exact match scores typically 2-5%
   - **Reason**: SQL generation is extremely difficult for small models
   - **Recommendation**: Use as research benchmark, not production metric

## 📄 License

MIT License

## 📞 Contact

For questions or issues, check documentation files (GUIDE.md, ARCHITECTURE.md) or open an issue.

---

**Model**: TinyLlama-1.1B-intermediate-step-1431k-3T (22 layers, 1.1B parameters)

**Hardware**: GPU with 16GB+ VRAM recommended
