#!/usr/bin/env python3
"""
Main experiment script for LoRA Module Sensitivity Analysis.

Usage:
    # Single experiment with preset
    python run_experiment.py --preset e2e_attention

    # Single experiment with custom config
    python run_experiment.py --dataset e2e_nlg --target attention_only --r 16 --lr 1e-4

    # Run all presets (batch mode)
    python run_experiment.py --batch

    # Compare results from previous experiments
    python run_experiment.py --compare results/exp1 results/exp2
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from transformers import TrainingArguments, Trainer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import (
    ExperimentConfig,
    DatasetType,
    TargetModuleGroup,
    PRESETS,
    config_to_dict,
)
from src.data.datasets import load_and_preprocess_dataset, get_data_collator
from src.models.lora_setup import setup_model_with_lora, print_trainable_parameters
from src.callbacks.gradient_norm import GradientNormCallback, GradientMeasuringTrainer
from src.callbacks.dynamic_lora import DynamicLoRACallback, create_phased_lora_config, create_layerwise_lora_config
from src.utils.visualization import (
    plot_gradient_norms,
    plot_layer_heatmap,
    plot_layer_comparison,
    plot_gradient_evolution,
    plot_training_loss,
    generate_summary_report,
    load_gradient_log_from_csv,
)
from src.evaluation import evaluate_e2e, evaluate_samsum, evaluate_wikisql, evaluate_multi_nli


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA Module Sensitivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --preset e2e_attention
  python run_experiment.py --dataset samsum --target mlp_only --max_steps 200
  python run_experiment.py --batch
  python run_experiment.py --compare results/e2e_attention results/e2e_mlp
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        help="Use a preset configuration",
    )
    mode_group.add_argument(
        "--batch",
        action="store_true",
        help="Run all preset experiments",
    )
    mode_group.add_argument(
        "--compare",
        nargs="+",
        metavar="PATH",
        help="Compare results from previous experiments",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["e2e_nlg", "samsum", "wikisql", "multi_nli"],
        default="e2e_nlg",
        help="Dataset to use (default: e2e_nlg)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of training samples (default: 500, 0 for full dataset)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )

    # LoRA configuration
    parser.add_argument(
        "--target",
        type=str,
        choices=["attention_only", "mlp_only", "both", "all"],
        default="attention_only",
        help="Target modules for LoRA (default: attention_only). 'all' includes embed_tokens and lm_head",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=4,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)",
    )

    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-device batch size (default: 4)",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum training steps (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Warmup steps (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        help="Base model ID",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )

    # Analysis configuration
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=1,
        help="Gradient norm logging frequency (default: 1)",
    )
    parser.add_argument(
        "--no_layer_norm",
        action="store_true",
        help="Disable layer-level gradient norm measurement",
    )

    # Evaluation configuration
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Disable automatic evaluation after training",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=100,
        help="Number of samples for evaluation (default: 100)",
    )

    # Dynamic LoRA configuration
    parser.add_argument(
        "--dynamic_lora",
        action="store_true",
        help="Enable dynamic LoRA module switching during training",
    )
    parser.add_argument(
        "--phase_strategy",
        type=str,
        default="both->mlp->attn",
        choices=["attn->mlp->both", "both->mlp->attn", "both->mlp->both", "early->late"],
        help="Phase switching strategy. 'early->late': early layers first, then late layers (layer-wise progressive)",
    )
    parser.add_argument(
        "--phase1_end",
        type=int,
        default=100,
        help="End step for phase 1. Default: 100",
    )
    parser.add_argument(
        "--phase2_end",
        type=int,
        default=200,
        help="End step for phase 2. Default: 200. Phase 3 starts after this.",
    )

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build ExperimentConfig from command line arguments."""
    config = ExperimentConfig()

    # Model
    config.model.model_id = args.model

    # Dataset
    config.data.dataset_type = DatasetType(args.dataset)
    config.data.num_samples = args.num_samples
    config.data.max_length = args.max_length

    # LoRA
    config.lora.target_module_group = TargetModuleGroup(args.target)
    config.lora.r = args.r
    config.lora.lora_alpha = args.lora_alpha
    config.lora.lora_dropout = args.lora_dropout

    # Training
    config.training.per_device_train_batch_size = args.batch_size
    config.training.gradient_accumulation_steps = args.grad_accum
    config.training.max_steps = args.max_steps
    config.training.learning_rate = args.lr
    config.training.warmup_steps = args.warmup_steps
    config.training.seed = args.seed

    # Analysis
    config.gradient_analysis.log_frequency = args.log_frequency
    config.gradient_analysis.measure_layer_norm = not args.no_layer_norm

    # Evaluation
    config.evaluation.enabled = not args.no_eval
    config.evaluation.num_samples = args.eval_samples

    # Dynamic LoRA
    config.dynamic_lora.enabled = args.dynamic_lora
    if args.dynamic_lora:
        config.dynamic_lora.phase_strategy = args.phase_strategy
        config.dynamic_lora.phase1_steps = (0, args.phase1_end)
        config.dynamic_lora.phase2_steps = (args.phase1_end, args.phase2_end)
        config.dynamic_lora.phase3_steps = (args.phase2_end, None)

    # Output
    config.output_dir = args.output_dir
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    return config


def run_single_experiment(config: ExperimentConfig) -> dict:
    """
    Run a single experiment with the given configuration.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary with experiment results
    """
    print("\n" + "=" * 70)
    print(f"Starting Experiment: {config.experiment_name}")
    print("=" * 70)

    # Set seed
    torch.manual_seed(config.training.seed)

    # Create output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_to_dict(config), f, indent=2, default=str)
    print(f"Configuration saved to: {config_path}")

    # Validate dynamic LoRA configuration
    if config.dynamic_lora.enabled:
        if config.lora.target_module_group not in [TargetModuleGroup.BOTH, TargetModuleGroup.ALL]:
            print("\n[WARNING] Dynamic LoRA is enabled but target_module_group is not 'both' or 'all'.")
            print(f"Current target: {config.lora.target_module_group.value}")
            print("Switching to 'both' to ensure both attention and mlp modules are available.")
            config.lora.target_module_group = TargetModuleGroup.BOTH

    # Setup model
    print("\n[1/4] Loading model and applying LoRA...")
    model, tokenizer = setup_model_with_lora(config.model, config.lora)
    print_trainable_parameters(model)

    # Load dataset
    print("\n[2/4] Loading and preprocessing dataset...")
    train_dataset = load_and_preprocess_dataset(config.data, tokenizer)
    print(f"Training samples: {len(train_dataset)}")

    # Create validation split
    eval_dataset = None
    if config.data.validation_split > 0:
        from datasets import Dataset

        # Split dataset
        total_size = len(train_dataset)
        eval_size = int(total_size * config.data.validation_split)
        train_size = total_size - eval_size

        # Create splits
        indices = list(range(total_size))
        import random
        random.seed(config.training.seed)
        random.shuffle(indices)

        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        eval_dataset = train_dataset.select(eval_indices)
        train_dataset = train_dataset.select(train_indices)

        print(f"Validation samples: {len(eval_dataset)}")

    # Setup training
    print("\n[3/4] Setting up trainer...")
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy=config.training.save_strategy if eval_dataset else "no",
        save_steps=config.training.save_steps if eval_dataset else None,
        save_total_limit=config.training.save_total_limit if eval_dataset else None,
        load_best_model_at_end=config.training.load_best_model_at_end if eval_dataset else False,
        metric_for_best_model=config.training.metric_for_best_model if eval_dataset else None,
        seed=config.training.seed,
        remove_unused_columns=False,
        report_to="none",
        logging_dir=str(output_dir / "logs"),
    )

    # Create gradient norm callback
    grad_callback = GradientNormCallback(
        config=config.gradient_analysis,
        output_dir=str(output_dir),
        experiment_name="gradient_analysis",
    )

    # Create callbacks list
    callbacks = [grad_callback]

    # Add dynamic LoRA callback if enabled
    if config.dynamic_lora.enabled:
        # Map strategy to phase modules
        strategy = config.dynamic_lora.phase_strategy

        # Layer-wise progressive strategy
        if strategy == "early->late":
            print(f"\n[Dynamic LoRA] Enabled with strategy: {strategy}")
            print(f"  Phase 1 (steps {config.dynamic_lora.phase1_steps[0]}-{config.dynamic_lora.phase1_steps[1]}): Early layers (L0-L10)")
            print(f"  Phase 2 (steps {config.dynamic_lora.phase2_steps[0]}+): Late layers (L11-L21)")

            phase_configs = create_layerwise_lora_config(
                num_layers=22,
                phase1_steps=config.dynamic_lora.phase1_steps,
                phase2_steps=config.dynamic_lora.phase2_steps,
                modules=["attention", "mlp"],
            )
        else:
            # Module-based strategies
            strategy_map = {
                "attn->mlp->both": {
                    "phase1": ["attention"],
                    "phase2": ["mlp"],
                    "phase3": ["attention", "mlp"],
                    "labels": ["Attention only", "MLP only", "Both (Attention + MLP)"],
                },
                "both->mlp->attn": {
                    "phase1": ["attention", "mlp"],
                    "phase2": ["mlp"],
                    "phase3": ["attention"],
                    "labels": ["Both (Attention + MLP)", "MLP only", "Attention only"],
                },
                "both->mlp->both": {
                    "phase1": ["attention", "mlp"],
                    "phase2": ["mlp"],
                    "phase3": ["attention", "mlp"],
                    "labels": ["Both (Attention + MLP)", "MLP only", "Both (Attention + MLP)"],
                },
            }

            if strategy not in strategy_map:
                raise ValueError(f"Unknown phase_strategy: {strategy}")

            strategy_config = strategy_map[strategy]

            print(f"\n[Dynamic LoRA] Enabled with strategy: {strategy}")
            print(f"  Phase 1 (steps {config.dynamic_lora.phase1_steps[0]}-{config.dynamic_lora.phase1_steps[1]}): {strategy_config['labels'][0]}")
            print(f"  Phase 2 (steps {config.dynamic_lora.phase2_steps[0]}-{config.dynamic_lora.phase2_steps[1]}): {strategy_config['labels'][1]}")
            print(f"  Phase 3 (steps {config.dynamic_lora.phase3_steps[0]}+): {strategy_config['labels'][2]}")

            phase_configs = create_phased_lora_config(
                phase1_steps=config.dynamic_lora.phase1_steps,
                phase2_steps=config.dynamic_lora.phase2_steps,
                phase3_steps=config.dynamic_lora.phase3_steps,
                phase1_modules=strategy_config["phase1"],
                phase2_modules=strategy_config["phase2"],
                phase3_modules=strategy_config["phase3"],
            )

        dynamic_lora_callback = DynamicLoRACallback(
            phase_configs=phase_configs,
            verbose=config.dynamic_lora.verbose,
        )
        callbacks.append(dynamic_lora_callback)

    # Create trainer with gradient measurement
    trainer = GradientMeasuringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_data_collator(tokenizer),
        gradient_callback=grad_callback,
        callbacks=callbacks,
    )

    # Train
    print("\n[4/4] Training...")
    train_result = trainer.train()

    # Save final model (best if validation was used, otherwise last)
    final_checkpoint = output_dir / "final_model"
    trainer.save_model(str(final_checkpoint))
    if eval_dataset:
        print(f"Best model (lowest eval_loss) saved to: {final_checkpoint}")
    else:
        print(f"Final model saved to: {final_checkpoint}")

    # Get gradient log
    grad_log = grad_callback.get_log()

    # Generate visualizations
    print("\n[+] Generating visualizations...")

    # Prepare phase transitions for visualization
    phase_transitions = None
    if config.dynamic_lora.enabled:
        strategy = config.dynamic_lora.phase_strategy

        if strategy == "early->late":
            # Layer-wise strategy only has 2 phases
            phase_transitions = [
                (config.dynamic_lora.phase1_steps[1], "Phase 2: Late layers (L11-L21)"),
            ]
        else:
            # Module-based strategies have 3 phases
            strategy_map = {
                "attn->mlp->both": ["Phase 2: MLP only", "Phase 3: Both"],
                "both->mlp->attn": ["Phase 2: MLP only", "Phase 3: Attention only"],
                "both->mlp->both": ["Phase 2: MLP only", "Phase 3: Both"],
            }
            labels = strategy_map.get(strategy, ["Phase 2", "Phase 3"])
            phase_transitions = [
                (config.dynamic_lora.phase1_steps[1], labels[0]),
                (config.dynamic_lora.phase2_steps[1], labels[1]),
            ]

    # Training loss plot
    plot_training_loss(
        trainer.state.log_history,
        title=f"Training Loss - {config.experiment_name}",
        save_path=str(output_dir / "training_loss.png"),
    )

    # Gradient norm plot (log scale)
    plot_gradient_norms(
        grad_log,
        title=f"Gradient Norms (Log Scale) - {config.experiment_name}",
        save_path=str(output_dir / "gradient_norms_log.png"),
        use_log_scale=True,
        phase_transitions=phase_transitions,
    )

    # Gradient norm plot (linear scale)
    plot_gradient_norms(
        grad_log,
        title=f"Gradient Norms (Linear Scale) - {config.experiment_name}",
        save_path=str(output_dir / "gradient_norms.png"),
        use_log_scale=False,
        phase_transitions=phase_transitions,
    )

    # Layer heatmap (if layer data available)
    if grad_log.layer_norms:
        plot_layer_heatmap(
            grad_log,
            title=f"Layer Heatmap - {config.experiment_name}",
            save_path=str(output_dir / "layer_heatmap.png"),
        )

        # Layer comparison (linear scale)
        plot_layer_comparison(
            grad_log,
            title=f"Layer Comparison (Linear Scale) - {config.experiment_name}",
            save_path=str(output_dir / "layer_comparison.png"),
            use_log_scale=False,
        )

        # Layer comparison (log scale)
        plot_layer_comparison(
            grad_log,
            title=f"Layer Comparison (Log Scale) - {config.experiment_name}",
            save_path=str(output_dir / "layer_comparison_log.png"),
            use_log_scale=True,
        )

        # Gradient evolution (log scale)
        plot_gradient_evolution(
            grad_log,
            title=f"Gradient Evolution (Log Scale) - {config.experiment_name}",
            save_path=str(output_dir / "gradient_evolution_log.png"),
            use_log_scale=True,
        )

        # Gradient evolution (linear scale)
        plot_gradient_evolution(
            grad_log,
            title=f"Gradient Evolution (Linear Scale) - {config.experiment_name}",
            save_path=str(output_dir / "gradient_evolution.png"),
            use_log_scale=False,
        )

    # Summary
    summary = grad_log.get_summary_stats()
    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": config.experiment_name,
            "train_loss": train_result.training_loss,
            "gradient_stats": summary,
        }, f, indent=2)

    print(f"\n[+] Results saved to: {output_dir}")
    print("\nGradient Norm Summary:")
    for group, stats in summary.items():
        print(f"  {group}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, peak_step={stats['peak_step']}")

    # Evaluation
    eval_result = None
    if config.evaluation.enabled:
        print("\n[+] Running evaluation...")

        # Cleanup trainer to free memory
        del trainer
        torch.cuda.empty_cache()

        try:
            # Determine which evaluation function to use
            if config.data.dataset_type == DatasetType.E2E_NLG:
                eval_result = evaluate_e2e(
                    model_path=str(final_checkpoint),
                    base_model_id=config.model.model_id,
                    num_samples=config.evaluation.num_samples,
                    output_path=str(output_dir / "evaluation_results.json"),
                )
                print("\n[+] Evaluation Results:")
                print(f"  BLEU:    {eval_result.bleu:.4f}")
                print(f"  ROUGE-1: {eval_result.rouge1:.4f}")
                print(f"  ROUGE-2: {eval_result.rouge2:.4f}")
                print(f"  ROUGE-L: {eval_result.rougeL:.4f}")
                print(f"  METEOR:  {eval_result.meteor:.4f}")
            elif config.data.dataset_type == DatasetType.SAMSUM:
                eval_result = evaluate_samsum(
                    model_path=str(final_checkpoint),
                    base_model_id=config.model.model_id,
                    num_samples=config.evaluation.num_samples,
                    output_path=str(output_dir / "evaluation_results.json"),
                )
                print("\n[+] Evaluation Results:")
                print(f"  BLEU:    {eval_result.bleu:.4f}")
                print(f"  ROUGE-1: {eval_result.rouge1:.4f}")
                print(f"  ROUGE-2: {eval_result.rouge2:.4f}")
                print(f"  ROUGE-L: {eval_result.rougeL:.4f}")
                print(f"  METEOR:  {eval_result.meteor:.4f}")
            elif config.data.dataset_type == DatasetType.WIKISQL:
                eval_result = evaluate_wikisql(
                    model_path=str(final_checkpoint),
                    base_model_id=config.model.model_id,
                    num_samples=config.evaluation.num_samples,
                    output_path=str(output_dir / "evaluation_results.json"),
                    batch_size=config.evaluation.batch_size,
                )
                # WikiSQL results are already printed in the function
            elif config.data.dataset_type == DatasetType.MULTI_NLI:
                eval_result = evaluate_multi_nli(
                    model_path=str(final_checkpoint),
                    base_model_id=config.model.model_id,
                    num_samples=config.evaluation.num_samples,
                    output_path=str(output_dir / "evaluation_results.json"),
                    batch_size=config.evaluation.batch_size,
                )
                # Multi-NLI results are already printed in the function

        except Exception as e:
            print(f"\nWarning: Evaluation failed with error: {e}")
            print("Continuing without evaluation results...")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        "config": config.experiment_name,
        "output_dir": str(output_dir),
        "gradient_log": grad_log,
        "summary": summary,
        "evaluation": eval_result.to_dict() if eval_result else None,
    }


def run_batch_experiments(args: argparse.Namespace) -> dict:
    """Run all preset experiments."""
    print("\n" + "=" * 70)
    print("Running Batch Experiments (All Presets)")
    print("=" * 70)

    all_results = {}
    all_logs = {}

    for preset_name, preset_fn in PRESETS.items():
        print(f"\n{'=' * 70}")
        print(f"Preset: {preset_name}")
        print("=" * 70)

        config = preset_fn()
        config.output_dir = args.output_dir

        # Override with CLI args if provided
        if args.max_steps != 100:
            config.training.max_steps = args.max_steps
        if args.num_samples != 500:
            config.data.num_samples = args.num_samples

        result = run_single_experiment(config)
        all_results[preset_name] = result["summary"]
        all_logs[preset_name] = result["gradient_log"]

    # Generate comparison report
    print("\n" + "=" * 70)
    print("Generating Comparison Report")
    print("=" * 70)

    report_dir = Path(args.output_dir) / "comparison_report"
    summary = generate_summary_report(all_logs, str(report_dir))

    # Print comparison
    print("\nExperiment Comparison:")
    print("-" * 70)
    print(f"{'Experiment':<25} {'Attn Mean':>12} {'MLP Mean':>12} {'Ratio':>10}")
    print("-" * 70)
    for exp_name, data in summary.get("comparison", {}).items():
        print(f"{exp_name:<25} {data['attention_mean']:>12.4f} {data['mlp_mean']:>12.4f} {data['attention_to_mlp_ratio']:>10.4f}")
    print("-" * 70)

    return all_results


def compare_experiments(paths: list) -> None:
    """Compare results from previous experiments."""
    print("\n" + "=" * 70)
    print("Comparing Previous Experiments")
    print("=" * 70)

    logs = {}
    for path in paths:
        path = Path(path)
        exp_name = path.name

        # Try to load from CSV or JSON
        json_path = path / "gradient_analysis.gradient_norms.json"

        if (path / "gradient_analysis.group_norms.csv").exists():
            log = load_gradient_log_from_csv(str(path / "gradient_analysis"))
            logs[exp_name] = log
            print(f"Loaded: {exp_name}")
        elif json_path.exists():
            from src.utils.visualization import load_gradient_log_from_json
            log = load_gradient_log_from_json(str(json_path))
            logs[exp_name] = log
            print(f"Loaded: {exp_name}")
        else:
            print(f"Warning: Could not load results from {path}")

    if len(logs) < 2:
        print("Need at least 2 experiments to compare")
        return

    # Generate comparison
    output_dir = Path(paths[0]).parent / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _ = generate_summary_report(logs, str(output_dir))

    print(f"\nComparison saved to: {output_dir}")


def main():
    """Main entry point."""
    args = parse_args()

    # Handle different modes
    if args.compare:
        compare_experiments(args.compare)
    elif args.batch:
        run_batch_experiments(args)
    elif args.preset:
        config = PRESETS[args.preset]()
        config.output_dir = args.output_dir
        run_single_experiment(config)
    else:
        config = build_config_from_args(args)
        run_single_experiment(config)


if __name__ == "__main__":
    main()
