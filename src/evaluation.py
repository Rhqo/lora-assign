"""
Evaluation utilities for LoRA fine-tuned models.
Supports BLEU, ROUGE, METEOR metrics for generation quality assessment.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from datasets import Dataset

# Evaluation metrics
import evaluate


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    # BLEU scores
    bleu: float = 0.0
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_3: float = 0.0
    bleu_4: float = 0.0

    # ROUGE scores
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    rougeLsum: float = 0.0

    # METEOR score
    meteor: float = 0.0

    # Perplexity (optional)
    perplexity: Optional[float] = None

    # Additional info
    num_samples: int = 0
    generation_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Evaluation results saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "EvaluationResult":
        """Load results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ModelEvaluator:
    """Evaluator for LoRA fine-tuned models."""

    def __init__(
        self,
        model_path: str,
        base_model_id: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to LoRA adapter or full model
            base_model_id: Base model ID (required for LoRA adapters)
            device: Device to use
            torch_dtype: Model dtype
        """
        self.device = device
        self.torch_dtype = torch_dtype

        # Load tokenizer
        tokenizer_path = model_path if base_model_id is None else base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if base_model_id is not None:
            # Load as LoRA adapter
            print(f"Loading base model: {base_model_id}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            print(f"Loading LoRA adapter: {model_path}")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load as full model
            print(f"Loading model: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
            )

        self.model.eval()

        # Load metrics
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        self.meteor_metric = evaluate.load("meteor")

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        batch_size: int = 8,
    ) -> List[str]:
        """
        Generate text from prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            batch_size: Batch size for generation

        Returns:
            List of generated texts
        """
        generations = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only new tokens
            for j, output in enumerate(outputs):
                input_length = inputs.input_ids[j].shape[0]
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                generations.append(generated_text.strip())

        return generations

    def compute_bleu(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Compute BLEU scores.

        Args:
            predictions: Generated texts
            references: Reference texts (list of lists for multiple refs)

        Returns:
            Dictionary with BLEU scores
        """
        # Tokenize for BLEU computation
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split() for ref in refs] for refs in references]

        result = self.bleu_metric.compute(
            predictions=pred_tokens,
            references=ref_tokens,
        )

        return {
            "bleu": result["bleu"],
            "bleu_1": result["precisions"][0] if result["precisions"] else 0.0,
            "bleu_2": result["precisions"][1] if len(result["precisions"]) > 1 else 0.0,
            "bleu_3": result["precisions"][2] if len(result["precisions"]) > 2 else 0.0,
            "bleu_4": result["precisions"][3] if len(result["precisions"]) > 3 else 0.0,
        }

    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            Dictionary with ROUGE scores
        """
        result = self.rouge_metric.compute(
            predictions=predictions,
            references=references,
        )

        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
            "rougeLsum": result["rougeLsum"],
        }

    def compute_meteor(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute METEOR score.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            Dictionary with METEOR score
        """
        result = self.meteor_metric.compute(
            predictions=predictions,
            references=references,
        )

        return {"meteor": result["meteor"]}

    def compute_perplexity(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> float:
        """
        Compute perplexity on given texts.

        Args:
            texts: List of texts to evaluate
            batch_size: Batch size

        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_tokens = 0

        for i in tqdm(range(0, len(texts), batch_size), desc="Computing PPL"):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss

            # Count non-padding tokens
            num_tokens = (inputs.input_ids != self.tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return perplexity

    def evaluate(
        self,
        prompts: List[str],
        references: List[str],
        compute_perplexity: bool = False,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Run full evaluation.

        Args:
            prompts: Input prompts
            references: Reference outputs
            compute_perplexity: Whether to compute perplexity
            generation_config: Generation parameters

        Returns:
            EvaluationResult with all metrics
        """
        gen_config = generation_config or {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }

        # Generate predictions
        predictions = self.generate(prompts, **gen_config)

        # Compute metrics
        # For BLEU, wrap references in list (single reference per sample)
        bleu_refs = [[ref] for ref in references]
        bleu_scores = self.compute_bleu(predictions, bleu_refs)
        rouge_scores = self.compute_rouge(predictions, references)
        meteor_scores = self.compute_meteor(predictions, references)

        # Optional: Compute perplexity
        ppl = None
        if compute_perplexity:
            full_texts = [p + " " + r for p, r in zip(prompts, references)]
            ppl = self.compute_perplexity(full_texts)

        result = EvaluationResult(
            bleu=bleu_scores["bleu"],
            bleu_1=bleu_scores["bleu_1"],
            bleu_2=bleu_scores["bleu_2"],
            bleu_3=bleu_scores["bleu_3"],
            bleu_4=bleu_scores["bleu_4"],
            rouge1=rouge_scores["rouge1"],
            rouge2=rouge_scores["rouge2"],
            rougeL=rouge_scores["rougeL"],
            rougeLsum=rouge_scores["rougeLsum"],
            meteor=meteor_scores["meteor"],
            perplexity=ppl,
            num_samples=len(prompts),
            generation_config=gen_config,
        )

        return result


def evaluate_e2e(
    model_path: str,
    base_model_id: str,
    num_samples: int = 100,
    output_path: Optional[str] = None,
) -> EvaluationResult:
    """
    Evaluate on E2E NLG dataset.

    Args:
        model_path: Path to LoRA adapter
        base_model_id: Base model ID
        num_samples: Number of samples to evaluate
        output_path: Path to save results

    Returns:
        EvaluationResult
    """
    from datasets import load_dataset

    # Load test set (use trust_remote_code for legacy datasets)
    dataset = load_dataset("tuetschek/e2e_nlg", split="test")

    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Prepare prompts and references
    prompts = []
    references = []

    for sample in dataset:
        # E2E format: meaning representation -> text
        mr = sample["meaning_representation"]
        ref = sample["human_reference"]
        prompt = f"Generate a restaurant description from the following attributes:\n{mr}\n\nDescription:"
        prompts.append(prompt)
        references.append(ref)

    # Evaluate
    evaluator = ModelEvaluator(model_path, base_model_id)
    result = evaluator.evaluate(prompts, references)

    if output_path:
        result.save(output_path)

    return result


def evaluate_samsum(
    model_path: str,
    base_model_id: str,
    num_samples: int = 100,
    output_path: Optional[str] = None,
) -> EvaluationResult:
    """
    Evaluate on SAMSum dataset.

    Args:
        model_path: Path to LoRA adapter
        base_model_id: Base model ID
        num_samples: Number of samples to evaluate
        output_path: Path to save results

    Returns:
        EvaluationResult
    """
    from datasets import load_dataset

    # Load test set
    try:
        dataset = load_dataset("knkarthick/samsum", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load knkarthick/samsum: {e}")
        print("Trying alternative dataset name...")
        dataset = load_dataset("Samsung/samsum", split="test")

    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Prepare prompts and references
    prompts = []
    references = []

    for sample in dataset:
        dialogue = sample["dialogue"]
        summary = sample["summary"]
        prompt = f"Summarize the following dialogue:\n\n{dialogue}\n\nSummary:"
        prompts.append(prompt)
        references.append(summary)

    # Evaluate
    evaluator = ModelEvaluator(model_path, base_model_id)
    result = evaluator.evaluate(prompts, references)

    if output_path:
        result.save(output_path)

    return result


def compare_models(
    model_paths: Dict[str, str],
    base_model_id: str,
    dataset_type: str = "e2e_nlg",
    num_samples: int = 100,
    output_dir: Optional[str] = None,
) -> Dict[str, EvaluationResult]:
    """
    Compare multiple LoRA models.

    Args:
        model_paths: Dict of {name: path} for models to compare
        base_model_id: Base model ID
        dataset_type: "e2e_nlg" or "samsum"
        num_samples: Number of samples per evaluation
        output_dir: Directory to save results

    Returns:
        Dict of {name: EvaluationResult}
    """
    results = {}

    eval_func = evaluate_e2e if dataset_type == "e2e_nlg" else evaluate_samsum

    for name, path in model_paths.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")

        output_path = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(output_dir) / f"{name}_eval.json")

        result = eval_func(path, base_model_id, num_samples, output_path)
        results[name] = result

        # Print summary
        print(f"\n{name} Results:")
        print(f"  BLEU: {result.bleu:.4f}")
        print(f"  ROUGE-1: {result.rouge1:.4f}")
        print(f"  ROUGE-2: {result.rouge2:.4f}")
        print(f"  ROUGE-L: {result.rougeL:.4f}")
        print(f"  METEOR: {result.meteor:.4f}")

    # Save comparison summary
    if output_dir:
        summary = {name: res.to_dict() for name, res in results.items()}
        summary_path = Path(output_dir) / "comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nComparison summary saved to: {summary_path}")

    return results


def print_comparison_table(results: Dict[str, EvaluationResult]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("Model Comparison Results")
    print("=" * 80)

    # Header
    print(f"{'Model':<25} {'BLEU':>8} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'METEOR':>8}")
    print("-" * 80)

    # Data rows
    for name, result in results.items():
        print(f"{name:<25} {result.bleu:>8.4f} {result.rouge1:>8.4f} {result.rouge2:>8.4f} {result.rougeL:>8.4f} {result.meteor:>8.4f}")

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned models")
    parser.add_argument("--model", "-m", required=True, help="Path to LoRA adapter")
    parser.add_argument("--base", "-b", required=True, help="Base model ID")
    parser.add_argument("--dataset", "-d", default="e2e_nlg", choices=["e2e_nlg", "samsum"])
    parser.add_argument("--samples", "-n", type=int, default=100, help="Number of samples")
    parser.add_argument("--output", "-o", help="Output path for results JSON")

    args = parser.parse_args()

    if args.dataset == "e2e_nlg":
        result = evaluate_e2e(args.model, args.base, args.samples, args.output)
    else:
        result = evaluate_samsum(args.model, args.base, args.samples, args.output)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"BLEU:    {result.bleu:.4f}")
    print(f"BLEU-1:  {result.bleu_1:.4f}")
    print(f"BLEU-2:  {result.bleu_2:.4f}")
    print(f"BLEU-3:  {result.bleu_3:.4f}")
    print(f"BLEU-4:  {result.bleu_4:.4f}")
    print(f"ROUGE-1: {result.rouge1:.4f}")
    print(f"ROUGE-2: {result.rouge2:.4f}")
    print(f"ROUGE-L: {result.rougeL:.4f}")
    print(f"METEOR:  {result.meteor:.4f}")
