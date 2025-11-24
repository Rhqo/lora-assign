"""
Dataset loading and preprocessing for E2E NLG and SAMSum.
"""
from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
import sys
sys.path.append(".")

from configs.config import DataConfig, DatasetType


def load_and_preprocess_dataset(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """
    Load and preprocess dataset based on configuration.

    Args:
        config: Data configuration
        tokenizer: Tokenizer for text encoding

    Returns:
        Preprocessed dataset ready for training
    """
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if config.dataset_type == DatasetType.E2E_NLG:
        dataset = _load_e2e_nlg(config.num_samples)
        preprocess_fn = _get_e2e_preprocess_fn(tokenizer, config.max_length)
    elif config.dataset_type == DatasetType.SAMSUM:
        dataset = _load_samsum(config.num_samples)
        preprocess_fn = _get_samsum_preprocess_fn(tokenizer, config.max_length)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")

    # Preprocess
    tokenized_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {config.dataset_type.value}",
    )

    return tokenized_dataset


def _load_e2e_nlg(num_samples: int) -> Dataset:
    """
    Load E2E NLG dataset directly from CSV.
    Source: https://github.com/tuetschek/e2e-dataset
    """
    csv_url = "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/trainset.csv"
    dataset = load_dataset("csv", data_files=csv_url, split="train")

    # Rename columns to match expected format
    dataset = dataset.rename_columns({"mr": "meaning_representation", "ref": "human_reference"})

    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset


def _load_samsum(num_samples: int) -> Dataset:
    """Load SAMSum dataset."""
    if num_samples > 0:
        dataset = load_dataset("knkarthick/samsum", split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset("knkarthick/samsum", split="train")
    return dataset


def _get_e2e_preprocess_fn(tokenizer: PreTrainedTokenizer, max_length: int):
    """
    Get preprocessing function for E2E NLG dataset (tuetschek/e2e_nlg).

    E2E NLG task: Convert structured meaning representation to natural language.
    Input format: "Table: name[The Eagle], eatType[coffee shop], ..."
    Output format: Natural language description

    Dataset columns: meaning_representation, human_reference
    """
    def preprocess(examples: Dict[str, Any]) -> Dict[str, Any]:
        # Create instruction-style prompts
        prompts = []
        for mr, ref in zip(examples["meaning_representation"], examples["human_reference"]):
            # Format: instruction with input/output
            prompt = (
                f"### Instruction: Generate a natural language description from the following structured data.\n\n"
                f"### Input:\n{mr}\n\n"
                f"### Output:\n{ref}"
            )
            prompts.append(prompt)

        # Tokenize
        model_inputs = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # For causal LM, labels are same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    return preprocess


def _get_samsum_preprocess_fn(tokenizer: PreTrainedTokenizer, max_length: int):
    """
    Get preprocessing function for SAMSum dataset.

    SAMSum task: Summarize dialogue conversations.
    Input format: Multi-turn dialogue
    Output format: Summary of the dialogue
    """
    def preprocess(examples: Dict[str, Any]) -> Dict[str, Any]:
        prompts = []
        for dialogue, summary in zip(examples["dialogue"], examples["summary"]):
            # Format: instruction with dialogue and summary
            prompt = (
                f"### Instruction: Summarize the following dialogue.\n\n"
                f"### Dialogue:\n{dialogue}\n\n"
                f"### Summary:\n{summary}"
            )
            prompts.append(prompt)

        # Tokenize
        model_inputs = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # For causal LM, labels are same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        return model_inputs

    return preprocess


def get_data_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorForLanguageModeling:
    """
    Get data collator for language modeling.

    Args:
        tokenizer: Tokenizer

    Returns:
        Data collator instance
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )


def get_validation_dataset(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Optional[Dataset]:
    """
    Load validation dataset if available.

    Args:
        config: Data configuration
        tokenizer: Tokenizer

    Returns:
        Validation dataset or None
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_samples = max(50, int(config.num_samples * config.validation_split))

    if config.dataset_type == DatasetType.E2E_NLG:
        # Load validation set from CSV
        csv_url = "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/devset.csv"
        dataset = load_dataset("csv", data_files=csv_url, split="train")
        dataset = dataset.rename_columns({"mr": "meaning_representation", "ref": "human_reference"})
        dataset = dataset.select(range(min(val_samples, len(dataset))))
        preprocess_fn = _get_e2e_preprocess_fn(tokenizer, config.max_length)
    elif config.dataset_type == DatasetType.SAMSUM:
        dataset = load_dataset("knkarthick/samsum", split=f"validation[:{val_samples}]")
        preprocess_fn = _get_samsum_preprocess_fn(tokenizer, config.max_length)
    else:
        return None

    tokenized_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing validation set",
    )

    return tokenized_dataset
