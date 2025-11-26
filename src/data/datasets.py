"""
Dataset loading and preprocessing for E2E NLG and SAMSum.
"""
from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer, default_data_collator
import sys
sys.path.append(".")

from configs.config import DataConfig, DatasetType, TaskType


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
    elif config.dataset_type == DatasetType.WIKISQL:
        dataset = _load_wikisql(config.num_samples)
        preprocess_fn = _get_wikisql_preprocess_fn(tokenizer, config.max_length)
    elif config.dataset_type == DatasetType.MULTI_NLI:
        dataset = _load_multi_nli(config.num_samples)
        preprocess_fn = _get_multinli_preprocess_fn(tokenizer, config.max_length)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")

    # Preprocess
    tokenized_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {config.dataset_type.value}",
        load_from_cache_file=False,  # IMPORTANT: Force re-process to apply padding mask fix
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


def _load_wikisql(num_samples: int) -> Dataset:
    """
    Load WikiSQL dataset.
    Task: Convert natural language question + table to SQL query.

    Note: We load from the raw data files since the dataset script is deprecated.
    """
    import json
    import tarfile
    from pathlib import Path
    import tempfile
    import urllib.request

    # Download and extract WikiSQL data
    data_url = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        archive_path = tmpdir_path / "data.tar.bz2"

        # Download
        print(f"Downloading WikiSQL data...")
        urllib.request.urlretrieve(data_url, archive_path)

        # Extract
        print(f"Extracting WikiSQL data...")
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(tmpdir_path)

        data_dir = tmpdir_path / "data"

        # Load training data
        train_file = data_dir / "train.jsonl"
        tables_file = data_dir / "train.tables.jsonl"

        # Load tables
        tables = {}
        with open(tables_file, 'r', encoding='utf-8') as f:
            for line in f:
                table = json.loads(line)
                tables[table['id']] = table

        # Load training examples
        examples = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                # Add table information
                example['table'] = tables[example['table_id']]

                # Handle missing fields
                example['table']['page_title'] = example['table'].get('page_title', '')
                example['table']['section_title'] = example['table'].get('section_title', '')
                example['table']['caption'] = example['table'].get('caption', '')
                example['table']['name'] = example['table'].get('name', '')
                example['table']['page_id'] = str(example['table'].get('page_id', ''))

                # Convert rows to strings
                example['table']['rows'] = [[str(cell) for cell in row] for row in example['table']['rows']]

                # Restructure conds to dict format
                conds = []
                for cond in example['sql']['conds']:
                    conds.append({
                        'column_index': cond[0],
                        'operator_index': cond[1],
                        'condition': str(cond[2])
                    })
                example['sql']['conds'] = conds

                examples.append(example)

                if num_samples > 0 and len(examples) >= num_samples:
                    break

        # Convert to dataset
        dataset = Dataset.from_list(examples)

    return dataset


def _load_multi_nli(num_samples: int) -> Dataset:
    """
    Load Multi-NLI dataset from NYU.
    Task: 3-way classification (entailment/neutral/contradiction).
    """
    dataset = load_dataset("nyu-mll/multi_nli", split="train")
    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
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
        # but we need to mask padding tokens (-100 = ignore in loss)
        import copy
        labels = copy.deepcopy(model_inputs["input_ids"])

        # Replace padding token id with -100 to ignore in loss calculation
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] == tokenizer.pad_token_id:
                    labels[i][j] = -100

        model_inputs["labels"] = labels

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
        # but we need to mask padding tokens (-100 = ignore in loss)
        import copy
        labels = copy.deepcopy(model_inputs["input_ids"])

        # Replace padding token id with -100 to ignore in loss calculation
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] == tokenizer.pad_token_id:
                    labels[i][j] = -100

        model_inputs["labels"] = labels

        return model_inputs

    return preprocess


def _get_wikisql_preprocess_fn(tokenizer: PreTrainedTokenizer, max_length: int):
    """
    Get preprocessing function for WikiSQL dataset.

    WikiSQL task: Convert natural language question + table schema → SQL query.
    Format: Instruction + Table headers + Question → SQL
    """
    def preprocess(examples: Dict[str, Any]) -> Dict[str, Any]:
        prompts = []

        for question, table, sql in zip(
            examples["question"],
            examples["table"],
            examples["sql"]
        ):
            # Extract table headers
            table_headers = ", ".join(table["header"])

            # Convert SQL dict to string
            # sql structure: {sel: int, agg: int, conds: [{column_index, operator_index, condition}]}
            agg_map = {0: "", 1: "MAX", 2: "MIN", 3: "COUNT", 4: "SUM", 5: "AVG"}
            cond_op_map = {0: "=", 1: ">", 2: "<", 3: "OP"}

            # Build SELECT clause
            agg_op = agg_map.get(sql["agg"], "")
            col_idx = sql["sel"]
            if col_idx < len(table["header"]):
                col_name = table["header"][col_idx]
                if agg_op:
                    select_clause = f"{agg_op}({col_name})"
                else:
                    select_clause = col_name
            else:
                select_clause = "col_" + str(col_idx)

            # Build WHERE clause
            where_conditions = []
            for cond in sql["conds"]:
                # cond is a dict with column_index, operator_index, condition
                cond_col_idx = cond["column_index"]
                cond_op_idx = cond["operator_index"]
                cond_val = cond["condition"]

                if cond_col_idx < len(table["header"]):
                    cond_col = table["header"][cond_col_idx]
                    op = cond_op_map.get(cond_op_idx, "=")
                    where_conditions.append(f"{cond_col} {op} {cond_val}")

            # Assemble SQL
            sql_query = f"SELECT {select_clause} FROM table"
            if where_conditions:
                sql_query += " WHERE " + " AND ".join(where_conditions)

            prompt = (
                f"### Instruction: Generate SQL from question and table.\n\n"
                f"### Table:\n{table_headers}\n\n"
                f"### Question:\n{question}\n\n"
                f"### SQL:\n{sql_query}"
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
        import copy
        labels = copy.deepcopy(model_inputs["input_ids"])

        # Replace padding token id with -100
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] == tokenizer.pad_token_id:
                    labels[i][j] = -100

        model_inputs["labels"] = labels

        return model_inputs

    return preprocess


def _get_multinli_preprocess_fn(tokenizer: PreTrainedTokenizer, max_length: int):
    """
    Get preprocessing function for Multi-NLI dataset.

    Multi-NLI task: 3-way classification (entailment/neutral/contradiction).
    We use text generation approach: model generates the label as text.
    """
    def preprocess(examples: Dict[str, Any]) -> Dict[str, Any]:
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        prompts = []

        for premise, hypothesis, label in zip(
            examples["premise"],
            examples["hypothesis"],
            examples["label"]
        ):
            label_text = label_map.get(label, "neutral")
            prompt = (
                f"### Instruction: Classify the relationship.\n\n"
                f"### Premise:\n{premise}\n\n"
                f"### Hypothesis:\n{hypothesis}\n\n"
                f"### Classification:\n{label_text}"
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
        import copy
        labels = copy.deepcopy(model_inputs["input_ids"])

        # Replace padding token id with -100
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] == tokenizer.pad_token_id:
                    labels[i][j] = -100

        model_inputs["labels"] = labels

        return model_inputs

    return preprocess


def get_data_collator(tokenizer: PreTrainedTokenizer):
    """
    Get data collator for language modeling.

    Args:
        tokenizer: Tokenizer

    Returns:
        Data collator function
    """
    # CRITICAL: Use default_data_collator, NOT DataCollatorForLanguageModeling
    # DataCollatorForLanguageModeling will overwrite our carefully masked labels
    # We already did padding and label masking in preprocessing
    # default_data_collator just converts to tensors without modifying data
    return default_data_collator


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
    elif config.dataset_type == DatasetType.WIKISQL:
        # Load WikiSQL validation set
        import json
        import tarfile
        from pathlib import Path
        import tempfile
        import urllib.request

        data_url = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            archive_path = tmpdir_path / "data.tar.bz2"

            print(f"Downloading WikiSQL validation data...")
            urllib.request.urlretrieve(data_url, archive_path)

            print(f"Extracting WikiSQL validation data...")
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(tmpdir_path)

            data_dir = tmpdir_path / "data"

            # Load validation data (dev split)
            dev_file = data_dir / "dev.jsonl"
            tables_file = data_dir / "dev.tables.jsonl"

            # Load tables
            tables = {}
            with open(tables_file, 'r', encoding='utf-8') as f:
                for line in f:
                    table = json.loads(line)
                    tables[table['id']] = table

            # Load validation examples
            examples = []
            with open(dev_file, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    example['table'] = tables[example['table_id']]

                    # Handle missing fields
                    example['table']['page_title'] = example['table'].get('page_title', '')
                    example['table']['section_title'] = example['table'].get('section_title', '')
                    example['table']['caption'] = example['table'].get('caption', '')
                    example['table']['name'] = example['table'].get('name', '')
                    example['table']['page_id'] = str(example['table'].get('page_id', ''))

                    # Convert rows to strings
                    example['table']['rows'] = [[str(cell) for cell in row] for row in example['table']['rows']]

                    # Restructure conds to dict format
                    conds = []
                    for cond in example['sql']['conds']:
                        conds.append({
                            'column_index': cond[0],
                            'operator_index': cond[1],
                            'condition': str(cond[2])
                        })
                    example['sql']['conds'] = conds

                    examples.append(example)

                    if len(examples) >= val_samples:
                        break

            dataset = Dataset.from_list(examples)

        preprocess_fn = _get_wikisql_preprocess_fn(tokenizer, config.max_length)
    elif config.dataset_type == DatasetType.MULTI_NLI:
        dataset = load_dataset("nyu-mll/multi_nli", split="validation_matched")
        dataset = dataset.select(range(min(val_samples, len(dataset))))
        preprocess_fn = _get_multinli_preprocess_fn(tokenizer, config.max_length)
    else:
        return None

    tokenized_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing validation set",
    )

    return tokenized_dataset
