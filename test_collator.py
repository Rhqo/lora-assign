"""
Test if DataCollator is overwriting our carefully masked labels.
"""
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import torch

# Initialize
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Pad token ID: {tokenizer.pad_token_id}")

# Create sample data with masked labels
sample_input_ids = [1, 2, 3, tokenizer.pad_token_id, tokenizer.pad_token_id]
sample_labels = [1, 2, 3, -100, -100]  # Properly masked

print("\n" + "="*70)
print("BEFORE DataCollator:")
print("="*70)
print(f"input_ids: {sample_input_ids}")
print(f"labels:    {sample_labels}")
print(f"Padding is masked: {sample_labels[3] == -100}")

# Create batch
batch = [{
    "input_ids": sample_input_ids,
    "labels": sample_labels,
}]

# Apply DataCollator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
collated = collator(batch)

print("\n" + "="*70)
print("AFTER DataCollator:")
print("="*70)
print(f"input_ids: {collated['input_ids'][0].tolist()}")
print(f"labels:    {collated['labels'][0].tolist()}")

# Check if labels were corrupted
labels_after = collated['labels'][0].tolist()
if labels_after[3] == -100:
    print("\n✅ GOOD: Padding still masked after DataCollator")
else:
    print(f"\n❌ PROBLEM: Padding NOT masked! labels[3] = {labels_after[3]}")
    print("   DataCollatorForLanguageModeling is overwriting our masked labels!")
    print("\n   SOLUTION: Don't use DataCollatorForLanguageModeling")
    print("             Use default_data_collator instead")
