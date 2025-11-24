# Module-wise Sensitivity Analysis in LoRA: Attention vs. MLP

## 1. Introduction & Motivation

**Low-Rank Adaptation (LoRA)**는 거대 언어 모델(LLM)을 효율적으로 튜닝하는 표준이 되었습니다. 기존 LoRA 논문은 주로 **Self-Attention Module**($W_q, W_v$)의 적응에 집중했지만, 최근 연구들은 **Feed-Forward Networks (MLP)**가 지식 저장소(Knowledge Storage)로서 중요한 역할을 한다고 제안합니다.

본 프로젝트는 **"Task의 성격(Structure vs. Semantics)에 따라 LoRA를 적용해야 할 최적의 모듈(Attention vs. MLP)이 다를 것이다"**라는 가설을 검증합니다. 이를 위해 **LISA (Layerwise Importance Sampling)**와 **SNIP** 등의 방법론에서 영감을 받아, 학습 초기의 **Gradient Norm**을 측정하여 각 모듈의 학습 민감도(Sensitivity)를 비교 분석합니다.

## 2. Hypothesis

우리는 모델의 각 모듈이 서로 다른 언어적 기능을 담당한다고 가정합니다.

* **Hypothesis 1**
    * 둘 다 MLP Module의 weight norm이 크게 나타난다면, 모델은 weight norm에 대부분의 정보를 저장하는 것이 맞을 것이다.
* **Hypothesis 2**
    * **Structure-heavy Tasks (e.g., E2E NLG):** 입력 데이터를 정해진 형식(Format)으로 변환해야 하는 Task에서는, 사실적 지식과 패턴을 처리하는 **MLP Module**의 Gradient Norm이 더 높게 나타날 것이다.
    * **Reasoning-heavy Tasks (e.g., SAMSum):** 긴 문맥의 인과관계를 파악하고 요약해야 하는 Task에서는, 정보의 라우팅과 문맥 혼합을 담당하는 **Attention Module**의 Gradient Norm이 더 높게 나타날 것이다.
* **Hypothesis 3**
    * mlp lora 적용한 layer와 attention lora 적용한 layer를 weight norm 기준으로 적절하게 섞으면 더 parameter-effecient하고 성능이 좋은 모델을 만들 수 있을 것이다.

## 3. Experimental Setup

### 3.1 Model
* **Base Model:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
    * 선정 이유: 빠른 실험이 가능하며, 1.1B 규모에서도 충분한 Generation 성능을 보여줌.

### 3.2 Datasets (Task Comparison)
1.  **E2E NLG (Structure/Format Focus):**
    * 입력: Key-Value 쌍 (e.g., `name[The Eagle], price[cheap]`)
    * 출력: 자연어 문장
    * 목표: 정형 데이터의 **구조적 변환(Syntactic Transformation)** 능력 검증.
2.  **SAMSum (Semantics/Context Focus):**
    * 입력: 메신저 대화 로그
    * 출력: 요약문
    * 목표: 대화의 흐름과 **함축적 의미(Semantic Reasoning)** 파악 능력 검증.

### 3.3 Comparison Groups
모든 실험에서 Trainable Parameter의 총량(Budget)은 유사하게 통제합니다.
* **Group A (Attention-Only):** `q_proj`, `k_proj`, `v_proj`, `o_proj`
* **Group B (MLP-Only):** `gate_proj`, `up_proj`, `down_proj`

## 4. Methodology: Gradient Norm Analysis

우리는 학습 과정에서 Backpropagation 직후, Optimizer Update 직전에 각 LoRA 모듈($A, B$ 행렬)의 **L2 Gradient Norm**을 측정합니다.

$$
\text{Norm}_{module} = \sqrt{\sum_{p \in \Theta_{module}} \|\nabla p\|_2^2}
$$

이 지표가 높을수록 해당 모듈이 Loss 감소에 더 민감하게 반응하며, 학습 초기에 더 큰 업데이트를 요구함을 의미합니다.

## 5. References

본 연구는 다음 논문들의 아이디어와 실험 결과를 기반으로 설계되었습니다.

* **Base Method (LoRA):** Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*, ICLR 2022.
    * LoRA의 기본 구조 및 $W_q, W_v$ 적용 효율성 참고.
* **Gradient Analysis (LISA):** Pan et al., *"LISA: Layerwise Importance Sampling for Finetuning"*, arXiv 2024.
    * Gradient Norm을 통한 Layer 중요도 분석 방법론 차용.
* **Adaptive Rank (AdaLoRA):** Zhang et al., *"AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"*, ICLR 2023.
    * Task별로 중요 모듈(Attn vs MLP)이 다를 수 있다는 근거.
* **Magnitude Analysis (DoRA):** Liu et al., *"DoRA: Weight-Decomposed Low-Rank Adaptation"*, ICML 2024.
    * Weight Update Magnitude 분석 아이디어 참고.

## 6. How to Run (Base Code)

### Requirements
```bash
pip install torch transformers peft datasets matplotlib
```

### Experiment Script (`experiment.py`)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. Custom Callback for Gradient Norm Measurement
# ---------------------------------------------------------
class GradientNormCallback(TrainerCallback):
    def __init__(self, log_dict):
        self.log_dict = log_dict

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        After backpropagation, calculate the L2 Norm of gradients 
        for the trainable LoRA parameters.
        """
        total_norm = 0.0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        self.log_dict['steps'].append(state.global_step)
        self.log_dict['norms'].append(total_norm)

# ---------------------------------------------------------
# 2. Setup Configuration
# ---------------------------------------------------------
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# Select Dataset: "e2e_nlg" or "samsum"
dataset_id = "e2e_nlg" 

# Experimental Groups: Comparison between Attn and MLP
experiments = {
    "Attention_Only": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "MLP_Only": ["gate_proj", "up_proj", "down_proj"]
}

results = {
    "Attention_Only": {'steps': [], 'norms': []},
    "MLP_Only": {'steps': [], 'norms': []}
}

# ---------------------------------------------------------
# 3. Data Preprocessing
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def process_data_e2e(samples):
    inputs = [f"Table: {mr}\nSummary: {ref}" for mr, ref in zip(samples['meaning_representation'], samples['human_reference'])]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Load Data (Subsample for quick analysis)
dataset = load_dataset(dataset_id, split="train[:500]")
tokenized_dataset = dataset.map(process_data_e2e, batched=True)

# ---------------------------------------------------------
# 4. Running Experiments
# ---------------------------------------------------------
for exp_name, target_modules in experiments.items():
    print(f"\n=== Running Experiment: {exp_name} on {dataset_id} ===")
    
    # Reload model for fresh start
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    # LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules # Switching Target Modules
    )
    
    model = get_peft_model(model, peft_config)
    
    # Calculate params to ensure fair comparison
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")

    args = TrainingArguments(
        output_dir=f"./results_{exp_name}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=100,  # Short steps for initial gradient analysis
        learning_rate=2e-4,
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none"
    )
    
    grad_callback = GradientNormCallback(results[exp_name])

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        callbacks=[grad_callback]
    )
    
    trainer.train()
    
    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# 5. Visualization & Analysis
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
for exp_name, data in results.items():
    # Smoothing for better visualization
    steps = data['steps']
    norms = data['norms']
    plt.plot(steps, norms, label=f"{exp_name} (Mean: {np.mean(norms):.4f})", alpha=0.8)

plt.xlabel('Training Steps')
plt.ylabel('Gradient Norm (L2)')
plt.title(f'Module Sensitivity Analysis on {dataset_id}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"grad_norm_{dataset_id}.png")
print(f"Analysis saved to grad_norm_{dataset_id}.png")
```

## 7. Expected Results

* **E2E NLG:** 입력의 구조적 변환이 지배적이므로 **MLP-Only LoRA**가 초기 Gradient Norm이 더 높게 나타나거나, Attention과 유사한 수준의 민감도를 보일 것으로 예상됨.
* **SAMSum:** 대화 맥락 처리가 중요하므로 **Attention-Only LoRA**의 Gradient Norm이 MLP보다 유의미하게 높게 나타날 것으로 예상됨.

이 결과는 Task 특성에 맞춰 LoRA Target Module을 선별적으로 적용하는 **"Task-Adaptive LoRA Configuration"**의 근거로 활용될 수 있습니다.

## 8. Datasets

### E2E NLG Challenge

```python
from datasets import load_dataset

ds = load_dataset("GEM/e2e_nlg")
```

### SamSum

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("knkarthick/samsum")
```