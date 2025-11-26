# LoRA Module Sensitivity Analysis - Usage Guide

## Quick Start

### 1. 환경 설정

```bash
# 의존성 설치
pip install torch transformers peft datasets matplotlib seaborn numpy pandas tqdm accelerate

# Evaluation 관련 패키지
pip install evaluate nltk rouge_score
```

### 2. 빠른 실행 (프리셋 사용)

```bash
# E2E NLG 데이터셋 + Attention-only LoRA
python run_experiment.py --preset e2e_attention

# E2E NLG 데이터셋 + MLP-only LoRA
python run_experiment.py --preset e2e_mlp

# SAMSum 데이터셋 + Attention-only LoRA
python run_experiment.py --preset samsum_attention

# SAMSum 데이터셋 + MLP-only LoRA
python run_experiment.py --preset samsum_mlp
```

### 3. 모든 실험 일괄 실행

```bash
python run_experiment.py --batch
```

---

## Hyperparameters Reference

### Dataset Configuration

| 파라미터 | CLI 옵션 | 기본값 | 설명 |
|---------|---------|-------|------|
| Dataset Type | `--dataset` | `e2e_nlg` | `e2e_nlg` 또는 `samsum` |
| Num Samples | `--num_samples` | `500` | 학습에 사용할 샘플 수 (0 = 전체) |
| Max Length | `--max_length` | `128` | 토큰 최대 길이 |

### LoRA Configuration

| 파라미터 | CLI 옵션 | 기본값 | 설명 |
|---------|---------|-------|------|
| Target Modules | `--target` | `attention_only` | `attention_only`, `mlp_only`, `both`, `all` |
| Rank (r) | `--r` | `8` | LoRA의 low-rank dimension |
| Alpha | `--lora_alpha` | `32` | Scaling factor (effective scale = alpha/r) |
| Dropout | `--lora_dropout` | `0.1` | LoRA dropout 확률 |

**Target Modules 상세:**
- `attention_only`: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `mlp_only`: `gate_proj`, `up_proj`, `down_proj`
- `both`: Attention + MLP 모듈
- `all`: Attention + MLP + `embed_tokens` + `lm_head` (LISA 스타일)

### Training Configuration

| 파라미터 | CLI 옵션 | 기본값 | 설명 |
|---------|---------|-------|------|
| Batch Size | `--batch_size` | `4` | Per-device batch size |
| Gradient Accum | `--grad_accum` | `4` | Gradient accumulation steps |
| Max Steps | `--max_steps` | `100` | 최대 학습 스텝 수 |
| Learning Rate | `--lr` | `2e-4` | Learning rate |
| Warmup Steps | `--warmup_steps` | `10` | Warmup 스텝 수 |
| Seed | `--seed` | `42` | Random seed |

### Model Configuration

| 파라미터 | CLI 옵션 | 기본값 | 설명 |
|---------|---------|-------|------|
| Model ID | `--model` | `TinyLlama/TinyLlama-1.1B-...` | HuggingFace 모델 ID |

### Analysis Configuration

| 파라미터 | CLI 옵션 | 기본값 | 설명 |
|---------|---------|-------|------|
| Log Frequency | `--log_frequency` | `1` | Gradient norm 기록 주기 |
| Layer Norm | `--no_layer_norm` | `False` | 레이어별 측정 비활성화 |

### Evaluation Configuration

| 파라미터 | CLI 옵션 | 기본값 | 설명 |
|---------|---------|-------|------|
| Enable Eval | `--no_eval` | `True` | 학습 후 자동 평가 비활성화 |
| Eval Samples | `--eval_samples` | `100` | 평가에 사용할 샘플 수 |

**평가 지표:**
- **BLEU**: 생성 텍스트와 참조 텍스트 간 n-gram 일치도
- **ROUGE-1/2/L**: 단어/바이그램/최장 공통 시퀀스 기반 recall
- **METEOR**: 형태소 기반 유사도

---

## Usage Examples

### 기본 실험 실행

```bash
# 기본 설정으로 E2E + Attention 실험
python run_experiment.py --preset e2e_attention

# All layers (embed + attention + mlp + lm_head) 실험
python run_experiment.py --dataset e2e_nlg --target all --max_steps 100

# Evaluation 포함 실행 (기본값)
python run_experiment.py --dataset e2e_nlg --target all --eval_samples 100

# Evaluation 없이 빠른 실행
python run_experiment.py --dataset e2e_nlg --target all --no_eval

# 결과 확인
ls results/e2e_attention_only/
```

### 하이퍼파라미터 커스터마이징

```bash
# LoRA rank를 16으로, learning rate를 1e-4로 변경
python run_experiment.py \
    --dataset e2e_nlg \
    --target attention_only \
    --r 16 \
    --lr 1e-4 \
    --max_steps 200 \
    --experiment_name "e2e_attn_r16_lr1e4"
```

### 더 큰 데이터셋으로 실험

```bash
# 전체 데이터셋 사용 (num_samples=0)
python run_experiment.py \
    --dataset e2e_nlg \
    --target mlp_only \
    --num_samples 0 \
    --max_steps 500
```

### Attention vs MLP 비교 실험

```bash
# 1. Attention-only 실험
python run_experiment.py \
    --dataset e2e_nlg \
    --target attention_only \
    --experiment_name "compare_attn"

# 2. MLP-only 실험
python run_experiment.py \
    --dataset e2e_nlg \
    --target mlp_only \
    --experiment_name "compare_mlp"

# 3. 결과 비교
python run_experiment.py --compare results/compare_attn results/compare_mlp
```

### 다양한 Rank 비교

```bash
for r in 4 8 16 32; do
    python run_experiment.py \
        --dataset e2e_nlg \
        --target attention_only \
        --r $r \
        --experiment_name "e2e_attn_r${r}"
done

# 비교
python run_experiment.py --compare results/e2e_attn_r4 results/e2e_attn_r8 results/e2e_attn_r16 results/e2e_attn_r32
```

---

## Output Structure

실험 완료 후 `results/<experiment_name>/` 디렉토리에 다음 파일들이 생성됩니다:

```
results/<experiment_name>/
├── config.json                         # 실험 설정
├── training_loss.png                   # ⭐ Training & Evaluation Loss 곡선
├── gradient_analysis.group_norms.csv   # 그룹별 gradient norm (step별)
├── gradient_analysis.layer_norms.csv   # 레이어별 gradient norm (step별)
├── gradient_analysis.summary.json      # Gradient 통계 요약
├── gradient_norms.png                  # Module별 gradient norm 그래프
├── layer_heatmap.png                   # 레이어별 히트맵 (3-panel)
├── layer_comparison.png                # 레이어별 바 차트 (stacked + side-by-side)
├── gradient_evolution.png              # Gradient evolution (rainbow 7-step)
├── evaluation_results.json             # ⭐ 평가 결과 (BLEU, ROUGE, METEOR)
├── results_summary.json                # 최종 결과 요약
├── checkpoints/                        # 학습 중 checkpoint (최대 2개: best + last)
│   ├── checkpoint-50/                 # Latest checkpoint
│   └── checkpoint-100/                # Best checkpoint (eval_loss 기준)
└── final_model/                        # ⭐ 최종 모델 (best 또는 last)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

### 주요 출력 파일 설명

**Loss 분석:**
- `training_loss.png`: Training loss와 evaluation loss 곡선 (최소값 표시 포함)
  - 파란색: Training loss
  - 주황색: Evaluation loss
  - 각 곡선의 최소값 위치와 값이 annotation으로 표시됨

**Gradient 분석:**
- `gradient_analysis.group_norms.csv`: embed, attention, mlp, lm_head, total 각각의 norm
- `gradient_analysis.layer_norms.csv`: 각 레이어(L0~L21)의 attention/mlp norm
- `gradient_evolution.png`: 학습 초기(빨강) → 후기(보라) gradient 변화

**Evaluation 결과 (evaluation_results.json):**
```json
{
  "bleu": 0.2567,
  "rouge1": 0.4789,
  "rouge2": 0.2356,
  "rougeL": 0.4012,
  "meteor": 0.3678,
  "num_samples": 100
}
```

**Checkpoint 정책:**
- Validation 사용 시: `save_total_limit=2`로 best + last만 유지
- Validation 미사용 시: 중간 checkpoint 저장 안함
- 최종 모델은 항상 `final_model/`에 저장

---

## Advanced Features

### 3D Gradient Visualization

학습 중 gradient evolution을 3차원으로 시각화할 수 있습니다:

```bash
# 3D surface plot 생성
python -m src.utils.3d_visualization \
    --input results/e2e_all/gradient_analysis \
    --output results/e2e_all/3d_plots \
    --type surface

# 모든 타입 생성 (surface, wireframe, bars)
python -m src.utils.3d_visualization \
    --input results/e2e_all/gradient_analysis \
    --output results/e2e_all/3d_plots \
    --type all
```

### Model Evaluation

학습된 모델을 수동으로 평가할 수 있습니다:

```bash
# 단일 모델 평가
python -m src.evaluation \
    --model results/e2e_all/final_model \
    --base TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --dataset e2e_nlg \
    --samples 100 \
    --output results/e2e_all/manual_eval.json
```

**Python에서 여러 모델 비교:**
```python
from src.evaluation import compare_models, print_comparison_table

results = compare_models(
    model_paths={
        "attention": "results/e2e_attention/final_model",
        "mlp": "results/e2e_mlp/final_model",
        "all": "results/e2e_all/final_model",
    },
    base_model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    dataset_type="e2e_nlg",
    num_samples=100,
    output_dir="results/eval_comparison",
)

print_comparison_table(results)
```

---

## Python API 사용

코드에서 직접 실험을 실행할 수도 있습니다:

```python
from configs.config import ExperimentConfig, DatasetType, TargetModuleGroup

# 설정 생성
config = ExperimentConfig()
config.data.dataset_type = DatasetType.E2E_NLG
config.lora.target_module_group = TargetModuleGroup.MLP_ONLY
config.lora.r = 16
config.training.max_steps = 200
config.training.learning_rate = 1e-4
config.experiment_name = "my_custom_experiment"

# 실험 실행
from run_experiment import run_single_experiment
result = run_single_experiment(config)

# 결과 확인
print(result["summary"])
```

### 결과 시각화

```python
from src.utils.visualization import (
    load_gradient_log_from_csv,
    plot_gradient_norms,
    plot_layer_heatmap,
    generate_summary_report,
)

# 이전 실험 결과 로드
log1 = load_gradient_log_from_csv("results/e2e_attention_only/gradient_analysis")
log2 = load_gradient_log_from_csv("results/e2e_mlp_only/gradient_analysis")

# 비교 그래프 생성
plot_gradient_norms(
    {"Attention": log1, "MLP": log2},
    title="E2E NLG: Attention vs MLP",
    save_path="comparison.png"
)
```

---

## Tips

### 메모리 부족 시
- `--batch_size`를 줄이고 `--grad_accum`을 늘리세요
- `--max_length`를 줄이세요

### 빠른 테스트
- `--num_samples 100 --max_steps 50`으로 빠르게 테스트

### 재현성
- `--seed` 값을 고정하여 실험 재현

### GPU 메모리 모니터링
```bash
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# 배치 사이즈 줄이기
python run_experiment.py --preset e2e_attention --batch_size 2 --grad_accum 8
```

### Dataset Loading Error
```bash
# 캐시 정리
rm -rf ~/.cache/huggingface/datasets/tuetschek___e2e_nlg
```

### Import Error
```bash
# 프로젝트 루트에서 실행하세요
cd /path/to/lora
python run_experiment.py --preset e2e_attention
```
