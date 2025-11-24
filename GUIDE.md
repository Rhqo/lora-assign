# LoRA Module Sensitivity Analysis - Usage Guide

## Quick Start

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 또는 개별 설치
pip install torch transformers peft datasets matplotlib seaborn numpy pandas tqdm accelerate
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
| Target Modules | `--target` | `attention_only` | `attention_only`, `mlp_only`, `both` |
| Rank (r) | `--r` | `8` | LoRA의 low-rank dimension |
| Alpha | `--lora_alpha` | `32` | Scaling factor (effective scale = alpha/r) |
| Dropout | `--lora_dropout` | `0.1` | LoRA dropout 확률 |

**Target Modules 상세:**
- `attention_only`: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `mlp_only`: `gate_proj`, `up_proj`, `down_proj`
- `both`: 위의 모든 모듈

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

---

## Usage Examples

### 기본 실험 실행

```bash
# 기본 설정으로 E2E + Attention 실험
python run_experiment.py --preset e2e_attention

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
├── config.json                      # 실험 설정
├── gradient_analysis.group_norms.csv   # 그룹별 gradient norm (step별)
├── gradient_analysis.layer_norms.csv   # 레이어별 gradient norm (step별)
├── gradient_analysis.summary.json      # 통계 요약
├── gradient_norms.png               # Attention vs MLP 비교 그래프
├── layer_heatmap.png               # 레이어별 히트맵
├── layer_comparison.png            # 레이어별 바 차트
├── results_summary.json            # 최종 결과 요약
└── checkpoints/                    # 모델 체크포인트
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
