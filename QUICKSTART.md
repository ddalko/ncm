# Quick Start Guide

## 설치

1. **필수 패키지 설치**

```bash
pip install -r requirements.txt
```

2. **Wandb 로그인** (선택사항)

```bash
wandb login
```

## 사용 방법

### 1. 데이터 준비

SSI 데이터셋을 다운로드하고 manifest 파일을 생성합니다:

```bash
python -m scripts.prepare_ssi \
    --output_dir data/ssi \
    --train_size 8000 \
    --dev_size 1500 \
    --test_size 1500 \
    --seed 42
```

이 명령은 다음과 같은 파일들을 생성합니다:
- `data/ssi/train.jsonl` - 학습 데이터
- `data/ssi/dev.jsonl` - 검증 데이터
- `data/ssi/test.jsonl` - 테스트 데이터

### 2. 모델 학습

Joint RNN-T + LAS 모델을 학습합니다:

```bash
# 방법 1: 쉘 스크립트 사용
bash scripts/run_joint.sh

# 방법 2: 직접 Python 실행
python -m src.train_joint experiment=joint_ssi

# 방법 3: 커스텀 설정으로 실행
python -m src.train_joint \
    experiment=joint_ssi \
    training.batch_size=32 \
    training.lr=0.001 \
    training.num_epochs=50
```

학습 중에는 다음과 같은 정보가 로깅됩니다:
- Training loss (total, RNN-T, LAS)
- Validation loss 및 WER/CER
- 학습률 변화
- 체크포인트는 `exp/joint_ssi/`에 저장됩니다

### 3. 모델 평가

#### RNN-T만 사용 (First Pass)

```bash
# 방법 1: 쉘 스크립트 사용
bash scripts/decode_rnnt.sh

# 방법 2: 직접 Python 실행
python -m src.decode_rnnt \
    model.checkpoint=exp/joint_ssi/best.ckpt \
    decoding.output_path=exp/joint_ssi/decode_rnnt.jsonl
```

#### Two-Pass (RNN-T + LAS Rescoring)

```bash
# 방법 1: 쉘 스크립트 사용
bash scripts/decode_two_pass.sh

# 방법 2: 직접 Python 실행
python -m src.decode_two_pass \
    model.checkpoint=exp/joint_ssi/best.ckpt \
    decoding.output_path=exp/joint_ssi/decode_2pass.jsonl
```

결과는 다음 파일들에 저장됩니다:
- `decode_rnnt.jsonl` - RNN-T 디코딩 결과
- `decode_2pass.jsonl` - Two-pass 디코딩 결과
- `summary_*.json` - WER/CER 메트릭

## 설정 커스터마이징

### Hydra 설정 파일 수정

설정을 변경하려면 `configs/` 디렉토리의 YAML 파일을 수정하거나, 명령줄에서 오버라이드할 수 있습니다:

```bash
# 배치 크기 변경
python -m src.train_joint training.batch_size=32

# 모델 크기 변경
python -m src.train_joint \
    model.encoder_hidden=1024 \
    model.encoder_layers=4

# 학습률 스케줄 변경
python -m src.train_joint \
    training.lr=0.0001 \
    training.num_epochs=100

# Loss weight 조정
python -m src.train_joint \
    model.loss_weights.rnnt=1.0 \
    model.loss_weights.las=0.5
```

### 새로운 실험 설정 만들기

`configs/experiment/` 디렉토리에 새 YAML 파일을 생성:

```yaml
# configs/experiment/my_experiment.yaml
data:
  root: data/ssi
  sample_rate: 16000
  n_mels: 80

model:
  encoder_hidden: 1024
  encoder_layers: 4
  # ... 기타 설정

training:
  batch_size: 32
  num_epochs: 100
  lr: 0.0001
```

그리고 다음과 같이 실행:

```bash
python -m src.train_joint experiment=my_experiment
```

## 트러블슈팅

### CUDA Out of Memory

배치 크기를 줄이거나 모델 크기를 줄입니다:

```bash
python -m src.train_joint training.batch_size=8
```

### 데이터셋 로드 오류

HuggingFace 데이터셋 로드에 실패하면 스크립트가 자동으로 더미 데이터를 생성합니다. 실제 데이터를 사용하려면 인터넷 연결을 확인하세요.

### Wandb 비활성화

Wandb를 사용하지 않으려면:

```bash
export WANDB_MODE=disabled
python -m src.train_joint experiment=joint_ssi
```

## 결과 분석

학습이 완료되면:

1. **Wandb 대시보드** - 학습 곡선, 메트릭 시각화
2. **체크포인트** - `exp/joint_ssi/best.ckpt`에서 최고 성능 모델
3. **디코딩 결과** - JSONL 파일로 모든 예측 결과 저장
4. **메트릭 요약** - JSON 파일로 WER/CER 통계

## 다음 단계

- 다른 데이터셋으로 실험하기
- 빔 서치 디코딩 구현하기
- 언어 모델 통합하기
- 모델 양자화 및 최적화하기
