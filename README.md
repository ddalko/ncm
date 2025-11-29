# **README**

# Joint Training of RNN-T + Two-Pass LAS ASR (SSI Dataset)

This repository implements a **Two-Pass End-to-End Automatic Speech Recognition (ASR) system** where:

- **First pass:** RNN-Transducer (RNN-T) — streaming ASR  
- **Second pass:** LAS (Listen-Attend-Spell) decoder — offline rescoring  
- **Both models are trained jointly** using a shared encoder and a combined loss

Dataset: `stapesai/ssi-speech-emotion-recognition`  
Configuration system: **Hydra**  
Training & metrics tracking: **Weights & Biases (wandb)**

---

## 1. Project Objectives

- [ ] Implement a compact RNN-T model  
- [ ] Implement a LAS decoder sharing the encoder with RNN-T  
- [ ] Integrate both models into a **joint training pipeline**  
- [ ] Log all losses and metrics via wandb  
- [ ] Use Hydra for experiment configuration  

---

## 2. Dataset

### 2.1 Source
We use the HuggingFace dataset:

**`stapesai/ssi-speech-emotion-recognition`**

Each sample contains:
- `audio` — waveform & sampling rate  
- `text` — transcription  
- Additional fields (`emotion`, `speaker_id`, …) but **not used** for ASR

### 2.2 ASR Usage
We treat this dataset as a **small speech corpus**:
- The `text` field becomes the transcription target
- The `audio` array is used as the waveform input

### 2.3 Recommended Data Split
We re-split the dataset to ensure reproducibility:

| Split | # Utterances | Purpose |
|-------|--------------|---------|
| **train_asr** | ~9,500 | Joint RNN-T + LAS training |
| **dev_asr**   | ~1,500 | Validation |

A deterministic split is created by:

````
python -m scripts.prepare_ssi
````

---

## 3. Repository Structure

```text
.
├── configs/
│   ├── experiment/
│   │   ├── joint_ssi.yaml      # Joint RNN-T + LAS experiment config (Hydra)
│   │   └── rnnt_only_ssi.yaml  # Optional RNN-T baseline
│   ├── decoding/
│   │   └── ssi.yaml
│   └── data/
│       └── ssi.yaml
├── src/
│   ├── data/
│   │   ├── ssi_dataset.py
│   │   └── featurizer.py
│   ├── models/
│   │   ├── rnnt.py
│   │   ├── las_decoder.py
│   │   ├── joint_model.py
│   │   ├── rnnt_loss.py
│   │   └── metrics.py
│   ├── train_joint.py
│   ├── train_rnnt_only.py
│   ├── decode_rnnt.py
│   └── decode_two_pass.py
├── scripts/
│   ├── prepare_ssi.py
│   ├── run_joint.sh
│   └── run_rnnt_only.sh
├── requirements.txt
└── README.md
````

---

## 4. Environment & Installation

### 4.1 Requirements

* Python **≥3.9**
* CUDA GPU recommended
* Key libraries:

  * `torch`, `torchaudio`
  * `datasets`
  * `numpy`, `scipy`
  * `jiwer`
  * `wandb`
  * `hydra-core`, `omegaconf`

---

## 5. wandb Integration

### 5.1 Logging Pattern

Used inside all training scripts (`train_joint.py`, `train_rnnt_only.py`):

```python
wandb.log({
    "train/loss_total": total_loss,
    "train/loss_rnnt": loss_rnnt,
    "train/loss_las": loss_las,
    "train/wer_rnnt": wer_rnnt,
    "train/wer_2pass": wer_2pass,
    "lr": lr_scheduler.get_last_lr()[0],
}, step=step)
```

Validation metrics (each epoch):

```python
wandb.log({
    "val/loss_total": val_total_loss,
    "val/loss_rnnt": val_loss_rnnt,
    "val/loss_las": val_loss_las,
    "val/wer_rnnt": val_wer_rnnt,
    "val/wer_2pass": val_wer_2pass,
    "epoch": epoch,
})
```

---

## 6. Hydra Configs

### Example: `configs/experiment/joint_ssi.yaml`

```yaml
defaults:
  - data: ssi
  - decoding: ssi

data:
  root: data/ssi
  train_manifest: ${data.root}/train.jsonl
  dev_manifest: ${data.root}/dev.jsonl
  sample_rate: 16000
  n_mels: 80

model:
  vocab_size: 300
  encoder_hidden: 512
  encoder_layers: 3
  pred_hidden: 512
  pred_layers: 1
  las:
    enc2_hidden: 512
    dec_hidden: 512
    embedding_dim: 256
    attention: "location"

  loss_weights:
    rnnt: 1.0
    las: 1.0

training:
  batch_size: 16
  num_epochs: 40
  lr: 1e-3
  grad_clip: 5.0
  output_dir: exp/joint_ssi

wandb:
  project: "ssi-two-pass"
  entity: "<your-entity>"
  run_name: "joint_ssi"

hydra:
  run:
    dir: ${training.output_dir}
```

---

## 7. Model Architecture

### 7.1 Shared Encoder (used by RNN-T & LAS)

* 3× uni-LSTM (hidden 512)
* Optional time-reduction (e.g., pooling)

### 7.2 RNN-T (Streaming)

**Prediction Network**

* 1× LSTM or GRU
* Hidden size: 512

**Joint Network**

* FC(1024 → 512)
* FC(512 → vocab_size + blank)

**Loss**

* Standard **RNN-T loss** (warp-rnnt / torchaudio)

### 7.3 Two-Pass LAS (Offline)

**LAS Encoder**

* 1× BiLSTM (hidden 512 → output 1024)

**Decoder**

* Embedding dim 256
* LSTM hidden 512
* Location or dot attention
* FC(512 → vocab_size)

**Loss**

* Token-level cross entropy (teacher forcing)

### 7.4 Joint Training Objective

```text
L_total = λ_rnnt * L_rnnt + λ_las * L_las
```

* Encoder is updated by **both losses**
* Prediction & Joint networks → only RNN-T
* LAS encoder/decoder → only LAS loss

---

## 8. Training Pipeline

### Step 0 — Prepare dataset

```bash
python -m scripts.prepare_ssi \
  output_dir=data/ssi/ \
  train_size=8000 \
  dev_size=1500 \
  test_size=1500 \
  seed=42
```

### Step 1 — Joint Training (RNN-T + LAS)

```bash
python -m src.train_joint experiment=joint_ssi
```

Metrics logged:

* loss_total, loss_rnnt, loss_las
* wer_rnnt, wer_2pass
* cer_rnnt, cer_2pass

---

## 9. Decoding & Evaluation

### RNN-T Only (Streaming)

```bash
python -m src.decode_rnnt \
  model.checkpoint=exp/joint_ssi/best.ckpt \
  data.test_manifest=data/ssi/test.jsonl \
  decoding.beam_size=4 \
  output_path=exp/joint_ssi/decode_rnnt.jsonl
```

### Two-Pass (RNN-T + LAS Rescoring)

```bash
python -m src.decode_two_pass \
  model.checkpoint=exp/joint_ssi/best.ckpt \
  data.test_manifest=data/ssi/test.jsonl \
  decoding.beam_size_rnnt=4 \
  decoding.beam_size_las=4 \
  output_path=exp/joint_ssi/decode_2pass.jsonl
```
