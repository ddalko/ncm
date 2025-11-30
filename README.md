# **README**

# Joint Training of RNN-T + Two-Pass LAS ASR + Neural Confidence Model (NCM)

This repository implements a **Two-Pass End-to-End Automatic Speech Recognition (ASR) system** and an additional **Neural Utterance Confidence Model (NCM)**:

* **First pass:** RNN-Transducer (RNN-T) — streaming ASR
* **Second pass:** LAS (Listen-Attend-Spell) decoder — offline rescoring
* **Both models are trained jointly using shared encoder**
* **NCM** predicts whether a decoded utterance is *correct or incorrect* using features extracted from both RNN-T and LAS
  (based on *Neural Utterance Confidence Measure for RNN-Transducers and Two-Pass Models* )

Dataset: `stapesai/ssi-speech-emotion-recognition`
Configuration system: **Hydra**
Training & metrics tracking: **wandb**
---

## 1. Project Objectives

### **NCM (Neural Confidence Model)**

* [ ] Extract predictor features from RNN-T
* [ ] Extract predictor features from LAS second pass
* [ ] Build NCM dataset: label hypotheses as **Accept (1)** or **Reject (0)**
* [ ] Train binary classifier for utterance-level confidence
* [ ] Evaluate NCM using AUC, EER, NCE
* [ ] Support distributed-ASR scenario metrics (CS vs RIER) as in the paper 

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
│   │   ├── joint_ssi.yaml
│   │   ├── rnnt_only_ssi.yaml
│   │   └── ncm_ssi.yaml          # NEW — NCM training config
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
│   │   ├── ncm_model.py          # NEW — 2-layer MLP for NCM
│   │   └── metrics.py
│   ├── train_joint.py
│   ├── train_rnnt_only.py
│   ├── train_ncm.py              # NEW — NCM binary classification trainer
│   ├── extract_features.py       # NEW — feature extractor for NCM
│   ├── decode_rnnt.py
│   └── decode_two_pass.py
├── scripts/
│   ├── prepare_ssi.py
│   ├── run_joint.sh
│   ├── run_rnnt_only.sh
│   ├── run_extract_features.sh   # NEW
│   └── run_ncm.sh                # NEW
├── requirements.txt
└── README.md
```

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

### NCM logging example

```python
wandb.log({
    "train/loss_ncm": loss,
    "train/auc": auc,
    "train/eer": eer,
    "train/nce": nce,
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

# **8. Neural Utterance Confidence Model (NCM)**

*(based on Gupta et al., ICASSP 2021 )*

The NCM is a **binary classifier** predicting whether a given ASR hypothesis is correct.

---

## 8.1 **Input Feature Types**

We extract all predictor features described in the paper:

### **From RNN-T**

* **Transcription network output (Trans):**
  Mean-pooled encoder features (acoustic summary)
* **Prediction network output (Pred):**
  Mean-pooled prediction network states
* **Joint network logits (Joint):**
  Top-K logits per timestep (self-attention pooling)

### **From LAS second pass**

* **LAS encoder output (Enc):**
  Summary of second-pass encoder
* **LAS decoder logits (Dec):**
  Top-K decoder logits
* **Beam scores:**
  Log probability of each beam (RNN-T + LAS)

### **Optional**

* **Multi-beam Joint features**
* **Multi-beam LAS decoder features**
  (Paper notes these often do NOT help for RNN-T systems due to blank-heavy beams)

---

## 8.2 **NCM Architecture**

A lightweight MLP:

```text
Input: concatenated feature vector
Hidden: 64 units (ReLU)
Hidden: 64 units (ReLU)
Output: Sigmoid → p(accept)
```

Same as described in the paper.

---

## 8.3 NCM Dataset Construction

Following the paper:

1. Run trained **RNN-T + LAS** on dev set with beam search
2. For each utterance:

   * If **ASR hypothesis == reference**, label as **1 (Accept)**
   * Else label as **0 (Reject)**
3. Extract all predictor features
4. Save as pickled dataset (`train.pkl`, `dev.pkl`)

Command:

```bash
python -m src.extract_features \
  model.checkpoint=exp/joint_ssi/best.ckpt \
  data.dev_manifest=data/ssi/dev.jsonl \
  output_dir=data/ncm_features/
```

---

## 8.4 Training NCM

```bash
python -m src.train_ncm experiment=ncm_ssi
```

Metrics:

* AUC
* EER
* Normalized Cross Entropy (NCE)
* Loss (binary cross entropy)

---

## 8.5 Using NCM in Distributed-ASR Scenario

The paper proposes **Cost Saving (CS) vs Relative Increase in Error Rate (RIER)** evaluation.
We include optional evaluation script:

```bash
python -m src.eval_ncm \
  ncm.checkpoint=exp/ncm_ssi/best.ckpt \
  test_manifest=data/ssi/test.jsonl
```

Outputs:

* AUC / EER / NCE
* CS@0%, CS@5%, CS@10% (as in Table 1 of the paper)

---

## 9. Training Pipeline

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

### Step 2 — Extract NCM features

```bash
bash scripts/run_extract_features.sh
```

### Step 3 — Train NCM

```bash
bash scripts/run_ncm.sh
```

---

## 10. Decoding & Evaluation

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

### Use NCM to filter low-confidence utterances

```bash
python -m src.decode_two_pass \
  use_ncm=true \
  ncm.checkpoint=exp/ncm_ssi/best.ckpt \
  threshold=0.6
```