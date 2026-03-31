# Hate Speech Detection and Classification

A deep learning project for automated hate speech detection and classification in online text, built on a 25-paper literature review and implemented using a Bidirectional LSTM (BiLSTM) architecture.

---

## Overview

Online platforms are flooded with harmful content that moderators cannot manually review at scale. This project builds a reliable, low-resource classifier that detects and classifies hate speech in social media text into three categories:

- **Hate Speech** — content targeting individuals or groups based on identity
- **Offensive Language** — offensive but not necessarily hateful
- **Neither** — clean content

The model is designed specifically to address the class imbalance problem that plagues most hate speech datasets, where genuine hate speech makes up only a small fraction of total posts.

---

## Dataset

We use the **Davidson et al. (2017)** Twitter dataset — one of the most widely cited benchmarks in hate speech research.

| Split | Size |
|-------|------|
| Total labeled tweets | 24,802 |
| Hate speech | ~5% |
| Offensive | ~77% |
| Neither | ~18% |

**Source:** [github.com/t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language)

> ⚠️ The dataset contains offensive and hateful language by nature. It is used strictly for research purposes.

---

## Gaps Identified from Literature (25 Papers)

Our literature survey identified five key gaps in existing approaches that shaped this project:

1. **Inconsistent labelling** — no universal dataset standard exists across studies, making cross-dataset comparison unreliable.
2. **Resource cost of high-accuracy models** — BERT achieves strong results but is impractical in low-resource or real-time deployment settings.
3. **Imbalanced datasets** — most corpora have very few genuine hate speech samples, causing classifiers to be biased toward the majority class.
4. **Poor cross-domain transfer** — models trained on one platform or language perform poorly on another due to varying hate speech norms.
5. **Overfitting in simple CNNs** — narrow CNN architectures generalise poorly beyond their training distribution.

BiLSTM was selected to directly address gaps 3 and 5: bidirectional recurrence captures contextual meaning across the full sequence, and it handles imbalanced data more robustly than simple convolutional models.

---

## Model Architecture

```
Input Text
    │
    ▼
Tokenization + Padding
    │
    ▼
Embedding Layer  (pre-trained / trainable)
    │
    ▼
Bidirectional LSTM
    │
    ▼
Dropout
    │
    ▼
Dense Layer
    │
    ▼
Softmax → [Hate Speech | Offensive | Neither]
```

**Why BiLSTM?**
Unlike a unidirectional LSTM, BiLSTM processes the sequence in both forward and backward directions, capturing context from both sides of each word. This is critical for hate speech, where meaning is highly context-dependent (e.g., reclaimed slurs, sarcasm, quoting).

---

## Repository Structure

```
hate-speech-detection/
├── data/               # Dataset files (raw and preprocessed)
├── models/             # Saved model weights and checkpoints
├── plots/              # Training curves, confusion matrices, evaluation plots
├── src/                # Source code (preprocessing, model, training, evaluation)
└── README.md
```

---

## Setup

**Requirements**

```
Python 3.8+
TensorFlow / Keras
NumPy
Pandas
scikit-learn
Matplotlib
```

**Install dependencies**

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

---

## Usage

**1. Preprocess the data**
```bash
python src/preprocess.py
```

**2. Train the model**
```bash
python src/train.py
```

**3. Evaluate**
```bash
python src/evaluate.py
```

Trained model weights are saved to `models/`. Evaluation plots (confusion matrix, training curves) are saved to `plots/`.

---

## Evaluation Metric

Overall accuracy is **not** the primary metric here — a model that predicts "Offensive" for everything would score ~77% accuracy while being useless.

We report **per-class F1-score** and **macro F1** as the primary metrics throughout, consistent with best practices in imbalanced classification tasks.

---

## Key References

| # | Paper | Relevance |
|---|-------|-----------|
| 1 | Davidson et al., ICWSM 2017 | Dataset + 3-class framework |
| 2 | Mullah & Zainon, IEEE Access 2021 | Survey: BiLSTM-CNN on imbalanced data |
| 3 | Badjatiya et al., WWW 2017 | LSTM + GBDT outperforms CNN baselines |
| 4 | Fortuna & Nunes, ACM CSUR 2019 | Hate speech challenges: context, sarcasm |
| 5 | MacAvaney et al., PLOS ONE 2019 | Cross-domain transfer failure modes |

Full literature review (25 papers) available in the project report.

---

## Limitations

- Trained on English Twitter data; generalisation to other platforms or languages is not validated.
- Sarcasm, quoting, and reclaimed language remain hard classification cases for all current models.
- The dataset reflects 2017 Twitter norms; hate speech language evolves and the model may not capture newer patterns without retraining.

---
