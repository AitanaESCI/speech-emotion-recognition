# Speech Emotion Recognition — Meeting Audio Classifier

**UPC AIDL Final Project**

**Team**: Angelica Tacca Dughetti, Andrea Sanchez, Aitana Diaz, Diego Alonso

**Supervisor**: Pol Caselles

## 🔗 Demo
- [Hugging Face Space](https://huggingface.co/spaces/aitanadiaz/speech-emotion-recognition) — Live demo
- [Emo Tuning - Labeling Tool](https://upc-emo-tuning.pages.dev/)

## 📋 Table of Contents
1. [Motivation](#1-motivation)
2. [Proposal](#2-proposal)
3. [Setup & Reproducibility](#3-setup--reproducibility)
4. [Experiments](#4-experiments)
5. [Results Summary](#5-results-summary)
6. [Conclusions & Future Work](#6-conclusions--future-work)

---

## 1. Motivation

<p align="justify">
Understanding emotion in spoken communication is a key building block for applications ranging from meeting analytics to mental health monitoring and human-computer interaction. While text-based sentiment analysis is widely studied, vocal tone carries emotional information that text alone cannot capture: the same sentence can convey frustration, sarcasm, or enthusiasm purely through how it's spoken.

This project focuses on **Speech Emotion Recognition (SER) for meeting audio**: given a short audio clip, classify the speaker's emotional state. Meetings are a particularly useful domain speech is often overlapping, informal, and emotionally subtle, unlike the acted, exaggerated emotion in many benchmark datasets. Successfully tackling this could enable tools that flag team friction, measure engagement, or summarize meeting tone automatically.

We chose this project because it combines a real audio signal processing challenge (extracting meaningful features from raw waveforms) with a classic deep learning problem: class imbalance, speaker-independent generalization, and the trade-off between model capacity and overfitting on a moderately sized dataset.

## 2. Proposal

**Architecture**
We use a CNN-based image classification approach applied to audio: raw waveforms are converted into log-mel spectrograms, which are then fed into a 2D convolutional classifier: first a custom architecture, later a fine-tuned EfficientNet-B0 backbone. 
Full architectural details and iterations are documented experiment-by-experiment in [Section 4](#4-experiments).

**Data**
- Dataset: [IEMOCAP](https://sail.usc.edu/iemocap/) -> ~10,000 audio clips from 10 actors across 5 sessions
- Original 9 emotion labels merged into 4 balanced classes: Anger, Happiness, Sadness, Neutral
- Speaker-independent train/validation/test split by session, to prevent speaker leakage and ensure the model generalizes to unseen voices

**Computational Requirements** (TODO review)
- Feature extraction: log-mel spectrograms (n_mels=80, n_fft=1024, hop_length=160, win_length=400)
- Training: single GPU (Google Colab), experiments ranging from 30 to 100 epochs depending on architecture
- For WavLM a A100 Computer power is suggested, but will take several hours.
- Experiment tracking: [Weights & Biases](https://wandb.ai/)

## 3. Setup & Reproducibility

### Repository Structure
```
├── demo-apps/
│   ├── hf_space/  # Hugging Face demo app
│   └── emo_tuning/  # Data labeling tool
│
└── train-cli/
    ├── models/  # Experiments models
    ├── train.py
    └── README.md  # Train CLI instructions.
```

### Train CLI tool

We developed a command line interface (CLI) tool to train our models. The tool is built using the uv package manager and allowed us to train our models with different configurations.

We kept track of our experiments on [Weights & Biases](https://wandb.ai/). Additionally we included an option to run our experiments in detached execution mode using [Modal](https://modal.com/) to take advantage of their remote GPU resources.

> [!TIP]
> See more details about these tools in our [Train CLI - README](train-cli/README.md).

#### Installation
```bash
cd train-cli/
# Requires uv package manager: https://docs.astral.sh/uv/
uv sync
```

#### Running experiments

Example of using our CLI tool to train a `cnn_lstm` model for 50 epochs with:
- Dropout rate of 0.1
- Pitch shift augmentation probability of 0.3
- [SpecAugment](https://research.google/blog/specaugment-a-new-data-augmentation-method-for-automatic-speech-recognition/) enabled

```bash
cd train-cli/
uv run python train.py --model-name simple_cnn --epochs 10 --dropout 0.3 --disable-spec-augment # Experiment 1 — Baseline CNN (9 classes)
uv run python train.py --model-name simple_cnn --epochs 15 --dropout 0.3 --enable-spec-augment # Experiment 2 — CNN + weighted loss + SpecAugment
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.3 --enable-spec-augment # Experiment 3 — CNN, 4-class merge
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.5 --pitch-shift-prob 0.2 --enable-spec-augment # Experiment 6/7/8 — EfficientNet-B0 (best CNN model)
uv run python train.py --model-name cnn_gru --epochs 50 --dropout 0.1 --pitch-shift-prob 0.3 --enable-spec-augment # Experiment 9 — CNN + GRU
uv run python train.py --model-name cnn_lstm --epochs 50 --dropout 0.1 --pitch-shift-prob 0.3 --enable-spec-augment # Experiment 9 — CNN + LSTM
uv run python train.py --model-name transcript_only --epochs 20 --dropout 0.1 # Experiment 10 — Transcript only (text baseline)
uv run python train.py --model-name gemma_audio --epochs 15 --dropout 0.1 # Experiment 10 — Gemma Audio
uv run python train.py --model-name wavlm_only --epochs 15 --dropout 0.1 --batch-size 16 # Experiment 11 — WavLM audio only 
uv run python train.py --model-name wavlm_and_transcript --epochs 15 --dropout 0.1 --batch-size 16 # Experiment 12 — WavLM + transcript (multimodal)

```

### Reproducing our best model(s)
We report the best model in each family:

- **Best from-scratch / CNN model**, **Experiment 7** (EfficientNet-B0, frozen layers,
  dropout 0.5, pitch shift augmentation): Test Acc 0.536, Macro F1 0.502.
- **Best pretrained model (and final selected model)**, **Experiment 11** (WavLM Large,
  audio only, base frozen, top 6 layers + MLP head fine-tuned): Test Acc 0.641, Macro F1 0.603.
  WavLM multimodal (audio + text, Experiment 12) scored slightly higher (0.656 / 0.619), but
  the text gain was not statistically significant (p = 0.095), so we keep the simpler
  audio-only model as final.

See [Section 4](#4-experiments) for full configurations.

___

## 4. Experiments

### Experiment 1: Baseline CNN (9 classes)

**Hypothesis**

A simple CNN trained directly on the original 9 emotion classes would establish a baseline, but we expected class imbalance to limit performance on minority emotions.

**Setup**
- Architecture: SimpleCNN (3 conv blocks, BatchNorm, MaxPool)
- Classes: 9 (original IEMOCAP emotions)
- Epochs: 10 | Dropout: 0.3 | Loss: CrossEntropy (unweighted) | Scheduler: None
- No SpecAugment, no skip connections

**Results**
| Metric | Value |
|---|---|
| Val Accuracy | 0.376 |
| Test Accuracy | 0.36 |
| Test Macro F1 | 0.282 |

**Conclusions**
The low macro F1 relative to accuracy confirmed our suspicion: class imbalance was hurting minority emotion classes. This motivated weighting the loss function and merging rare classes in subsequent experiments.

---

### Experiment 2: Weighted Loss + SpecAugment + ReduceLROnPlateau

**Hypothesis**

Adding class weighting and data augmentation would improve generalization and boost performance on underrepresented emotions.

**Setup**
- Same architecture as Exp 1
- Classes: 9 | Epochs: 15 | Dropout: 0.3
- Loss: Weighted CrossEntropy | Scheduler: ReduceLROnPlateau | SpecAugment: Yes

**Results**
| Metric | Value |
|---|---|
| Val Accuracy | 0.295 |
| Test Accuracy | 0.31 |
| Test Macro F1 | 0.27 |

**Conclusions**
Counter-intuitively, results were *worse* than the baseline. With 9 highly imbalanced classes, aggressive weighting combined with SpecAugment likely added too much noise relative to the signal available per class. This suggested the real fix was not weighting alone, but reducing the number of classes.

---

### Experiment 3: Class Merging (9 → 4 classes)

**Hypothesis**

Merging the 9 original emotions into 4 broader, better-balanced classes (anger, happiness, sadness, neutral) would give the model enough samples per class to learn meaningful patterns.

**Setup**
- Same architecture as Exp 1-2
- Classes: 4 | Epochs: 20 | Dropout: 0.3
- Loss: Sqrt-weighted CrossEntropy | Scheduler: ReduceLROnPlateau | SpecAugment: Yes

**Results**
| Metric | Value |
|---|---|
| Val Accuracy | 0.451 |
| Test Accuracy | 0.49 |
| Test Macro F1 | 0.451 |

**Conclusions**
A substantial jump in every metric, validating that class imbalance (not model capacity) was the main bottleneck. This became our new baseline for all subsequent architecture experiments.

---

### Experiment 4: Skip Connections

**Hypothesis**

Adding residual connections would help gradients flow through deeper layers, allowing the model to learn richer features without degradation.

**Setup**
- Architecture: EmotionCNN with 3 residual blocks (Conv→BN→ReLU→Conv→BN + shortcut)
- Classes: 4 | Epochs: 20 | Dropout: 0.0
- Loss: Sqrt-weighted CrossEntropy | Scheduler: ReduceLROnPlateau | SpecAugment: No

**Results**
| Metric | Value |
|---|---|
| Val Accuracy | 0.477 |
| Test Accuracy | 0.52 |
| Test Macro F1 | 0.487 |

**Conclusions**
Skip connections improved both accuracy and macro F1, confirming that a deeper residual architecture could extract better features than the plain CNN at this data scale.

---

### Experiment 5: Reduced Dropout

**Hypothesis**

With skip connections stabilizing training, the original 0.3 dropout might be overly conservative: reducing it could let the model use its full capacity.

**Setup**
- Same residual architecture as Exp 4
- Classes: 4 | Epochs: 20 | Dropout: 0.1
- Loss: CrossEntropy | Scheduler: ReduceLROnPlateau | SpecAugment: No

**Results**
| Metric | Value |
|---|---|
| Val Accuracy | 0.497 |
| Test Accuracy | 0.52 |
| Test Macro F1 | 0.485 |

**Conclusions**
Validation accuracy improved slightly, though test macro F1 stayed roughly flat. This indicated we were near the ceiling of what this architecture could achieve without further regularization or a stronger backbone.

---

### Experiment 6: EfficientNet-B0 (Transfer Learning)

**Hypothesis**

A pretrained EfficientNet-B0 backbone (ImageNet weights) would outperform our custom CNN by leveraging general-purpose visual features, even though spectrograms differ from natural images.

**Setup**
- Architecture: EfficientNet-B0, first conv adapted from 3→1 channel (averaged RGB weights), dataset `id=2` (improved class merging + padding)
- Classes: 4 | Epochs: 30 | Dropout: 0.1
- Loss: CrossEntropy | LR: 1e-4 (no scheduler) | SpecAugment: Yes

**Results**
| Metric | Value |
|---|---|
| Val Accuracy | 0.51 |
| Test Accuracy | 0.53 |
| Test Macro F1 | 0.49 |

**Conclusions**
Best result so far, but training curves showed clear overfitting: training accuracy kept climbing while validation plateaued and oscillated after epoch 3. This meant the gain was not fully reliable and needed regularization.

---

### Experiment 7: EfficientNet-B0, Frozen Layers + Higher Dropout + Pitch Shift

**Hypothesis**

Freezing more of the pretrained backbone, increasing dropout, and adding pitch shift augmentation would reduce overfitting while preserving the transfer learning benefit.

**Setup**
- Same EfficientNet-B0 backbone, fewer layers unfrozen
- Classes: 4 | Epochs: 30 | Dropout: 0.5 | Batch size: 32
- Loss: CrossEntropy | Scheduler: CosineAnnealingLR | SpecAugment: Yes | Pitch shift p: 0.2

**Results**
| Metric | Value |
|---|---|
| Val Accuracy | 0.5106 |
| Test Accuracy | 0.536 |
| Test Macro F1 | 0.502 |

**Conclusions**
This solved the overfitting problem from Experiment 6 (training and validation curves tracked much more closely), while also achieving our best test macro F1. This became our final selected model.

---

### Experiment 8: EfficientNet-B0 Manual Class Weights

**Hypothesis**

Replacing automatic sqrt-weighting with manually tuned class weights (emphasizing harder classes) might further improve performance on underperforming classes like Neutral and Positive.

**Setup**
- Same architecture/config as Exp 7
- Loss: CrossEntropy with manual weights `[1.0, 2.5, 2.5, 3.5]`

**Results**
| Metric | Value |
|---|---|
| Val Accuracy | 0.503 |
| Test Accuracy | 0.504 |
| Test Macro F1 | 0.478 |

**Conclusions**
Overfitting remained solved, but performance slightly regressed compared to Exp 7. Manual weights didn't translate into better discrimination. Neutral and Positive classes still underperformed, suggesting the issue is more about feature separability than loss weighting. We kept **Experiment 7** as our final model.

---

### Experiment 9: CNN + GRU/LSTM

**Hypothesis**
A recurrent layer (GRU/LSTM) on top of the CNN would capture the timing and rhythm of
speech that a static spectrogram CNN misses.

**Setup**
- Architecture: CNN feature extractor + GRU/LSTM head
- Classes: 4 | Loss: sqrt-weighted CrossEntropy + Focal

**Results**
| Metric | Value |
|---|---|
| Test Accuracy | 0.478 |
| Test Macro F1 | 0.459 |

**Conclusions** The dataset was too small for recurrence to help, and aggressive pooling
before the GRU left too little context to use. It did not beat the plain CNN.

---

### Experiment 10: Pretrained speech backbones, Wav2Vec2 & HuBERT (audio only)

**Hypothesis**
A model pretrained on large amounts of speech would transfer richer audio features
(tone, prosody) than a CNN trained from scratch on spectrograms.

**Setup**
- Architecture: Wav2Vec2 and HuBERT, partial fine-tuning, raw waveform input
- Classes: 4 | Loss: soft-label CrossEntropy

**Results**
| Model | Test Accuracy | Test Macro F1
|---|---|---|
| Wav2Vec2 | 0.510 | 0.494 |
| HuBERT | 0.565 | 0.554 |

**Conclusions**: Both beat the from-scratch CNN and EfficientNet, pretraining on speech (not images) was the biggest lever so far. HuBERT clearly outperformed Wav2Vec2.

---

### Experiment 11: WavLM (audio only) 

**Hypothesis**
WavLM, pretrained self-supervised on ~94k hours of speech, would capture tone and prosody
better than any previous model.

**Setup**
- Architecture: WavLM Large (24 layers, 316M params). Freeze the base, fine-tune the top 6 layers + a small MLP head. Mean + std pooling over time.
- Loss: soft CrossEntropy on soft labels | Classes: 4
- Training: ~15 epochs, ~5h on a Colab A100 GPU


**Results**
| Metric | Value |
|---|---|
| Test Accuracy | 0.641 |
| Test Macro F1 | 0.603 |


**Conclusions**: Best model of the whole project (~+0.09 macro-F1 over from-scratch models). Large speech pretraining plus light fine-tuning gave the biggest jump. This is our final selected model.

---


### Experiment 12: WavLM Multimodal (audio + text)

**Hypothesis**
Adding the text transcript (what was said) to the audio (how it was said) would give extra
signal and improve results.

**Setup**
- Architecture: WavLM audio branch + MiniLM text encoder on the transcript, fused before the classifier head. Same training regime as Experiment 11.

**Results**
| Metric | Value |
|---|---|
| Test Accuracy | 0.656 |
| Test Macro F1 | 0.619 |

**Conclusions**
Multimodal was almost the same as audio-only. The text gain was not
statistically significant (p = 0.095). This is a real finding, not a failure: for emotion,
most of the signal lives in the audio, not the words.


---

## 5. Results Summary

| # | Experiment | Classes | Val Acc | Test Acc | Test Macro F1 | Notes |
|---|---|---|---|---|---|---|
| 1 | Baseline CNN | 9 | 0.376 | 0.36 | 0.282 | First experiment |
| 2 | Weighted loss + SpecAug | 9 | 0.295 | 0.31 | 0.27 | Worse than baseline |
| 3 | 4-class merge | 4 | 0.451 | 0.49 | 0.451 | Key turning point |
| 4 | Skip connections | 4 | 0.477 | 0.52 | 0.487 | |
| 5 | Reduced dropout | 4 | 0.497 | 0.52 | 0.485 | |
| 6 | EfficientNet-B0 | 4 | 0.51 | 0.53 | 0.49 | Best so far, overfitting |
| 7 | EfficientNet-B0 + frozen layers + pitch shift | 4 | 0.5106 | 0.536 | **0.502** | Overfitting solved |
| 8 | EfficientNet-B0 + manual weights | 4 | 0.503 | 0.504 | 0.478 | Didn't outperform Exp 7 |
| 9 | CNN + GRU/LSTM | 4 | - | 0.478 | 0.459 | Too little data |
| 10 | HuBERT (audio only) | 4 | - | 0.565 | 0.554 | Pretraining helps a lot|
| 10 | Wav2Vec2 (audio only) | 4 | - | 0.510 | 0.494 | Pretrained speech |
| 11 | WavLM (audio only) | 4 | - | 0.641 | 0.603 | Best pre trained model|
| 12 | WavLM multimodal (audio+text) | 4 | - | 0.656 | 0.619 | Text not significant |


## 6. Conclusions & Future Work

**What we learned**
- Class imbalance was the single biggest bottleneck early on: merging 9 emotions into 4 balanced classes had a bigger impact than any architectural change.
- Adding capacity (skip connections, EfficientNet transfer learning) helped, but only once paired with appropriate regularization: naive transfer learning overfit quickly on a dataset this size.
- Manually tuned loss weights did not outperform principled automatic weighting (sqrt-weighting), reinforcing that the harder classes (Neutral, Positive) suffer more from feature overlap than from data scarcity alone.
- Speaker-independent evaluation is essential for an honest performance estimate and makes this a genuinely harder problem than speaker-dependent benchmarks suggest.
- On small data, pretraining wins. WavLM beat every model trained from scratch. The single biggest jump came from switching to a speech-pretrained backbone, not from architecture tweaks.
- Data quality is the ceiling. Training accuracy (0.75) was only a little above validation(0.65) a small gap, so the model is not overfitting; the labels themselves are noisy. The last rounds of tuning only moved results by 1–2 points.
- Multimodality needs a reason. Adding the text transcript did not significantly help. For emotion, the signal lives in the audio (tone/prosody), not the words.

**Limitations**
- Final macro F1 (~0.6) shows the model still struggles to separate Neutral and Positive emotions specifically.
- The demo app's diarization component is not fully functional yet.
- Performance was evaluated only on IEMOCAP (acted emotion); real meeting audio is likely more subtle and may not transfer perfectly.

**Future Work**
- Fix and properly evaluate the speaker diarization component for multi-speaker meeting audio.
- Collect or fine-tune on more naturalistic (non-acted) emotional speech data.
</p>
