# Speech Emotion Recognition, Meeting Audio Classifier

**UPC AIDL Final Project**

**Team**: Angelica Tacca Dughetti, Andrea Sanchez, Aitana Diaz, Diego Alonso

**Supervisor**: Pol Caselles

## 🔗 Demo
- [Hugging Face Space](https://huggingface.co/spaces/angietd/meeting-emotion-demo) - Live demo
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

This project focuses on **Speech Emotion Recognition (SER) for meeting audio**: given a short audio clip, classify the speaker's emotional state. Meetings are a particularly useful domain, speech is often overlapping, informal, and emotionally subtle, unlike the acted, exaggerated emotion in many benchmark datasets. Successfully tackling this could enable tools that flag friction, measure engagement, or summarize meeting tone automatically.

This combines a real audio signal processing challenge (extracting meaningful features from raw waveforms) with a classic deep learning problem: class imbalance, speaker-independent generalization, and the trade-off between model capacity and overfitting on a moderately sized dataset.

</p>

## 2. Proposal

**Architecture**

A CNN-based image classification approach applied to audio: raw waveforms are converted into log-mel spectrograms, which are then fed into a 2D convolutional classifier, first a custom architecture, later a fine-tuned EfficientNet-B0 backbone, and finally a fine-tuned WavLM audio encoder, which became the best-performing and final selected model.
Full architectural details and iterations are documented category-by-category in [Section 4](#4-experiments).

**Data**
- Dataset: [IEMOCAP](https://sail.usc.edu/iemocap/), ~10,000 audio clips from 10 actors across 5 sessions
- Original 9 emotion labels merged into 4 balanced classes: Neutral, Negative, Positive, Unclear
- Speaker-independent train/validation/test split by session, to prevent speaker leakage and ensure the model generalizes to unseen voices

**Computational Requirements**
- Feature extraction: log-mel spectrograms (n_mels=80, n_fft=1024, hop_length=160, win_length=400)
- Training: single GPU (Google Colab), experiments ranging from 10 to 100 epochs depending on architecture
- For WavLM, an A100-class GPU is recommended; training still takes several hours
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

A command line interface (CLI) tool handles training. The tool is built using the uv package manager and supports training with different configurations.

Experiments are tracked on [Weights & Biases](https://wandb.ai/). An option to run experiments in detached execution mode using [Modal](https://modal.com/) is also included, to take advantage of their remote GPU resources.

> [!TIP]
> See more details about these tools in the [Train CLI - README](train-cli/README.md).

#### Installation
```bash
cd train-cli/
# Requires uv package manager: https://docs.astral.sh/uv/
uv sync
```

#### Running experiments

```bash
cd train-cli/
uv run python train.py --chunks-group-id 1 --model-name simple_cnn --epochs 10 --dropout 0.3 --disable-spec-augment # Category 1: Baseline CNN (9 classes)
uv run python train.py --model-name simple_cnn --epochs 15 --dropout 0.3 --enable-spec-augment # Category 1: Baseline CNN + SpecAugment (9 classes)
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.3 --enable-spec-augment # Category 2: 4-class grouping
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.3 --enable-spec-augment # Category 3: Class weights + Focal Loss
uv run python train.py --model-name simple_cnn --epochs 50 # Category 4: Fixed epoch budget (no early stop)
uv run python train.py --model-name cnn_gru --epochs 100 # Category 5: CNN + GRU/LSTM
uv run python train.py --model-name cnn_gru --epochs 100 --pitch-shift-prob 0.3 --enable-spec-augment # Category 6: SpecAugment + Pitch Shift
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.5 --pitch-shift-prob 0.2 --enable-spec-augment # Category 7: Pretrained backbones, EfficientNet-B0
uv run python train.py --model-name wavlm_only --epochs 15 --dropout 0.1 --batch-size 16 # Category 7: Pretrained backbones, WavLM audio only
uv run python train.py --model-name transcript_only --epochs 20 --dropout 0.1 # Category 8: Text only
uv run python train.py --model-name wavlm_and_transcript --epochs 15 --dropout 0.1 --batch-size 16 # Category 8: WavLM multimodal (audio + text)
```

### Reproducing the best model

Best pretrained model and final selected model: WavLM Large, audio only, top 6 layers fine-tuned: Test Acc 0.641, Test Macro-F1 0.603.
WavLM multimodal (audio + text) scored slightly higher (0.656 / 0.619), but the text gain was not statistically significant (p = 0.095), so the simpler audio-only model was kept as final.

See [Section 4](#4-experiments) for full configurations.

---

## 4. Experiments

Experiments are grouped into the 8 categories below, each answering the question the previous one raised. Reproduction commands use the unified `train-cli` tool (`train-cli/train.py`).

**Shared preprocessing pipeline (applies to every category below unless noted):**
- Soft labels, a probability spread across classes rather than a single hard label
- Re-audited ground truth, clips flagged as low-quality by Whisper + Inworld were excluded
- Speaker-independent evaluation, split by session, to avoid speaker leakage between train/val/test
- **Metric of record: Macro-F1** (not raw accuracy), accuracy and per-setup Val/Test Acc are reported alongside for reference, but Macro-F1 is what results are compared against

### Category 1: Baseline CNN + SpecAug

**Hypothesis**

A simple CNN on the original 9 IEMOCAP classes gives an initial reference point. Weighting the loss and adding SpecAugment on top were tested as a first attempt at fixing class imbalance.

**Setup**

| Config | Classes | Details |
| --- | --- | --- |
| Baseline CNN | 9 | 10 epochs, Dropout 0.3, CrossEntropy, no SpecAugment |
| Baseline CNN + weighted loss + SpecAugment | 9 | 15 epochs, Dropout 0.3, weighted CrossEntropy, SpecAugment: Yes |

**Results**

| Config | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Baseline CNN | 9 | 0.376 | 0.36 | 0.282 | First configuration tested |
| + weighted loss + SpecAugment | 9 | 0.295 | 0.31 | 0.27 | Worse than baseline |

**Conclusions**

Weighting the loss and adding SpecAugment on top of 9 fine-grained classes made results worse, not better. The classes were too granular and ambiguous for loss weighting alone to fix. This pointed to class granularity, not loss weighting, as the real bottleneck at this stage.

**Reproduce (current CLI)**

```bash
uv run python train.py --chunks-group-id 1 --model-name simple_cnn --epochs 10 --dropout 0.3 --disable-spec-augment
uv run python train.py --model-name simple_cnn --epochs 15 --dropout 0.3 --enable-spec-augment
```

---

### Category 2: 4-class grouping

**Hypothesis**

Grouping the original 9 IEMOCAP emotions into 4 broader, better-balanced classes (Neutral, Negative, Positive, Unclear) was tested as a more structural fix for label ambiguity and imbalance.

**Setup**

| Config | Classes | Details |
| --- | --- | --- |
| 4-class grouping | 4 | 20 epochs, Dropout 0.3, CrossEntropy, SpecAugment: Yes |

**Results**

| Config | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| 4-class grouping | 4 | 0.451 | 0.49 | 0.451 | Key turning point |

**Conclusions**

Macro-F1 jumped from 0.28 to 0.45, the biggest single gain in the whole project, larger than any architectural change made afterward. This confirmed class granularity, not model capacity, was the initial bottleneck, and this configuration became the shared baseline going forward.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.3 --enable-spec-augment
```

---

### Category 3: Class weights + Focal Loss

**Hypothesis**

On top of the 4-class setup, per-class loss weighting and Focal Loss were tested as ways to address the remaining imbalance and the ambiguity between classes such as Positive vs Neutral.

**Setup**

| Config | Classes | Details |
| --- | --- | --- |
| Manual class weights | 4 | `[1.0, 2.5, 2.5, 3.5]`, CrossEntropy, 30 epochs, Dropout 0.5 |
| Sqrt-weighted + Focal Loss (γ=1.5) | 4 | CrossEntropy → FocalLoss, 15 epochs, Dropout 0.3, SpecAugment: Yes |

**Results**

| Config | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Manual weights (CrossEntropy) | 4 | 0.503 | 0.504 | 0.478 | |
| Sqrt-weighted + Focal Loss (γ=1.5) | 4 | 0.501 | 0.501 | 0.514 | Best in this category |

**Conclusions**

Manually tuned weights did not outperform principled, automatic weighting. Sqrt-weighted Focal Loss improved Macro-F1 by targeting ambiguous examples directly. The Positive class still kept the weakest recall, pointing to acoustic confusability rather than data scarcity. This configuration was adopted for the final model.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.3 --enable-spec-augment
```
> Note: `train.py` currently exposes CrossEntropy-style weighting via the dataset/loss config rather than a CLI flag; Focal Loss (γ=1.5) and sqrt-weighting are set in `train-cli/models/`.

---

### Category 4: Fixed epoch budget (no early stop)

**Hypothesis**

An aggressive early-stopping patience was suspected of hiding the model's real behavior, stopping training at the first sign of stagnation instead of letting it run its course. Right after Category 3, gains started looking smaller than expected given how much the loss function had improved, which raised the question of whether each model's real ceiling was even being observed. Once the training regime was fixed, two independent architecture tweaks were tested on the 4-class CNN under the corrected budget: residual connections, and simply reducing dropout to let the model use more of its capacity.

**Setup**

| Config | Classes | Details |
| --- | --- | --- |
| Baseline patience | - | `patience_stop=15` |
| Fixed epoch budget | - | 50 fixed epochs, patience = epoch count, effectively disabling early stopping |
| Skip connections | 4 | 20 epochs, Dropout 0.0, CrossEntropy, residual blocks (Conv→BN→ReLU→Conv→BN + shortcut) |
| Reduced dropout | 4 | 20 epochs, Dropout 0.1, CrossEntropy, SpecAugment: Yes |

**Results**

| Config | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| Skip connections | 4 | 0.477 | 0.52 | 0.487 | |
| Reduced dropout | 4 | 0.497 | 0.52 | 0.485 | About equal to skip connections |

No isolated ablation row exists for the early-stopping change itself, it surfaced as a qualitative finding during review rather than a dedicated experiment run. A before/after comparison recovered roughly 1-2 UA points once training ran the full epoch budget.

**Conclusions**

A meaningful chunk of underperformance traced back not to architecture or augmentation, but to `patience_stop=15` silently cutting training short before convergence. Once a CosineAnnealingLR schedule was committed to instead, patience was removed entirely to reduce complexity, rather than adding it back to the CLI. This fixed 50-epoch budget was adopted for every category from this point forward. Under that corrected budget, skip connections and reduced dropout gave a modest, roughly equivalent bump over the 4-class baseline, landing in the same range as each other. Neither clearly beat the other, suggesting the ceiling of what a from-scratch CNN of this size could do on this dataset was close, which motivated testing a pretrained backbone next.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name simple_cnn --epochs 50
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.0 --enable-spec-augment
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.1 --enable-spec-augment
```
> The current `train.py` has no `--patience` or early-stopping flag: training always runs for the full `--epochs` budget.

---

### Category 5: CNN + GRU/LSTM

**Hypothesis**

Adding a recurrent layer (GRU or LSTM) on top of the CNN's feature maps was tested to see whether it captures the temporal, prosodic dynamics of speech better than a purely convolutional model.

**Setup**

| Config | Classes | Details |
| --- | --- | --- |
| CNN + GRU | 4 | 100 epochs, Dropout 0.1, CrossEntropy, no pitch shift, no SpecAugment |

**Results**

| Config | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| CNN + GRU | 4 | ~0.532* | 0.478 | 0.459 | Too little data |

*Val Acc taken from the SpecAugment + Pitch Shift variant (Category 6) as an approximation; no Val Acc was recorded for this exact 100-epoch, no-augmentation configuration.

**Conclusions**

The recurrent layer underperformed the plain CNN. With a dataset the size of IEMOCAP, there wasn't enough data for the GRU to learn useful temporal patterns on top of the CNN features, it added training time and complexity without a corresponding gain.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name cnn_gru --epochs 100
```

---

### Category 6: SpecAugment + Pitch Shift

**Hypothesis**

The overfitting seen with recurrent architectures raised a follow-up question: with a dataset as limited as IEMOCAP, would augmenting the spectrogram (SpecAugment) and randomizing pitch improve generalization directly?

**Setup**

A transversal ablation rather than a standalone model, isolated by comparing paired runs of the same architecture with only the augmentation flags changed:
- CNN + GRU with `--enable-spec-augment --pitch-shift-prob 0.3` vs. without
- EfficientNet-B0 with `--pitch-shift-prob 0.2` vs. `--pitch-shift-prob 0`

**Results**

| Config | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| CNN + GRU + SpecAugment + Pitch Shift | 4 | 0.532 | 0.52 | 0.465 | Overfitting reduced vs. Category 5 |

SpecAugment measurably reduced overfitting on the recurrent architecture (Category 5). Pitch shift as a standalone hyperparameter gave mixed results and was not conclusively better across runs.

**Conclusions**

Augmentation helped in architectures already prone to overfitting, but didn't fix the deeper problem: noisy, disagreement-prone labels set a ceiling that augmentation alone can't push past. Delta/delta-delta channels were prioritized as the robustness mechanism in the final model's documented configuration instead.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name cnn_gru --epochs 100 --pitch-shift-prob 0.3 --enable-spec-augment
uv run python train.py --model-name cnn_gru --epochs 100 --pitch-shift-prob 0.0 --disable-spec-augment
```

---

### Category 7: Pretrained backbones (EfficientNet-B0 & WavLM)

**Hypothesis**

Having reached the limits of loss design, class balance, training regime, and augmentation on a from-scratch CNN, the next question was how far behind that left things: would a backbone pretrained on a large external corpus, EfficientNet-B0 (ImageNet) or WavLM (large audio corpora), transfer richer representations than anything trained from scratch on IEMOCAP alone?

**Setup**

| Config | Classes | Details |
| --- | --- | --- |
| EfficientNet-B0 | 4 | 30 epochs, Dropout 0.1, CrossEntropy, SpecAugment: Yes |
| EfficientNet-B0, frozen layers + pitch shift | 4 | 30 epochs, Dropout 0.5, `--pitch-shift-prob 0.2`, SpecAugment: Yes |
| WavLM (audio only) | 4 | 15 epochs, Dropout 0.1, batch size 16, top 6 layers fine-tuned |

**Results**

| Config | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| EfficientNet-B0 | 4 | 0.51 | 0.53 | 0.49 | Best so far, overfitting |
| EfficientNet-B0, frozen layers + pitch shift | 4 | 0.5106 | 0.536 | 0.502 | Overfitting solved, best CNN-based configuration |
| WavLM (audio only) | 4 | 0.631 | 0.641 | 0.603 | Best model overall |

**Conclusions**

EfficientNet-B0 beat every from-scratch CNN configuration, confirming pretraining had real value even from an unrelated domain (vision). Freezing layers and adding pitch shift fixed its overfitting. WavLM, an audio-native encoder, went further still and became the best-performing configuration in the whole project, comfortably ahead of both the from-scratch CNN and EfficientNet-B0. WavLM audio-only is the final selected model.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.1 --enable-spec-augment
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.5 --pitch-shift-prob 0.2 --enable-spec-augment
uv run python train.py --model-name wavlm_only --epochs 15 --dropout 0.1 --batch-size 16
```
> Note: the current `train.py` fine-tunes `microsoft/wavlm-large`'s top 6 layers by default; a fully frozen encoder variant was also tested but isn't exposed as a CLI flag.

---

### Category 8: Text Only + Multimodal

**Hypothesis**

With WavLM confirming pretrained audio alone as the strongest lever, the remaining open question was whether adding another modality, transcribed text alone or alongside audio, could push results further still.

**Setup**

| Config | Classes | Details |
| --- | --- | --- |
| Transcript only | 4 | 20 epochs, Dropout 0.1 |
| WavLM multimodal (audio + text) | 4 | 15 epochs, Dropout 0.1, batch size 16 |

**Results**

| Config | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| WavLM multimodal (audio + text) | 4 | 0.653 | 0.656 | 0.619 | Slightly higher than audio-only, not statistically significant |

**Conclusions**

Multimodal performance came out roughly on par with audio-only. A dedicated ablation showed text was not statistically significant (t-test p = 0.095), the model largely ignores the text channel. Text was a finding, not a failure: for this task, the signal lives mostly in the audio, not the words. The simpler audio-only configuration (Category 7) was kept as final.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name transcript_only --epochs 20 --dropout 0.1
uv run python train.py --model-name wavlm_and_transcript --epochs 15 --dropout 0.1 --batch-size 16
```

## 5. Results Summary

| # | Category | Best Test Acc | Best Test Macro-F1 |
| --- | --- | --- | --- |
| 1 | Baseline CNN + SpecAug | 0.36 | 0.282 |
| 2 | 4-class grouping | 0.49 | 0.451 |
| 3 | Class weights + Focal Loss | 0.504 | 0.514 |
| 4 | Fixed epoch budget (no early stop) | 0.52 | 0.487 |
| 5 | CNN + GRU/LSTM | 0.478 | 0.459 |
| 6 | SpecAugment + Pitch Shift | 0.52 | 0.465 |
| 7 | Pretrained backbones (EfficientNet-B0 & WavLM) | 0.641 | 0.603 |
| 8 | Text Only + Multimodal | 0.656 | 0.619 |

**Cross-cutting takeaways (not tied to a single category)**

- Speaker-independent (LOSO) evaluation is essential for an honest estimate, a genuinely harder setup than speaker-dependent benchmarks suggest.
- Manually tuned class weights consistently underperformed automatic or principled weighting across every architecture tested (Category 3).

## 6. Conclusions & Future Work

**What was learned**
- Class granularity, not loss weighting, was the real early bottleneck: merging 9 emotions into 4 balanced classes had a bigger impact than weighting the loss on the original 9 classes, which made things worse.
- Pretraining consistently outperformed training from scratch, first with EfficientNet-B0 (a vision model repurposed for spectrograms), then even more clearly with WavLM (an audio-native encoder). The single biggest jump in the whole project came from switching to a speech-pretrained backbone, not from architecture tweaks.
- Manually tuned loss weights did not outperform principled automatic weighting, consistent with the harder classes (Neutral, Positive) suffering more from feature overlap than from data scarcity alone.
- Speaker-independent evaluation is essential for an honest performance estimate and makes this a genuinely harder problem than speaker-dependent benchmarks suggest.
- Data quality sets a ceiling. Training accuracy (0.75) was only a little above validation accuracy (0.65) on the best model, a small gap, so the model isn't overfitting, the labels themselves are noisy.
- Multimodality needs a reason. Adding the text transcript did not significantly help. For emotion, the signal lives mostly in the audio (tone/prosody), not the words.

**Limitations**
- Final Macro-F1 (~0.6) shows the model still struggles to separate Neutral and Positive emotions specifically.
- The demo app's diarization component is not fully functional yet.
- Performance was evaluated only on IEMOCAP (acted emotion); real meeting audio is likely more subtle and may not transfer perfectly.

**Future Work**
- Fix and properly evaluate the speaker diarization component for multi-speaker meeting audio.
- Collect or fine-tune on more naturalistic (non-acted) emotional speech data.
