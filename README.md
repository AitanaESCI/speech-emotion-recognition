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
uv run python train.py --model-name simple_cnn --epochs 10 --dropout 0.3 --disable-spec-augment # Experiment 1: Baseline CNN (9 classes)
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.3 --enable-spec-augment # Experiment 2: 4-class grouping
uv run python train.py --model-name simple_cnn --epochs 15 --dropout 0.3 --enable-spec-augment # Experiment 3: Class weighting + Focal Loss
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.5 --pitch-shift-prob 0.2 --enable-spec-augment # Experiment 7: Pretrained backbones, EfficientNet-B0
uv run python train.py --model-name cnn_gru --epochs 50 --dropout 0.1 --pitch-shift-prob 0.3 --enable-spec-augment # Experiment 5: CNN + GRU
uv run python train.py --model-name cnn_lstm --epochs 50 --dropout 0.1 --pitch-shift-prob 0.3 --enable-spec-augment # Experiment 5: CNN + LSTM
uv run python train.py --model-name transcript_only --epochs 20 --dropout 0.1 # Experiment 8: Transcript only (text baseline)
uv run python train.py --model-name gemma_audio --epochs 15 --dropout 0.1 # Experiment 8: Gemma Audio
uv run python train.py --model-name wavlm_only --epochs 15 --dropout 0.1 --batch-size 16 # Experiment 7: Pretrained backbones, WavLM audio only
uv run python train.py --model-name wavlm_and_transcript --epochs 15 --dropout 0.1 --batch-size 16 # Experiment 8: WavLM + transcript (multimodal)

```

### Reproducing our best model(s)
We report the best model in each family:

- **Best from-scratch / CNN model**, **Experiment 7** (EfficientNet-B0, frozen layers,
  dropout 0.5, pitch shift augmentation): Test Acc 0.536, Macro F1 0.502.
- **Best pretrained model (and final selected model)**, **Experiment 11** (WavLM Large,
  audio only, base frozen, top 6 layers + MLP head fine-tuned): Test Acc 0.641, Macro F1 0.603.
  WavLM multimodal (audio + text) scored slightly higher (0.656 / 0.619), but
  the text gain was not statistically significant (p = 0.095), so we keep the simpler
  audio-only model as final.

See [Section 4](#4-experiments) for full configurations.

___

## 4. Experiments

[#4-experiments](#4-experiments)

*Consolidated from the shared team experiment log (Results sheet), all experiments were run collaboratively as a team, presented here in the order they actually happened, since each one built directly on what the last one taught us. Reproduction commands use the unified `train-cli` tool (`train-cli/train.py`), which replaced the earlier notebook-based workflow. Where an experiment predates the unified CLI and its architecture isn't part of the current `ModelName` enum (`simple_cnn`, `cnn_gru`, `cnn_lstm`, `transcript_only`, `gemma_audio`, `wavlm_only`, `wavlm_and_transcript`, `efficientnet_b0`), that's noted explicitly instead of a fabricated command.*

**Shared preprocessing pipeline (applies to every experiment below unless noted):**
- Soft labels, a probability spread across classes rather than a single hard label
- Re-audited ground truth, clips flagged as low-quality by Whisper + Inworld were excluded
- Speaker-independent evaluation, split by session, to avoid speaker leakage between train/val/test
- **Metric of record: Macro-F1** (not raw accuracy), accuracy and per-setup Val/Test Acc are reported alongside for reference, but Macro-F1 is what we optimize for and compare against

### Experiment 1: Baseline CNN (9 classes)

[#experiment-1-baseline-cnn-9-classes](#experiment-1-baseline-cnn-9-classes)

**Hypothesis**

A simple CNN trained on the original 9 IEMOCAP emotions would establish an initial reference point before any preprocessing or architectural changes.

**Setup**

- Architecture: Simple CNN (3 conv blocks, BatchNorm, MaxPool)
- 9 classes | 10 epochs | Dropout 0.3 | CrossEntropy (unweighted) | No SpecAugment

**Results**

| Setup | Val Acc | Test Acc | Test Macro-F1 |
| --- | --- | --- | --- |
| Baseline (9 classes) | 0.376 | 0.36 | 0.282 |

**Conclusions**

The 9-class baseline confirmed what we suspected: with fine-grained, imbalanced emotion labels, even a reasonable CNN architecture struggles to learn meaningful patterns. This became the reference point every later experiment was measured against. **Not adopted** (superseded by 4-class grouping, Experiment 2).

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name simple_cnn --epochs 10 --dropout 0.3 --disable-spec-augment
```

---

### Experiment 2: 4-Class Grouping (9 → 4 classes)

[#experiment-2-4-class-grouping-9--4-classes](#experiment-2-4-class-grouping-9--4-classes)

**Hypothesis**

Grouping the original 9 IEMOCAP emotions into 4 broader, better-balanced classes (neutral, sad, negative, positive) reduces label ambiguity and imbalance, giving the model enough samples per class to learn meaningful patterns.

**Setup**

- Architecture: Simple CNN (same as Experiment 1)
- 4 classes | 20 epochs | Dropout 0.3 | CrossEntropy | SpecAugment: Yes

**Results**

| Setup | Val Acc | Test Acc | Test Macro-F1 |
| --- | --- | --- | --- |
| 4-class grouping | 0.451 | 0.49 | 0.451 |

**Conclusions**

Class merging was the first unlock: going from 9 to 4 classes lifted Macro-F1 from 0.28 to 0.45, a bigger jump than any architectural change made afterward. This confirmed class imbalance and label granularity, not model capacity, were the initial bottleneck, and became the shared baseline for every experiment that followed. **Adopted for the final model.**

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.3 --enable-spec-augment
```

---

### Experiment 3: Class Weighting + Focal Loss

[#experiment-3-class-weighting--focal-loss](#experiment-3-class-weighting--focal-loss)

**Hypothesis**

Weighting the loss function per class would help with the remaining imbalance, but the confusion between classes (e.g. happy/excited vs. neutral) looked more like label ambiguity than pure scarcity, Focal Loss should handle that better than a weighted CrossEntropy.

**Setup**

- Manual class weights: `[1.0, 2.5, 2.5, 3.5]`, CrossEntropy, 30 epochs, Dropout 0.5
- Sqrt-weighted + Focal Loss (γ=1.5): CrossEntropy → FocalLoss, 15 epochs, Dropout 0.3, SpecAugment: Yes

**Results**

| Setup | Val Acc | Test Acc | Test Macro-F1 |
| --- | --- | --- | --- |
| Manual weights (CrossEntropy) | 0.503 | 0.504 | 0.478 |
| Sqrt-weighted + Focal Loss (γ=1.5) | 0.501 | 0.501 | 0.514 |

**Conclusions**

Manually tuned weights did not outperform principled automatic weighting, switching from inverse-linear to sqrt-weighted was one of the best isolated improvements in the whole project, since linear weights over-correct and over-punish the priority classes. Focal Loss (γ=1.5) further improved Macro-F1 by targeting ambiguous examples directly. Even so, "happy" kept the worst recall of any class: not a data-scarcity problem, but acoustic confusability that survives even Focal Loss. **Adopted for the final model.**

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name simple_cnn --epochs 15 --dropout 0.3 --enable-spec-augment
```
> Note: `train.py` currently exposes CrossEntropy-style weighting via the dataset/loss config rather than a CLI flag; Focal Loss (γ=1.5) and sqrt-weighting are set in `train-cli/models/`, flag this for the team to confirm whether it's worth exposing as `--loss-fn` / `--gamma` options for reproducibility.

---

### Experiment 4: Disabling Early Stopping (Fixed Epoch Budget)

[#experiment-4-disabling-early-stopping-fixed-epoch-budget](#experiment-4-disabling-early-stopping-fixed-epoch-budget)

**Hypothesis**

Right after Experiment 3, gains started looking smaller than expected given how much the loss function had improved, which raised the question of whether we were even seeing each model's real ceiling. An aggressive early-stopping patience should save compute and prevent overfitting, but that same patience was likely hiding the model's real behavior: stopping at the first sign of stagnation instead of waiting for a second or third occurrence meant we never got to see how the model performed past that point.

**Setup**

- Baseline: `patience_stop=15`
- Revised: patience set to the full epoch budget (50 fixed epochs), effectively disabling early stopping

**Results**

No isolated ablation row exists in the shared spreadsheet for this change, it surfaced as a qualitative finding during review, not a dedicated experiment run. In our own before/after comparison, disabling early stopping recovered roughly 1–2 UA points that had previously been lost.

**Conclusions**

We traced a meaningful chunk of underperformance not to architecture or augmentation, but to `patience_stop=15` silently cutting training short before convergence, the model was being judged on its first hiccup rather than its actual trajectory. This directly matched what we'd started to suspect after Experiment 3, so from this point on every experiment in this document ran with a fixed 50-epoch budget and patience = epoch count. **Adopted**, and it's the reason none of the runs in Experiments 5-8 below use early stopping.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name simple_cnn --epochs 50
```
> The current `train.py` has no `--patience` or early-stopping flag at all, training always runs for the full `--epochs` budget. That's a direct consequence of this experiment: the team removed early stopping from the unified pipeline instead of just tuning its patience.

---

### Experiment 5: CNN + GRU / LSTM (Recurrent Architectures)

[#experiment-5-cnn--gru--lstm-recurrent-architectures](#experiment-5-cnn--gru--lstm-recurrent-architectures)

**Hypothesis**

With the training regime fixed (Experiment 4), we could finally trust a "no" result as a real "no" rather than a training artifact. So: would adding a recurrent layer on top of the CNN's feature maps capture the temporal/prosodic dynamics of speech better than a purely convolutional model?

**Setup**

- CNN + GRU: 50 epochs, Dropout 0.1, `--pitch-shift-prob 0.3`, SpecAugment: Yes
- CNN + LSTM: 50 epochs, same hyperparameters as CNN + GRU

**Results**

| Setup | Val Acc | Test Acc | Test Macro-F1 |
| --- | --- | --- | --- |
| CNN + GRU (with SpecAugment) | 0.532 | 0.52 | 0.465 |
| CNN + GRU (without SpecAugment) | 0.52 | 0.50 | 0.455 |
| CNN + LSTM | 0.5157 | 0.505 | 0.446 |

**Conclusions**

Confirmed independently by two team members, and now with the full epoch budget in place: a plain CNN matched or beat CNN+RNN variants. The recurrent layer added ~1 hour of extra training time per run and, without SpecAugment, overfit hard (near-100% memorization by epoch 90), likely because aggressive pooling before the GRU left too little temporal context for it to exploit. **Not adopted.**

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name cnn_gru --epochs 50 --dropout 0.1 --pitch-shift-prob 0.3 --enable-spec-augment
uv run python train.py --model-name cnn_lstm --epochs 50 --dropout 0.1 --pitch-shift-prob 0.3 --enable-spec-augment
```

---

### Experiment 6: SpecAugment & Pitch Shift

[#experiment-6-specaugment--pitch-shift](#experiment-6-specaugment--pitch-shift)

**Hypothesis**

The overfitting we just saw in Experiment 5 raised an obvious follow-up: with a dataset as limited as IEMOCAP, would augmenting the spectrogram (SpecAugment) and randomizing pitch improve generalization and reduce that overfitting directly?

**Setup**

This is a transversal ablation rather than a standalone model, it's isolated by comparing paired runs of the same architecture with only the augmentation flags changed:
- CNN + GRU with `--enable-spec-augment` vs. `--disable-spec-augment` (see Experiment 5 table)
- EfficientNet-B0 with `--pitch-shift-prob 0.2` vs. `--pitch-shift-prob 0`

**Results**

SpecAugment measurably reduced overfitting in the CNN+GRU ablation (Experiment 5): without it, training memorized the data by epoch 90; with it, that failure mode didn't appear. Pitch shift as a standalone hyperparameter gave mixed results and was not conclusively better across runs.

**Conclusions**

Augmentation helped in specific architectures (recurrent models prone to overfitting) but didn't fix the deeper problem: noisy, disagreement-prone labels set a ceiling that augmentation alone can't push past. **Not included** in the final model's documented configuration, we prioritized delta/delta-delta channels as our robustness mechanism instead.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name cnn_gru --epochs 50 --dropout 0.1 --enable-spec-augment --pitch-shift-prob 0.3
uv run python train.py --model-name cnn_gru --epochs 50 --dropout 0.1 --disable-spec-augment --pitch-shift-prob 0.0
```

---

### Experiment 7: Pretrained Backbones, EfficientNet-B0 & WavLM

[#experiment-7-pretrained-backbones--efficientnet-b0--wavlm](#experiment-7-pretrained-backbones--efficientnet-b0--wavlm)

**Hypothesis**

Having pushed the from-scratch CNN as far as loss design, class balance, training regime, and augmentation could take it, the natural next question was how far behind we actually were: would a model pretrained on a large external corpus, EfficientNet-B0 (ImageNet) or WavLM (large audio corpora), transfer richer representations than anything we could train from scratch on IEMOCAP alone?

**Setup**

- EfficientNet-B0 (best CNN-based variant): 30 epochs, Dropout 0.5, `--pitch-shift-prob 0.2`, SpecAugment: Yes
- WavLM audio-only (fine-tuned top layers): 15 epochs, Dropout 0.1, batch size 16

**Results**

| Setup | Val Acc | Test Acc | Test Macro-F1 |
| --- | --- | --- | --- |
| EfficientNet-B0 (unfrozen, id=1) | 0.56 | 0.572 | 0.579 |
| EfficientNet-B0 (unregularized, id=2) | 0.51 | 0.53 | 0.49 |
| EfficientNet-B0 (frozen layers + dropout 0.5 + pitch shift) | 0.5106 | 0.536 | 0.502 |
| WavLM fine-tuned (audio-only, 3-seed avg) | 0.631 | 0.641 | 0.606 |

**Conclusions**

Pretraining was the single biggest lever the team found, WavLM beat every from-scratch architecture from Experiments 1-6 by a wide margin, answering the question we'd been building toward: the gap wasn't in our loss function or training regime, it was in not having seen 94,000 hours of speech. But regularization still mattered as much as architecture: EfficientNet-B0 only stopped overfitting once we froze layers and added dropout; without that, extra capacity alone just overfit faster. The WavLM results also didn't fully transfer to new, real-world audio despite strong IEMOCAP numbers. **Not adopted**, both approaches violate the project's from-scratch constraint.

**WavLM-specific observations** *(don't generalize to the other experiments in this document)*:
- Training accuracy (0.75) was only a little above validation accuracy (0.65), a small train/val gap like this means the model isn't overfitting; the labels themselves are just noisy.
- Validation score stopped improving after about epoch 4. Audio-only and audio+text lines sit on top of each other for the rest of training, which is the clearest early evidence that adding text doesn't meaningfully help this setup (see Experiment 8).

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.5 --pitch-shift-prob 0.2 --enable-spec-augment
uv run python train.py --model-name wavlm_only --epochs 15 --dropout 0.1 --batch-size 16
```
> Note: the current `train.py` fine-tunes `microsoft/wavlm-large`'s top 6 layers by default, the "frozen encoder" variant we also tried isn't exposed as a CLI flag yet.

---

### Experiment 8: Text/Multimodal & Large Audio Models (Gemma Audio)

[#experiment-8-textmultimodal--large-audio-models-gemma-audio](#experiment-8-textmultimodal--large-audio-models-gemma-audio)

**Hypothesis**

With WavLM confirming that pretrained audio alone was a big lever (Experiment 7), the last open question was whether adding another modality, transcribed text and/or a large-scale pretrained audio model like Gemma Audio, could push things further still.

**Setup**

- Transcript-only (MiniLM embeddings + linear MLP): 20 epochs, Dropout 0.1
- Gemma Audio conformer: 15 epochs, Dropout 0.1
- WavLM multimodal (audio + text): 15 epochs, Dropout 0.1, batch size 16

**Results**

| Setup | Val Acc | Test Acc | Test Macro-F1 |
| --- | --- | --- | --- |
| Transcript-only (MiniLM) | 0.627 | 0.586 | 0.553 |
| Gemma Audio conformer | 0.59 | 0.60 | 0.58 |
| WavLM multimodal + augmentation | 0.631 | 0.647 | 0.623 |

**Conclusions**

Text was a finding, not a failure: multimodal performance came out roughly on par with audio-only, confirming the plateau we'd already spotted in Experiment 7's training curves, and text was not statistically significant in a dedicated ablation (t-test p = 0.095; shuffle-text ablation ≈ 0 effect), the model largely ignores the text channel. Both the transcript-only and Gemma Audio results were also flagged for data leakage (IEMOCAP sessions are scripted, and Gemma's audio conformer is itself trained for speech recognition). **Not adopted.** This closed out the experimentation phase: from here, every remaining lever either violated the from-scratch constraint or added modality complexity without adding signal.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name transcript_only --epochs 20 --dropout 0.1
uv run python train.py --model-name gemma_audio --epochs 15 --dropout 0.1
uv run python train.py --model-name wavlm_and_transcript --epochs 15 --dropout 0.1 --batch-size 16
```

## 5. Results Summary

[#5-results-summary](#5-results-summary)

| # | Experiment | Best Test Acc | Best Test Macro-F1 | Final Model |
| --- | --- | --- | --- | --- |
| 1 | Baseline CNN (9 classes) | 0.36 | 0.282 | ❌ No |
| 2 | 4-class grouping | 0.49 | 0.451 | ✅ Yes |
| 3 | Class weighting + Focal Loss | 0.504 | 0.514 | ✅ Yes |
| 4 | Disabling early stopping (fixed epochs) | N/A | N/A | ✅ Yes |
| 5 | CNN + GRU / LSTM | 0.52 | 0.465 | ❌ No |
| 6 | SpecAugment & Pitch Shift | 0.52 | 0.465 | ❌ No |
| 7 | Pretrained backbones (EfficientNet-B0 / WavLM) | 0.641 | 0.606 | ❌ No (violates from-scratch constraint) |
| 8 | Text/Multimodal & Gemma Audio | 0.647 | 0.623 | ❌ No |

**Cross-team takeaways (not tied to a single experiment)**

- Speaker-independent (LOSO) evaluation is essential for an honest estimate, a genuinely harder setup than speaker-dependent benchmarks suggest.
- There's a practical ceiling on this dataset: noisy, disagreement-prone labels cap how far any architecture can go, by the last rounds, gains were down to 1–2 points.

---

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
