# Speech Emotion Recognition - Meeting Audio Classifier

**UPC AIDL Final Project**

**Team**: Angelica Tacca Dughetti, Andrea Sanchez, Aitana Diaz, Diego Alonso

**Supervisor**: Pol Caselles

## 🔗 Demo
- [Hugging Face Space](https://huggingface.co/spaces/aitanadiaz/speech-emotion-recognition) - Live demo
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

This project focuses on **Speech Emotion Recognition (SER) for meeting audio**: given a short audio clip, classify the speaker's emotional state. Meetings are a particularly useful domain, speech is often overlapping, informal, and emotionally subtle, unlike the acted, exaggerated emotion in many benchmark datasets. Successfully tackling this could enable tools that flag team friction, measure engagement, or summarize meeting tone automatically.

We chose this project because it combines a real audio signal processing challenge (extracting meaningful features from raw waveforms) with a classic deep learning problem: class imbalance, speaker-independent generalization, and the trade-off between model capacity and overfitting on a moderately sized dataset.

</p>

## 2. Proposal

**Architecture**

We use a CNN-based image classification approach applied to audio: raw waveforms are converted into log-mel spectrograms, which are then fed into a 2D convolutional classifier, first a custom architecture, later a fine-tuned EfficientNet-B0 backbone, and finally a fine-tuned WavLM audio encoder, which became our best-performing and final selected model.
Full architectural details and iterations are documented experiment-by-experiment in [Section 4](#4-experiments).

**Data**
- Dataset: [IEMOCAP](https://sail.usc.edu/iemocap/), ~10,000 audio clips from 10 actors across 5 sessions
- Original 9 emotion labels merged into 4 balanced classes: Anger, Happiness, Sadness, Neutral
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

```bash
cd train-cli/
uv run python train.py --chunks-group-id 1 --model-name simple_cnn --epochs 10 --dropout 0.3 --disable-spec-augment # Experiment 1: Baseline CNN (9 classes)
uv run python train.py --model-name simple_cnn --epochs 15 --dropout 0.3 --enable-spec-augment # Experiment 2: Weighted loss + SpecAugment (9 classes)
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.3 --enable-spec-augment # Experiment 3: 4-class merge
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.0 --enable-spec-augment # Experiment 4: Skip connections
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.1 --enable-spec-augment # Experiment 5: Reduced dropout
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.1 --enable-spec-augment # Experiment 6: EfficientNet-B0
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.5 --pitch-shift-prob 0.2 --enable-spec-augment # Experiment 7: EfficientNet-B0, frozen layers + pitch shift
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.5 --enable-spec-augment # Experiment 8: EfficientNet-B0 + manual weights
uv run python train.py --model-name cnn_gru --epochs 100 # Experiment 9: CNN + GRU
uv run python train.py --model-name wavlm_only --epochs 15 --dropout 0.1 --batch-size 16 # Experiment 10: WavLM audio only
uv run python train.py --model-name wavlm_and_transcript --epochs 15 --dropout 0.1 --batch-size 16 # Experiment 11: WavLM multimodal (audio + text)
```

### Reproducing our best model(s)

We report the best model in each family:

- **Best CNN-based model**, **Experiment 7** (EfficientNet-B0, frozen layers, dropout 0.5, pitch shift augmentation): Test Acc 0.536, Test Macro-F1 0.502.
- **Best pretrained model and final selected model**, **Experiment 10** (WavLM Large, audio only, top 6 layers fine-tuned): Test Acc 0.641, Test Macro-F1 0.603.
  WavLM multimodal (audio + text, Experiment 11) scored slightly higher (0.656 / 0.619), but the text gain was not statistically significant (p = 0.095), so we kept the simpler audio-only model as final.

See [Section 4](#4-experiments) for full configurations.

---

## 4. Experiments

### Phase A: Baseline and Class Imbalance (Experiments 1-3)

**Hypothesis**

A simple CNN on the original 9 IEMOCAP emotions would give us a reference point. From there, weighting the loss seemed like the natural first fix for class imbalance, and grouping the 9 emotions into 4 broader classes seemed like a more structural fix for the same problem.

**Setup**

| # | Experiment | Classes | Config |
| --- | --- | --- | --- |
| 1 | Baseline CNN | 9 | 10 epochs, Dropout 0.3, CrossEntropy, no SpecAugment |
| 2 | Weighted loss + SpecAugment | 9 | 15 epochs, Dropout 0.3, weighted CrossEntropy, SpecAugment: Yes |
| 3 | 4-class merge | 4 | 20 epochs, Dropout 0.3, CrossEntropy, SpecAugment: Yes |

**Results**

| # | Experiment | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baseline CNN | 9 | 0.376 | 0.36 | 0.282 | First experiment |
| 2 | Weighted loss + SpecAug | 9 | 0.295 | 0.31 | 0.27 | Worse than baseline |
| 3 | 4-class merge | 4 | 0.451 | 0.49 | 0.451 | Key turning point |

**Conclusions**

Weighting the loss on top of 9 fine-grained classes actually made things worse, the classes were too granular and ambiguous for weighting alone to fix. Merging them into 4 broader classes is what actually worked: Macro-F1 jumped from 0.28 to 0.45, a bigger gain than anything else in this phase. This confirmed class granularity, not loss weighting, was the real bottleneck, and became the shared baseline for every experiment that followed.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name simple_cnn --epochs 10 --dropout 0.3 --disable-spec-augment
uv run python train.py --model-name simple_cnn --epochs 15 --dropout 0.3 --enable-spec-augment
uv run python train.py --model-name simple_cnn --epochs 20 --dropout 0.3 --enable-spec-augment
```

---

### Phase B: Architecture Refinements on the 4-Class CNN (Experiments 4-5)

**Hypothesis**

With 4 classes established as the working setup, two independent architectural tweaks looked promising: residual connections should improve gradient flow (as in ResNet), and reducing dropout should let the model use more of its capacity now that the classification problem was easier.

**Setup**

| # | Experiment | Classes | Config |
| --- | --- | --- | --- |
| 4 | Skip connections | 4 | 20 epochs, Dropout 0.0, CrossEntropy, residual blocks (Conv→BN→ReLU→Conv→BN + shortcut) |
| 5 | Reduced dropout | 4 | 20 epochs, Dropout 0.1, CrossEntropy, SpecAugment: Yes |

**Results**

| # | Experiment | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | Skip connections | 4 | 0.477 | 0.52 | 0.487 | |
| 5 | Reduced dropout | 4 | 0.497 | 0.52 | 0.485 | |

**Conclusions**

Both changes gave a modest, roughly equivalent bump over the 4-class baseline (Experiment 3), landing in the same Test Acc / Macro-F1 range as each other. Neither one clearly beat the other, suggesting we were bumping into the ceiling of what a from-scratch CNN of this size could do on this dataset, which is what motivated trying a pretrained backbone next.

---

### Phase C: Pretrained CNN Backbone, EfficientNet-B0 (Experiments 6-8)

**Hypothesis**

Having plateaued with architectural tweaks on the from-scratch CNN, we wanted to see how much a backbone pretrained on a large external corpus (EfficientNet-B0, pretrained on ImageNet) could add, even though it was never designed for audio.

**Setup**

| # | Experiment | Classes | Config |
| --- | --- | --- | --- |
| 6 | EfficientNet-B0 | 4 | 30 epochs, Dropout 0.1, CrossEntropy, SpecAugment: Yes |
| 7 | EfficientNet-B0, frozen layers + pitch shift | 4 | 30 epochs, Dropout 0.5, `--pitch-shift-prob 0.2`, SpecAugment: Yes |
| 8 | EfficientNet-B0 + manual weights | 4 | 30 epochs, Dropout 0.5, manual class weights, SpecAugment: Yes |

**Results**

| # | Experiment | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | EfficientNet-B0 | 4 | 0.51 | 0.53 | 0.49 | Best so far, overfitting |
| 7 | EfficientNet-B0, frozen layers + pitch shift | 4 | 0.5106 | 0.536 | 0.502 | Overfitting solved |
| 8 | EfficientNet-B0 + manual weights | 4 | 0.503 | 0.504 | 0.478 | Didn't outperform Experiment 7 |

**Conclusions**

EfficientNet-B0 immediately beat every from-scratch CNN configuration from Phases A and B, confirming that pretraining, even from an unrelated domain (vision), had real value. But it overfit right away; freezing layers and adding pitch shift (Experiment 7) fixed that and became our best CNN-based model overall (Test Acc 0.536, Test Macro-F1 0.502). Swapping in manual class weights on top of that (Experiment 8) didn't help, reinforcing what we'd already seen in Phase A: principled, automatic approaches to class imbalance outperform manually tuned ones.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.1 --enable-spec-augment
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.5 --pitch-shift-prob 0.2 --enable-spec-augment
uv run python train.py --model-name efficientnet_b0 --epochs 30 --dropout 0.5 --enable-spec-augment
```

---

### Phase D: Recurrent Architecture (Experiment 9)

**Hypothesis**

Separately from the pretrained-backbone track, we wanted to test a from-scratch idea we hadn't tried yet: would adding a recurrent layer (GRU or LSTM) on top of the CNN's feature maps capture the temporal, prosodic dynamics of speech better than a purely convolutional model?

**Setup**

| # | Experiment | Classes | Config |
| --- | --- | --- | --- |
| 9 | CNN + GRU | 4 | 100 epochs, Dropout 0.1, CrossEntropy, no pitch shift, no SpecAugment (the SpecAugment/pitch-shift variant is covered separately in Phase F below) |

**Results**

| # | Experiment | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 9 | CNN + GRU | 4 | N/A | 0.478 | 0.459 | Too little data |

**Conclusions**

The recurrent layer underperformed both the plain CNN (Phase B) and EfficientNet-B0 (Phase C). With a dataset the size of IEMOCAP, there simply wasn't enough data for the GRU to learn useful temporal patterns on top of the CNN features, it added training time and complexity without a corresponding gain.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name cnn_gru --epochs 100
```

---

### Phase E: Pretrained Audio Backbone, WavLM (Experiments 10-11)

**Hypothesis**

EfficientNet-B0 (Phase C) showed that pretraining helps even from an unrelated domain. That raised the obvious next question: how much further could pretraining go with an audio-native encoder, and would adding text on top of audio (multimodal) push results even higher?

**Setup**

| # | Experiment | Classes | Config |
| --- | --- | --- | --- |
| 10 | WavLM (audio only) | 4 | 15 epochs, Dropout 0.1, batch size 16, top 6 layers fine-tuned |
| 11 | WavLM multimodal (audio + text) | 4 | 15 epochs, Dropout 0.1, batch size 16 |

**Results**

| # | Experiment | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | WavLM (audio only) | 4 | N/A | 0.641 | 0.603 | Best pretrained model |
| 11 | WavLM multimodal (audio + text) | 4 | N/A | 0.656 | 0.619 | Text not significant |

**Conclusions**

WavLM was, by a wide margin, the best-performing model of the entire project, comfortably ahead of both the from-scratch CNN (Phases A-B) and EfficientNet-B0 (Phase C). Adding text on top of audio (Experiment 11) nudged the numbers up slightly, but a dedicated ablation showed the gain wasn't statistically significant (t-test p = 0.095), so we kept the simpler audio-only model (Experiment 10) as our final selected model.

**Reproduce (current CLI)**

```bash
uv run python train.py --model-name wavlm_only --epochs 15 --dropout 0.1 --batch-size 16
uv run python train.py --model-name wavlm_and_transcript --epochs 15 --dropout 0.1 --batch-size 16
```

## 5. Results Summary

| # | Experiment | Classes | Val Acc | Test Acc | Test Macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baseline CNN | 9 | 0.376 | 0.36 | 0.282 | First experiment |
| 2 | Weighted loss + SpecAug | 9 | 0.295 | 0.31 | 0.27 | Worse than baseline |
| 3 | 4-class merge | 4 | 0.451 | 0.49 | 0.451 | Key turning point |
| 4 | Skip connections | 4 | 0.477 | 0.52 | 0.487 | |
| 5 | Reduced dropout | 4 | 0.497 | 0.52 | 0.485 | |
| 6 | EfficientNet-B0 | 4 | 0.51 | 0.53 | 0.49 | Best so far, overfitting |
| 7 | EfficientNet-B0 + frozen layers + pitch shift | 4 | 0.5106 | 0.536 | 0.502 | Overfitting solved, best CNN-based model |
| 8 | EfficientNet-B0 + manual weights | 4 | 0.503 | 0.504 | 0.478 | Didn't outperform Experiment 7 |
| 9 | CNN + GRU | 4 | N/A | 0.478 | 0.459 | Too little data |
| 10 | WavLM (audio only) | 4 | N/A | 0.641 | 0.603 | Best pretrained model, **final selected model** |
| 11 | WavLM multimodal (audio + text) | 4 | N/A | 0.656 | 0.619 | Text not significant |

**Cross-cutting takeaways (not tied to a single experiment)**

- Speaker-independent (LOSO) evaluation is essential for an honest estimate, a genuinely harder setup than speaker-dependent benchmarks suggest.
- An overly aggressive early-stopping patience was quietly cutting some early training runs short before convergence. This was tested separately but not added to the train CLI, we removed patience entirely to reduce complexity once we committed to a CosineAnnealingLR schedule instead.
- Manually tuned class weights consistently underperformed automatic or principled weighting across every architecture we tried them on (Phase A and Experiment 8).

## 6. Conclusions & Future Work

**What we learned**
- Class granularity, not loss weighting, was the real early bottleneck: merging 9 emotions into 4 balanced classes had a bigger impact than weighting the loss on the original 9 classes, which actually made things worse.
- Pretraining consistently outperformed training from scratch, first with EfficientNet-B0 (a vision model repurposed for spectrograms), then even more clearly with WavLM (an audio-native encoder). The single biggest jump in the whole project came from switching to a speech-pretrained backbone, not from architecture tweaks.
- Manually tuned loss weights did not outperform principled automatic weighting, reinforcing that the harder classes (Neutral, Positive) suffer more from feature overlap than from data scarcity alone.
- Speaker-independent evaluation is essential for an honest performance estimate and makes this a genuinely harder problem than speaker-dependent benchmarks suggest.
- Data quality sets a ceiling. Training accuracy (0.75) was only a little above validation accuracy (0.65) on our best model, a small gap, so the model isn't overfitting; the labels themselves are noisy. The last rounds of tuning only moved results by 1-2 points.
- Multimodality needs a reason. Adding the text transcript did not significantly help. For emotion, the signal lives mostly in the audio (tone/prosody), not the words.

**Limitations**
- Final Macro-F1 (~0.6) shows the model still struggles to separate Neutral and Positive emotions specifically.
- The demo app's diarization component is not fully functional yet.
- Performance was evaluated only on IEMOCAP (acted emotion); real meeting audio is likely more subtle and may not transfer perfectly.

**Future Work**
- Fix and properly evaluate the speaker diarization component for multi-speaker meeting audio.
- Collect or fine-tune on more naturalistic (non-acted) emotional speech data.
