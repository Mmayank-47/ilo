# Voice Emotion Analysis Module

This is a production-grade, modular Voice Emotion Analysis codebase built on PyTorch. It detects distress-relevant emotional states (stress, anxiety, fear, sadness, calm, neutral, etc.) from audio clips. 

This module serves as a single acoustic input channel in a larger, multimodal distress score prediction pipeline. It is designed to work in tandem with text sentiment analysis and a future facial expression analysis module.

---

## Codebase Architecture

```
voice_emotion/
├── config/
│   └── default.yaml           # Master YAML configuration file
├── data/
│   ├── __init__.py
│   ├── dataset.py             # RAVDESS loader, multi-dataset registry, and speaker splits
│   ├── features.py            # Feature extraction (MFCCs+Δ+ΔΔ, pitch, ZCR, RMS, tempo, silence)
│   └── augmentation.py        # Audio augmentation: pitch shift, stretch, noise, SpecAugment
├── models/
│   ├── __init__.py
│   ├── baseline.py            # Baseline LSTM reference model (2 layers, 128 units)
│   ├── improved.py            # Improved BiLSTM + self-attention + optional CNN frontend
│   └── fusion.py              # MultimodalFusionHead interface stub (concat/cross-attention ready)
├── training/
│   ├── __init__.py
│   ├── losses.py              # Focal Loss, class-weighted CE, label smoothing CE
│   ├── trainer.py             # Speaker-aware cross-validation training loop
│   └── callbacks.py           # EarlyStopping, ModelCheckpoint, MetricsLogger
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # Class metrics (precision, recall, F1) & confusion matrix
│   ├── threshold_tuning.py    # Post-hoc per-class threshold tuning
│   └── report.py              # Automated JSON + Markdown run summary report generator
├── explainability/
│   ├── __init__.py
│   ├── attention_viz.py       # Time-aligned attention weight plot over audio timeline
│   └── feature_importance.py  # Integrated Gradients / gradient saliency explainer
├── tests/
│   ├── test_features.py       # Feature shape & value range sanity checks
│   ├── test_model.py          # Model architecture shape verification
│   └── test_dataset.py        # Split logic & file parsing tests
├── infer.py                   # Stable prediction API (VoiceEmotionPredictor)
├── train.py                   # Training & CV entrypoint
├── evaluate.py                # Checkpoint validation & test-set evaluation entrypoint
└── requirements.txt           # Python dependency specifications
```

---

## Setup Instructions

### 1. Set Up Environment
Create and activate your Python virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```
*(Optional)* For the full Integrated Gradients explainability method, install `captum`:
```powershell
pip install captum
```
*Note: If `captum` is not installed, the explainer will automatically fallback to input-gradient saliency.*

### 3. Download the Dataset
1. Download the RAVDESS Speech dataset (`Audio_Speech_Actors_01-24.zip`) from [Zenodo](https://zenodo.org/record/1188976).
2. Unzip it.
3. Configure the path in `config/default.yaml` under `dataset.root_dir` (or pass it via command-line arguments).

---

## How to Run

### 1. Run Unit Tests
Verify feature extraction and model shapes using `pytest`:
```powershell
pytest tests/ -v
```

### 2. Run Training
Train the model using 5-fold cross-validation. Run a dry-run first to check settings:
```powershell
python train.py --dry-run
```
To run full training:
```powershell
python train.py --config config/default.yaml
```
You can override configuration settings on-the-fly using the `--set` option:
```powershell
python train.py --set model.type=baseline --set training.epochs=10
```

### 3. Evaluate Checkpoint
Evaluate the model checkpoint on the test set:
```powershell
python evaluate.py --checkpoint outputs/fold_0/best_model.pt --config config/default.yaml --thresholds outputs/fold_0/thresholds.json --explain-n 3
```

---

## Public API Usage (Downstream Integration)

Downstream consumers (IVRS, chatbot backends, fusion hubs) only need to interact with the stable public class `VoiceEmotionPredictor` in `infer.py`.

```python
from infer import VoiceEmotionPredictor

# Load the predictor from a saved checkpoint
predictor = VoiceEmotionPredictor.from_checkpoint(
    checkpoint_path="outputs/fold_0/best_model.pt",
    config_path="config/default.yaml",
    thresholds_path="outputs/fold_0/thresholds.json"
)

# Predict emotion, distress sub-score, and return the multimodal embedding
result = predictor.predict("path/to/caller_input.wav")

print("Calibrated Probabilities:", result["emotion_probs"])
print("Predicted Emotion:", result["predicted_emotion"])
print("Distress Subscore:", result["distress_subscore"])
print("Embedding (128-d Vector):", result["embedding"].shape)
```

---

## Explainability features

### 1. Self-Attention Timeline
The attention timeline shows exactly which audio frames (silent vs voiced parts, high pitch segments) influenced the emotion prediction. When `predictor.explain(...)` is called, a plot is saved displaying:
- The audio waveform
- The Mel spectrogram
- A time-aligned visual heatmap of the self-attention weights

### 2. Feature Saliency
Provides a horizontal bar plot showing the attribution of the prediction to specific acoustic categories (e.g., Pitch instability, RMS energy, Zero-Crossing Rate, or specific MFCC bins) to help clinicians/operators understand the distress score rationale.
