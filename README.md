# Student Mental Health Facial Analysis Service

An async Python backend microservice built with **FastAPI**, **MediaPipe FaceMesh**, and **PyTorch** to analyze short facial video sessions from students. The service estimates continuous risk indicators (0.0 to 1.0) for **Depression**, **Anxiety**, and **Stress**, alongside epistemic prediction confidence scores, top contributing facial descriptors, and automatic human-review escalation flags.

---

> [!IMPORTANT]
> ### 🛡️ Ethical & Clinical Framing for School Counselors & Reviewers
> - **Screening Aid Only**: This system is **NOT a clinical diagnostic tool** and does NOT generate diagnostic labels. It operates purely as an early screening aid to alert counselors to potential distress.
> - **Probabilistic Outputs**: All scores represent continuous probabilistic severity estimates paired with explicit prediction confidence scores.
> - **Human-in-the-Loop Safeguard**: Any session returning an elevated risk indicator ($\ge 0.60$) automatically sets `flag_for_review: true` and routes the session to a qualified human reviewer or counselor queue.
> - **Student Copy Protection**: The service never automatically messages students with clinical or diagnostic-sounding terminology. Frontend interfaces use the `confidence` score to suppress or soften copy when uncertainty is high.
> - **Privacy First**: Raw video session files are deleted immediately after frame feature extraction unless the student explicitly provides opt-in consent (`allow_raw_storage: true`).

---

## 🚀 Quick Start & Local Execution

### 1. Environment Setup
Requires Python 3.9+ (Python 3.10+ recommended).

```bash
# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the FastAPI Service
Start the service locally on port 8000:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation (Swagger)**: `http://localhost:8000/docs`
- **Alternative ReDoc UI**: `http://localhost:8000/redoc`
- **Health Check Endpoint**: `http://localhost:8000/v1/health`

### 3. Real-Time OpenCV Webcam Application
Run the live real-time webcam analysis script with an interactive visual HUD overlay:

```bash
# Run with live physical webcam (device index 0)
py -3 realtime_webcam.py

# Specify custom camera index (e.g. camera index 1)
py -3 realtime_webcam.py --camera 1

# Run in mock frame stream simulation mode (for testing without a camera)
py -3 realtime_webcam.py --mock
```

### 4. Running Automated Tests
Run the test suite covering API endpoints, MediaPipe pipelines, explainability logic, and PyTorch model inference:

```bash
pytest
```

---

## 📡 API Contract & Request/Response Specification

### Endpoint: `POST /v1/sessions/{session_id}/analyze`

#### Sample Request Payload:
```json
{
  "modalities": ["face"],
  "face": {
    "video_ref": "https://storage.example.com/sessions/student_9482.mp4",
    "allow_raw_storage": false
  }
}
```

#### Sample Response Payload:
```json
{
  "session_id": "student_session_9482",
  "modalities_used": ["face"],
  "scores": {
    "depression": { "value": 0.4215, "confidence": 0.7500 },
    "anxiety":    { "value": 0.3150, "confidence": 0.7500 },
    "stress":     { "value": 0.6820, "confidence": 0.7500 }
  },
  "top_features": [
    "flattened affect",
    "reduced blink rate",
    "sustained eyebrow furrowing"
  ],
  "flag_for_review": true
}
```

---

## 🧩 Voice Modality Extension Architecture

The service is intentionally designed to support future voice analysis without requiring breaking changes to endpoints or schemas:

1. **Array-based Modalities**: The request `modalities` and response `modalities_used` fields are arrays (e.g. `["face", "voice"]`).
2. **Keyed Score Dictionary**: The `scores` dictionary indexes indicators by name (`depression`, `anxiety`, `stress`), allowing multimodal score aggregation without positional dependency.
3. **Abstract Interface (`BaseModalityAnalyzer`)**:
   - Located in `app/modalities/base.py`.
   - To add a voice module, implement `VoiceAnalyzer(BaseModalityAnalyzer)` in `app/modalities/voice.py` and register it in `app/api/v1/endpoints.py`:
     ```python
     MODALITY_ANALYZERS = {
         "face": FacialAnalyzer(),
         "voice": VoiceAnalyzer() # Future extension
     }
     ```
4. **Modality Fusion Engine (`app/modalities/fusion.py`)**:
   - Automatically handles single-modality vs. multi-modality score weighting and applies a single-modality confidence cap (0.75) for face-only sessions.

---

## 🏋️ Model Training & Calibration

To run the PyTorch training loop on new or fine-tuning session datasets:

```bash
python -m training.train --epochs 10 --batch-size 8 --save-path checkpoints/facial_model.pt
```

For a quick dry run to verify the training and metric generation pipeline:

```bash
python -m training.train --dry-run
```

The script reports evaluation metrics (**MAE**, **F1-score**, **AUC-ROC**, **ECE**) per severity head and generates calibration curve plots (`calibration_depression.png`, etc.). Single-modality face predictions are calibrated to express higher uncertainty relative to future multimodal fused models.
