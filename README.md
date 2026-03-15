# Multimodal Weld Defect Detection
## Therness Hackathon - DATA forge

A reproducible multimodal weld defect detection pipeline combining acoustic, visual, and sensor signals via a multitask deep learning model.

---

## Final Leaderboard Result

| Metric | Score |
|---|---|
| **Final Score** | `0.9567` |
| **Binary F1** | `0.9677` |
| **Type Macro F1** | `0.9401` |

**Team:** `DATA forge`

### Local Validation Snapshot

```text
Binary F1  : 0.9835
Macro F1   : 0.9830
FinalScore : 0.9833
ECE        : 0.0308
Temperature: 0.5527
```

> Note: leaderboard score is computed on hidden test data and may differ from local validation.

---

## Overview

The project supports:

- Audio (MFCC + mel-spectrogram + spectral features)
- Video (MobileNetV3 visual embedding + temporal attention pooling)
- Optional sensor CSV branch (pressure, current, voltage, CO2 flow)
- Multitask learning (7-class defect type + auxiliary binary good/defect head)
- Temperature scaling calibration
- Leakage auditing
- End-to-end submission generation and post-processing optimization

---

## Architecture: WeldFusionNet

```text
Audio  (18 × 25)     → Conv1dEncoder (1D-CNN) → 64-d
Video  (5, 3, H, W)  → MobileNetV3-Small + Temporal Attention → 128-d
                                     ↓
                         Concat (192-d)
                         FC → BN → ReLU → Dropout → 128-d
                                     ↓
                 ┌───────────────────┴───────────────────┐
           7-Class Head                             Binary Head
           Linear(128, 7)                           Linear(128, 1)
```

### Loss Function

```text
Total Loss = α × FocalLoss(7-class) + β × BCE(binary)
α = 0.7,  β = 0.3,  focal_γ = 1.0
```

### Fusion Options

| Mode | Description |
|---|---|
| `concat` (default) | Concatenate embeddings → MLP fusion |
| `transformer` (Tier 3) | Treat each modality as a token, cross-attend via Transformer encoder |

---

## Audio Analysis Specifications

Audio is captured as FLAC files during the welding process. Each file is mono at 16 kHz.

### Processing Pipeline (step3_audio.py)

| Parameter | Value |
|---|---|
| **Format** | FLAC (lossless, mono) |
| **Sample Rate** | 16,000 Hz |
| **Library** | `librosa` + `soundfile` |
| **Output** | `output/audio/{run_id}.npz` |

### Mel-Spectrogram

| Parameter | Value |
|---|---|
| **FFT Window** (`n_fft`) | 2048 samples |
| **Hop Length** (`hop_length`) | 512 samples (~32 ms) |
| **Mel Bands** (`n_mels`) | 128 |
| **Scale** | Power → dB (log-mel, `ref=np.max`) |
| **Output shape** | `(128, time_frames)` |

### MFCC Features

| Parameter | Value |
|---|---|
| **Coefficients** (`n_mfcc`) | 13 |
| **FFT Window** | 2048 samples |
| **Hop Length** | 512 samples |
| **Output shape** | `(13, time_frames)` |

### Frame-Level Spectral Features

| Feature | Description |
|---|---|
| `rms` | Root mean square energy per hop window |
| `spectral_centroid` | Frequency-weighted center of mass (Hz) |
| `spectral_bandwidth` | Spread around the spectral centroid (Hz) |
| `zcr` | Zero crossing rate |
| `spectral_rolloff` | Frequency below which 85% of energy falls |

### Model Input

Audio tensors are shaped `(B, 18, 25)` — 18 feature channels × 25 time steps — fed into `Conv1dEncoder` (two Conv1d layers + `AdaptiveAvgPool1d`) → **64-dimensional embedding**.

### Dashboard Derived Signals (Live Sync overlay)

| Signal | Hop | Description |
|---|---|---|
| `audio_rms` | 50 ms | RMS energy envelope over time |
| `audio_centroid` | 50 ms | Spectral centroid (Hz) over time |

---

## Video Analysis Specifications

Video is recorded as AVI files capturing the weld pool and arc during each weld run.

### EDA Frame Extraction (step4_video.py)

Used for dashboard thumbnails and exploratory analysis only. The neural model reads AVI directly at training time (step6).

| Parameter | Value |
|---|---|
| **Format** | AVI |
| **Library** | OpenCV (`cv2`) |
| **EDA Extraction Rate** | 1.0 fps |
| **Resize** | 480 × 300 px |
| **Output frames** | `output/frames/{run_id}/frame_{idx:05d}.jpg` |
| **Output stats** | `output/frames/{run_id}/frame_stats.csv` |

### Per-Frame Statistics

| Column | Description |
|---|---|
| `brightness_mean` | Mean grayscale pixel value |
| `brightness_std` | Std dev of grayscale pixel values |
| `red_mean` / `green_mean` / `blue_mean` | Per-channel mean values |
| `motion_energy` | Mean absolute pixel difference from previous frame |
| `frame_idx` | Original frame index in the AVI |
| `timestamp_sec` | Frame timestamp in seconds |

### Neural Model Video Branch (step6 + step9_model.py)

| Parameter | Value |
|---|---|
| **Backbone** | MobileNetV3-Small (ImageNet pretrained) |
| **Frames per sample** | 5 (subsampled from 25 per chunk) |
| **Input resolution** | 224 × 224 px |
| **Backbone feature dim** | 576-d (after `AdaptiveAvgPool2d(1)`) |
| **Frozen layers** | First 80% of MobileNetV3 feature layers |
| **Temporal pooling** | Learned attention gate — softmax over frame scores |
| **Output embedding** | 128-d |

### Dashboard Derived Signals (Live Sync overlay, sampled at 5 fps)

| Signal | Description |
|---|---|
| `video_brightness` | Mean grayscale intensity per frame over time |
| `video_motion` | Mean absolute frame-difference (motion energy) over time |

---

## Label Map

| Code | Label |
|---|---|
| `00` | good_weld |
| `01` | excessive_penetration |
| `02` | burn_through |
| `06` | overlap |
| `07` | lack_of_fusion |
| `08` | excessive_convexity |
| `11` | crater_cracks |

---

## Repository Structure

```text
multimodal-weld-defect-detection/
├── dataset/
│   ├── defect-weld/
│   └── good-weld/
├── dataset-test/                    # test split used for submission
├── pipeline/
│   ├── step1_validate.py            # run discovery & sanity checks
│   ├── step2_sensor.py              # CSV sensor preprocessing
│   ├── step3_audio.py               # FLAC → mel / MFCC / spectral features
│   ├── step4_video.py               # AVI → frames + per-frame stats (EDA)
│   ├── step5_align.py               # sensor / audio / video alignment
│   ├── step6_dataset.py             # PyTorch dataset builder
│   ├── step7_tabular_baseline.py    # LightGBM baseline
│   ├── step8_dataset_torch.py       # DataLoader setup
│   ├── step9_model.py               # WeldFusionNet definition
│   ├── step10_losses.py             # Focal + BCE multitask loss
│   ├── step11_train.py              # training loop
│   ├── step12_calibrate.py          # temperature scaling calibration
│   ├── step13_evaluate.py           # validation evaluation
│   ├── step14_inference.py          # test-set inference
│   ├── step15_sanity_leakage.py     # leakage audit
│   └── run_all.py                   # pipeline orchestrator
├── dashboard/
│   └── weld_project_template/       # Streamlit visualization dashboard
├── output/                          # generated artifacts (models, metrics)
├── config.yaml                      # master pipeline config
├── requirements.txt
├── generate_final_submission.py
├── optimize_postprocessing.py
└── fix_submission.py
```

---

## Installation

```bash
git clone https://github.com/Sajjad-Shahali/multimodal-weld-defect-detection.git
cd multimodal-weld-defect-detection

python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
```

---

## Run the Pipeline

### Phase 1 — Preprocessing

```bash
python -m pipeline.run_all --steps 1 2 3 4 5 6
```

### Phase 2 — Training

```bash
python -m pipeline.run_all --steps 7 8 9 10 11
```

### Phase 3 — Calibration and Evaluation

```bash
python -m pipeline.run_all --steps 12 13
```

### All Phases

```bash
python -m pipeline.run_all
```

---

## Submission Generation

```bash
# End-to-end submission
python generate_final_submission.py

# Post-processing optimization
python optimize_postprocessing.py

# Consistency fixer
python fix_submission.py --input submission.csv --output submission_fixed.csv
```

---

## Dashboard

The interactive Streamlit dashboard visualizes all pipeline outputs:

```bash
cd dashboard/weld_project_template
python scripts/run_dashboard.py --config configs/default.yaml
```

See [`dashboard/weld_project_template/README.md`](dashboard/weld_project_template/README.md) for full setup instructions.

**Dashboard pages:**

| Page | Description |
|---|---|
| Dataset Stats | Manifest summary, splits, class distribution |
| Sensor Stats | Run-level aggregations, correlation heatmap |
| Tabular Baseline | LightGBM metrics, feature importance |
| WeldFusionNet Training | Loss/F1 curves, per-epoch table |
| Temperature Calibration | ECE before/after, temperature value |
| Validation Results | Per-class F1, confusion matrices |
| Holdout Test | Test-set metrics, predictions table |
| Model Comparison | LightGBM vs WeldFusionNet side-by-side |
| 3D Explorer | Interactive 3D scatter by sensor regime & label |
| **Live Sync** | Video + sensor timeline synchronized in real time |
| Feature Insights | Correlation heatmap, feature importance |
| Inference | Batch or single-file prediction |

---

## Engineering Notes

- Deterministic splits (seed = 42)
- Config-driven pipeline via `config.yaml`
- End-to-end reproducibility
- Optional leakage audit (Step 15)
- Crash-safe per-sample inference
- Temperature scaling calibration (ECE-optimized)

---

## Scope Note

The `weld/` directory in this repository is a local virtual environment snapshot. It is not part of the project source logic and should not be treated as application code.

---

## License

MIT License
