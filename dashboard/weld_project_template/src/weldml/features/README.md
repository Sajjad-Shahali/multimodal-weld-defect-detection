# weldml/features/

Feature extraction and real-time derived signal computation for the dashboard.

---

## Files

| File | Description |
|---|---|
| `extract.py` | Feature extraction driver — reads JSONL index, writes per-run features to parquet |
| `derived_signals.py` | Audio and video signal computation for the Live Sync overlay |

---

## extract.py

Reads `run_index.jsonl` and computes per-run features, writing the result to a parquet or CSV file.

### Key Function

```python
extract_features(cfg: dict) -> None
```

Reads the JSONL index path from `cfg["paths"]["index"]`, iterates over all runs, calls `extract_features_from_run()` for each, and writes a combined table to `cfg["paths"]["features"]`.

### `extract_features_from_run(record: dict) -> dict`

Stub extractor returning:

| Field | Description |
|---|---|
| `run_id` | Run identifier |
| `label` | Defect class label |
| `sensor_duration_s` | Sensor recording duration (from media metadata) |
| `audio_duration_s` | Audio recording duration |
| `video_duration_s` | Video recording duration |

> **Extension point:** Replace or extend this function with domain-specific audio MFCC statistics, sensor derived features, or video temporal features as required by your analysis.

---

## derived_signals.py

Computes four time-series signals from the raw audio FLAC and video AVI files for use as overlays in the **Live Sync** dashboard page. Results are cached to disk as `.npz` files to avoid re-computation on every page load.

### Signals

#### Audio Signals

| Function | Signal | Window | Description |
|---|---|---|---|
| `compute_audio_rms` | `audio_rms` | 50 ms hop | RMS energy envelope over time |
| `compute_audio_spectral_centroid` | `audio_centroid` | 50 ms hop | Spectral centroid (Hz) over time |

Both functions return `(t_sec: np.ndarray, values: np.ndarray)` tuples.

**Audio processing:**
1. Read FLAC via `soundfile` (mono, native sample rate)
2. Resample if target SR differs from file SR (via `librosa.resample`)
3. Compute with `librosa.feature.rms` / `librosa.feature.spectral_centroid`
4. Return time axis computed from hop size and sample rate

#### Video Signals

| Function | Signal | Sample Rate | Description |
|---|---|---|---|
| `compute_video_brightness` | `video_brightness` | 5 fps | Mean grayscale pixel value per frame |
| `compute_video_motion` | `video_motion` | 5 fps | Mean absolute pixel difference from previous frame |

Both functions return `(t_sec: np.ndarray, values: np.ndarray)` tuples.

**Video processing:**
1. Open AVI via `cv2.VideoCapture`
2. Sample every `fps / sample_fps` frames
3. Convert to grayscale, compute brightness mean or frame difference
4. Return time axis from frame index and native FPS

### `get_derived_signals(audio_path, video_path, cache_dir, run_id, hop_ms=50, video_sample_fps=5)`

Orchestrates all four signals with disk caching.

```python
from weldml.features.derived_signals import get_derived_signals
from pathlib import Path

signals = get_derived_signals(
    audio_path=Path("assets/sample_data/08-17-22-0011-00/weld.flac"),
    video_path=Path("assets/sample_data/08-17-22-0011-00/weld.avi"),
    cache_dir=Path("outputs/reports/dashboards/cache"),
    run_id="08-17-22-0011-00",
)

t_rms, rms_values = signals["audio_rms"]
t_bright, brightness = signals["video_brightness"]
```

**Cache structure:**
```text
cache_dir/
└── {run_id}/
    ├── audio_rms.npz
    ├── audio_centroid.npz
    ├── video_brightness.npz
    └── video_motion.npz
```

Each `.npz` stores two arrays: `t` (time axis) and `v` (signal values).

---

## Error Handling

All signal computation functions catch exceptions silently and return empty arrays `(np.array([]), np.array([]))`. The dashboard checks for empty arrays before plotting, so missing or corrupt media files degrade gracefully without crashing the page.
