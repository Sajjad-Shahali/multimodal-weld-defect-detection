# Weld Project Template

Interactive Streamlit dashboard scaffold for visualizing the multimodal weld defect detection pipeline outputs. Drop new run folders under `data/raw/` and run the scripts in `scripts/` to get started.

---

## Directory Structure

```text
weld_project_template/
├── configs/
│   └── default.yaml          # All paths and processing parameters
├── assets/
│   └── sample_data/          # 10 pre-loaded sample weld runs (AVI + CSV + FLAC)
├── scripts/
│   ├── build_index.py        # Discover runs → inventory.csv + run_index.jsonl
│   ├── extract_features.py   # Compute features → features.parquet
│   ├── build_top15_features.py  # (placeholder) Top-15 feature builder
│   └── run_dashboard.py      # Launch the Streamlit dashboard
├── src/
│   └── weldml/               # Python package — all dashboard logic
│       ├── dashboard/app.py  # Main multi-page Streamlit app (~1300 lines)
│       ├── features/         # Feature extraction and derived signals
│       ├── data/             # Run discovery, indexing, inventory
│       ├── models/           # (reserved) model loading helpers
│       └── utils/            # Config loader
├── data/                     # Pipeline data artifacts (populated at runtime)
│   ├── raw/                  # Raw weld run folders
│   ├── interim/              # Manifest, splits, normalization stats, index
│   └── processed/            # Feature parquet, dashboard data
├── outputs/                  # Training and evaluation artifacts
│   ├── tabular/              # Step 7 — LightGBM outputs
│   ├── checkpoints/          # Step 11/12 — model checkpoints
│   ├── eval/                 # Step 13 — evaluation results
│   ├── predictions/          # Step 14 — inference outputs
│   └── reports/              # Audit reports, dashboard cache
└── docs/
    └── dashboard_expected_outputs.md  # Expected file formats for each pipeline step
```

---

## Expected Run Folder Format

Each weld run is a folder containing:

| File | Description |
|---|---|
| `sensor.csv` | Timestamped sensor readings (pressure, current, voltage, CO2 flow) |
| `weld.flac` | Mono 16 kHz audio recording of the weld |
| `weld.avi` | Video of the weld pool |
| `images/` | ~5 static JPG snapshots of the weld |

Example layout:

```text
data/raw/train/good_weld/08-17-22-0011-00/
  sensor.csv
  weld.flac
  weld.avi
  images/
    0001.jpg
    0002.jpg
    ...
    0005.jpg
```

Run ID is derived from the folder name. Label is encoded in the last two digits of the folder name (e.g., `-00` = good_weld, `-01` = excessive_penetration).

---

## Pipeline Data Outputs

### 1. Index & Inventory

| File | Description |
|---|---|
| `data/interim/index/inventory.csv` | One row per run — health check (has_sensor, has_audio, has_video, durations) |
| `data/interim/index/run_index.jsonl` | One JSON object per run — full paths + media metadata |

### 2. Features

| File | Description |
|---|---|
| `data/processed/features/features.parquet` | One row per run — computed feature table |
| `data/processed/chunks/chunks.parquet` | Aligned time-series chunks per run (for deep models) |

### 3. Training & Evaluation Artifacts (for dashboard pages)

| Path | Step | Contents |
|---|---|---|
| `outputs/tabular/` | Step 7 | `model_lgb.pkl`, `val_predictions.csv`, `val_metrics.json`, `feature_importance.csv`, `class_weights.json` |
| `outputs/checkpoints/` | Step 11/12 | `best_model.pt`, `training_log.json`, `training_summary.json`, `calibration_*.json` |
| `outputs/eval/` | Step 13 | `val_predictions.csv`, `val_metrics.json`, `confusion_matrix.png`, `per_class_report.csv` |
| `outputs/predictions/` | Step 14 | `predictions.csv` |
| `outputs/reports/` | Step 15 | `audit_metadata_bias_report.json`, `audit_duplicates_report.json` |

---

## Quickstart

### 1. Build the run index

```bash
# From raw folder structure
python scripts/build_index.py --config configs/default.yaml

# From pipeline manifest (recommended when using the main pipeline outputs)
python scripts/build_index.py --config configs/default.yaml --from-manifest
```

### 2. Extract features

```bash
python scripts/extract_features.py --config configs/default.yaml
```

### 3. Launch the dashboard

```bash
python scripts/run_dashboard.py --config configs/default.yaml
```

The dashboard will be available at `http://localhost:8501`.

---

## Using with the Main Pipeline

When running the full pipeline from the repo root, copy the following files to `data/interim/`:

```bash
cp output/manifest.csv         data/interim/manifest.csv
cp output/split_dict.json      data/interim/split_dict.json
cp output/dataset_meta.json    data/interim/dataset_meta.json
cp output/dataset/norm_stats.json data/interim/norm_stats.json
```

Then update `configs/default.yaml` → `paths.data_root` to point to the folder containing `good_weld/` and `defect_data_weld/`.

---

## Configuration

All paths and processing parameters live in `configs/default.yaml`. The live inference model paths (`checkpoint`, `norm_stats_inf`, `pipeline_config`) point to the main pipeline outputs and can use absolute paths.

See [`configs/README.md`](configs/README.md) for full parameter reference.

---

## Requirements

- Python 3.9+
- All packages in the root `requirements.txt`
- **ffmpeg** — required for AVI playback in the browser. The dashboard auto-converts AVI → MP4 when ffmpeg is installed.
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Dataset Stats** | Manifest summary, train/val/test splits, class distribution, normalization stats |
| **Sensor Stats** | Per-run aggregations, duration distribution, per-sensor analysis, correlation heatmap |
| **Tabular Baseline** | LightGBM metrics, feature importance, validation predictions |
| **WeldFusionNet Training** | Training curves (loss, F1, LR), per-epoch metrics table |
| **Temperature Calibration** | ECE before/after calibration, temperature value |
| **Validation Results** | Per-class F1, confusion matrices (7-class + binary), per-class report |
| **Holdout Test** | Test-set metrics, per-class F1, predictions table |
| **Model Comparison** | LightGBM vs WeldFusionNet side-by-side |
| **3D Explorer** | Interactive 3D scatter of sensor means/stds/duration colored by label |
| **Live Sync** | HTML5 video player synced with sensor timeline (JavaScript interpolation) |
| **Class Distribution** | Class counts, pie chart, class weights for loss balancing |
| **Feature Insights** | Correlation heatmap, feature importance, tabular summary |
| **Model Performance Insights** | Per-class precision/recall/F1, Tabular vs DL comparison |
| **Inference** | Batch directory scan or single-file upload for live predictions |
