# configs/

Dashboard configuration files. All paths and processing parameters for the dashboard are defined here.

---

## Files

| File | Description |
|---|---|
| `default.yaml` | Main configuration — paths, audio settings, video settings, dashboard panel paths |

---

## default.yaml Reference

### `project`

| Key | Description |
|---|---|
| `name` | Project name (`weld-hackathon`) |

---

### `paths`

All paths are resolved relative to the `weld_project_template/` root unless they are absolute.

| Key | Default | Description |
|---|---|---|
| `raw_train_good` | `data/raw/train/good_weld` | Training good weld run folders |
| `raw_train_defect` | `data/raw/train/defect_weld` | Training defect weld run folders |
| `raw_test` | `data/raw/test` | Test run folders |
| `index` | `data/interim/index/run_index.jsonl` | JSONL index output |
| `inventory` | `data/interim/index/inventory.csv` | CSV inventory output |
| `manifest` | `data/interim/manifest.csv` | Pipeline manifest (from main pipeline) |
| `split_dict` | `data/interim/split_dict.json` | Train/val/test split assignments |
| `dataset_meta` | `data/interim/dataset_meta.json` | Dataset-level metadata |
| `norm_stats` | `data/interim/norm_stats.json` | Normalization statistics |
| `data_root` | `null` | Override root for manifest-based path rewriting |
| `features` | `data/processed/features/features.parquet` | Extracted feature table |
| `chunks` | `data/processed/chunks/chunks.parquet` | Aligned time-series chunks |

---

### `processing`

| Key | Default | Description |
|---|---|---|
| `active_window.current_threshold_a` | `5.0` | Current threshold (Amps) to detect the active welding window |
| `active_window.pad_seconds` | `0.75` | Padding added before/after the active window |
| `resample_hz` | `25` | Target sample rate for sensor/audio alignment (Hz) |

---

### `audio`

| Key | Default | Description |
|---|---|---|
| `sr` | `null` | Override sample rate (null = use file native rate) |
| `n_mels` | `128` | Number of mel-spectrogram frequency bands |
| `n_fft` | `2048` | FFT window size in samples |
| `hop_length` | `512` | Hop size in samples (~32 ms at 16 kHz) |

---

### `video`

| Key | Default | Description |
|---|---|---|
| `sample_fps` | `2` | Frame sampling rate for derived signal computation in the dashboard |
| `frame_size` | `224` | Frame resize dimension (square, for model input) |

---

### `dashboard`

Paths to pipeline output files consumed by each dashboard page.

| Key | Default | Description |
|---|---|---|
| `cache_dir` | `outputs/reports/dashboards/cache` | Cache directory for computed derived signals |
| `points_3d` | `outputs/reports/dashboards/points_3d.json` | Data for 3D Explorer page |
| `tabular` | `outputs/tabular` | Step 7 — LightGBM model and metrics directory |
| `checkpoints` | `outputs/checkpoints` | Step 11/12 — model checkpoints directory |
| `eval` | `outputs/eval` | Step 13 — evaluation results directory |
| `predictions` | `outputs/predictions` | Step 14 — inference predictions directory |
| `reports` | `outputs/reports` | Audit and report outputs |
| `dataset` | `data/interim` | Interim dataset metadata directory |
| `sample_data` | `assets/sample_data` | Pre-loaded sample runs for Live Sync |
| `sensor_stats` | `data/interim/sensor_stats.csv` | Per-run sensor aggregate statistics |
| `top15_csv` | `data/processed/dashboard_top15_features.csv` | Top-15 features data for Feature Insights |
| `top15_summary` | `data/processed/dashboard_feature_summary.json` | Feature summary JSON |
| `feature_importance` | `outputs/tabular/feature_importance.csv` | LightGBM feature importance |
| `audit_bias` | `outputs/audit_metadata_bias_report.json` | Metadata bias audit report |
| `audit_duplicates` | `outputs/audit_duplicates_report.json` | Duplicate detection audit report |
| `class_weights` | `outputs/tabular/class_weights.json` | Per-class weights for loss balancing |
| `eval_val_predictions` | `outputs/eval/val_predictions.csv` | Step 13 validation predictions |
| `eval_val_metrics` | `outputs/eval/val_metrics.json` | Step 13 validation metrics |
| `eval_per_class` | `outputs/eval/per_class_report.csv` | Step 13 per-class precision/recall/F1 |
| `tabular_val_metrics` | `outputs/tabular/val_metrics.json` | Step 7 tabular validation metrics |
| `inference_predictions` | `outputs/predictions/predictions.csv` | Step 14 test-set predictions |
| `checkpoint` | *(absolute path)* | `best_model.pt` from main pipeline (for live inference) |
| `norm_stats_inf` | *(absolute path)* | `norm_stats.json` from main pipeline (for live inference) |
| `pipeline_config` | *(absolute path)* | `config.yaml` from main pipeline root |

---

## Notes

- Relative paths are resolved from `weld_project_template/` (the directory containing `configs/`).
- Absolute paths (with drive letters on Windows) are used only for live inference artifacts that live in the main pipeline output folder.
- Set `paths.data_root` when using `--from-manifest` and the raw data has been moved from the original paths stored in `manifest.csv`.
