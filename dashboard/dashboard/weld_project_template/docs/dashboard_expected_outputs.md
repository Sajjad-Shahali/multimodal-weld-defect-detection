# Dashboard Expected Output Formats

The dashboard reads these files from the pipeline. Ensure your steps produce them in this format.

## Step 7 – Tabular Baseline (`outputs/tabular/`)

| File | Format |
|------|--------|
| `val_metrics.json` | `{ "binary_f1": float, "macro_f1": float, "ece": float, "confusion_matrix": [[...]] }` |
| `val_predictions.csv` | `run_id, true_label, predicted_label, p_defect, ...` |
| `model_lgb.pkl` | LightGBM model (pickle) |

## Step 11 – Neural Training (`outputs/checkpoints/`)

| File | Format |
|------|--------|
| `training_log.json` | `{ "epochs": [0,1,...], "loss": [...], "f1": [...] }` or `{ "train_loss": [...], "val_f1": [...] }` |
| `training_summary.json` | `{ "best_epoch": int, "best_score": float, "training_time_s": float }` |
| `best_model.pt` | PyTorch checkpoint |

## Step 12 – Calibration (`outputs/checkpoints/`)

| File | Format |
|------|--------|
| `calibration_temperature.json` | `{ "temperature": float }` |
| `calibration_metrics.json` | `{ "ece_before": float, "ece_after": float, "bin_acc": [...], "bin_conf": [...] }` |

## Step 13 – Evaluation (`outputs/eval/`)

| File | Format |
|------|--------|
| `val_metrics.json` | `{ "binary_f1", "macro_f1", "ece", "confusion_matrix" }` |
| `val_predictions.csv` | `run_id, chunk_idx, true_label, predicted_label, probabilities...` |
| `per_class_report.csv` | Per-class precision, recall, F1 |
| `confusion_matrix.png` | Image file |

## Dataset (`data/interim/`)

| File | Format |
|------|--------|
| `manifest.csv` | `file, run_id, chunk_idx, label_code, avi_path, split, ...` |
| `norm_stats.json` | `{ "sensor_mean": [...], "sensor_std": [...], "audio_mean": [...], "audio_std": [...] }` |
| `dataset_meta.json` | `{ "num_classes", "class_codes", "class_names" }` |
