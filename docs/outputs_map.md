# Outputs Map

This document maps pipeline steps to generated artifacts. It is documentation-only and does not change runtime behavior.

## Step Outputs

1. `step1_validate.py`
- `output/inventory.csv`

2. `step2_sensor.py`
- `output/sensor/*.csv`
- `output/sensor_stats.csv`

3. `step3_audio.py`
- `output/audio/*.npz`

4. `step4_video.py`
- `output/frames/<run_id>/*.jpg`
- `output/frames/<run_id>/frame_stats.csv`
- `output/video_stats.csv`

5. `step5_align.py`
- `output/aligned/*_sensor_31hz.csv`
- `output/alignment_summary.csv`

6. `step6_dataset.py`
- `output/dataset/chunks/*.npz`
- `output/dataset/manifest.csv`
- `output/dataset/split_dict.json`

7. `step7_tabular_baseline.py`
- `output/tabular/model_lgb.pkl`
- `output/tabular/val_predictions.csv`
- `output/tabular/val_metrics.json`
- `output/tabular/class_weights.json`
- `output/tabular/feature_importance.csv`

8. `step11_train.py`
- `output/checkpoints/best_model.pt`
- `output/checkpoints/training_log.json`
- `output/checkpoints/training_summary.json`

9. `step12_calibrate.py`
- `output/checkpoints/calibration_report.json`
- updates `output/checkpoints/best_model.pt` with temperature metadata

10. `step13_evaluate.py`
- `output/evaluation/val_metrics.json`
- `output/evaluation/val_predictions.csv`
- `output/evaluation/per_class_report.csv`
- `output/evaluation/confusion_matrix_mc.png`
- `output/evaluation/confusion_matrix_binary.png`

11. `step14_inference.py`
- `output/inference/predictions.csv`
- `output/inference/metrics.json` (when labels are available)
- `output/inference/confusion_matrix.png` (when labels are available)

12. `step15_sanity_leakage.py`
- `output/leakage_report.json`

## Submission Scripts

- `generate_final_submission.py` -> `submission.csv` (default output path)
- `optimize_postprocessing.py` -> `submission_optimized.csv` (default output path)
- `fix_submission.py` -> `submission_fixed.csv` (default output path)
