# weldml/dashboard/

Streamlit multi-page dashboard for the multimodal weld defect detection project. Visualizes all pipeline outputs from dataset statistics through final model evaluation and live inference.

---

## Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit application — ~1300 lines, 14 analysis pages |

---

## Launch

```bash
# Via the launch script (recommended)
python scripts/run_dashboard.py --config configs/default.yaml

# Direct Streamlit invocation
streamlit run src/weldml/dashboard/app.py -- --config configs/default.yaml
```

Dashboard available at `http://localhost:8501`.

---

## Dashboard Pages

### 1. Dataset Stats

Loads from `data/interim/` (manifest, dataset_meta, norm_stats).

- Manifest summary — total runs, split counts
- Class distribution bar chart (train / val / test)
- Normalization statistics (mean/std per feature channel)
- Split details table

### 2. Sensor Stats

Loads from `data/interim/sensor_stats.csv`.

- Per-run sensor aggregate table (AgGrid)
- Duration distribution histogram
- Per-sensor mean/std analysis
- Correlation heatmap (Plotly)

### 3. Tabular Baseline (Step 7)

Loads from `outputs/tabular/`.

- LightGBM validation metrics (Binary F1, Macro F1, Final Score)
- Feature importance bar chart
- Validation predictions table with label columns
- Class weights visualization

### 4. WeldFusionNet Training (Step 11)

Loads from `outputs/checkpoints/training_log.json`.

- Training and validation loss curves
- Binary F1 and Macro F1 over epochs
- Learning rate schedule
- Per-epoch metrics table (AgGrid)

### 5. Temperature Calibration (Step 12)

Loads from `outputs/checkpoints/calibration_*.json`.

- ECE (Expected Calibration Error) before vs. after calibration
- Calibrated temperature value
- Calibration method and bin count

### 6. Validation Results (Step 13)

Loads from `outputs/eval/`.

- Overall metrics (Binary F1, Macro F1, Final Score)
- 7-class confusion matrix (Plotly heatmap)
- Binary confusion matrix
- Per-class precision / recall / F1 table

### 7. Holdout Test (Step 14)

Loads from `outputs/predictions/predictions.csv`.

- Test-set metrics if ground truth is available
- Per-class F1 bar chart
- Full predictions table (AgGrid)

### 8. Model Comparison

Combines outputs from Step 7 (tabular) and Step 13 (neural).

- LightGBM vs WeldFusionNet side-by-side metric table
- Binary F1 and Macro F1 comparison bar chart

### 9. 3D Explorer

Loads from `outputs/reports/dashboards/points_3d.json`.

- Interactive 3D scatter plot (Plotly)
- Axes: sensor mean / sensor std / duration (or similar run-level stats)
- Color-coded by defect class
- Hover shows run ID and label

### 10. Live Sync

Loads from `configs.dashboard.sample_data` (default: `assets/sample_data/`).

- Dropdown to select weld run
- HTML5 video player (AVI auto-converted to MP4 via ffmpeg)
- Sensor timeline chart synchronized to video playback
- JavaScript interpolation updates sensor cursor in real time as video plays
- Derived signal overlays: audio RMS, audio spectral centroid, video brightness, video motion energy

> Requires **ffmpeg** for AVI → MP4 conversion. Without it, video will not play in the browser.

### 11. Class Distribution

Loads from manifest and `outputs/tabular/class_weights.json`.

- Class count bar chart
- Pie chart of class proportions
- Class weights table (as used in the loss function)

### 12. Feature Insights

Loads from `data/processed/dashboard_top15_features.csv` and `outputs/tabular/feature_importance.csv`.

- Top-15 features bar chart
- Feature correlation heatmap
- Tabular baseline summary stats

### 13. Model Performance Insights

Combines tabular and neural evaluation outputs.

- Per-class precision / recall / F1 grouped bar chart
- LightGBM vs WeldFusionNet F1 comparison
- Summary metrics table

### 14. Inference

Supports two inference modes using the live `best_model.pt` checkpoint:

- **Batch mode** — scan a directory of run folders, run model on each
- **Upload mode** — upload individual files (AVI, FLAC, CSV) for single-run prediction

Outputs predicted defect type and confidence score.

---

## Key Technical Details

### Caching
Data loading functions are decorated with `@st.cache_data` to avoid re-reading files on every interaction.

### Video Compatibility
The dashboard auto-converts AVI to MP4 using a subprocess ffmpeg call before embedding the video as Base64. The converted MP4 is cached to disk.

### Live Sync JavaScript
The sensor timeline chart updates in real time using a JavaScript `requestAnimationFrame` loop that reads `video.currentTime` and interpolates the sensor value at that timestamp. The chart and cursor are injected as an HTML component via `st.components.v1.html`.

### Styling
- Plotly charts use the `plotly_dark` template
- AgGrid tables use custom CSS for dark-mode compatibility
- Navigation uses `streamlit-option-menu` for a sidebar tab interface

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web framework |
| `plotly` | Interactive charts |
| `st-aggrid` | Advanced sortable/filterable data tables |
| `streamlit-option-menu` | Sidebar navigation menu |
| `pandas` / `numpy` | Data manipulation |
| `librosa` | Audio feature extraction (derived signals) |
| `soundfile` | FLAC audio reading |
| `opencv-python` | Video frame extraction (derived signals) |
| `torch` / `torchvision` | Live inference (model loading) |
