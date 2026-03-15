# scripts/

Entry-point scripts for building the data index, extracting features, and launching the dashboard. All scripts accept a `--config` argument pointing to the YAML configuration file.

---

## Files

| Script | Description |
|---|---|
| `build_index.py` | Discover weld runs → write `run_index.jsonl` + `inventory.csv` |
| `extract_features.py` | Run feature extraction → write `features.parquet` |
| `build_top15_features.py` | (Placeholder) Build top-15 feature dataset for the Feature Insights panel |
| `run_dashboard.py` | Launch the Streamlit dashboard |

---

## build_index.py

Scans the raw data folders (or reads from a manifest) to produce:
- `data/interim/index/run_index.jsonl` — one JSON object per run with full file paths and media metadata
- `data/interim/index/inventory.csv` — one row per run with health flags and duration info

### Usage

```bash
# From raw folder structure (good_weld / defect_data_weld subfolders)
python scripts/build_index.py --config configs/default.yaml

# From pipeline manifest + split_dict (recommended after running the full pipeline)
python scripts/build_index.py --config configs/default.yaml --from-manifest
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Path to the YAML configuration file |
| `--from-manifest` | off | Use `manifest.csv` + `split_dict.json` instead of scanning raw folders |

### When to use `--from-manifest`

Use this mode when you have already run the main pipeline and have:
- `data/interim/manifest.csv`
- `data/interim/split_dict.json`

This preserves the exact train/val/test split from the pipeline run instead of re-discovering splits from folder structure.

---

## extract_features.py

Reads the JSONL run index and computes per-run features, writing the result to `data/processed/features/features.parquet`.

> Note: The feature extractor in `src/weldml/features/extract.py` is a stub that returns basic run metadata. Extend it with domain-specific feature computation as needed.

### Usage

```bash
python scripts/extract_features.py --config configs/default.yaml
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Path to the YAML configuration file |

---

## build_top15_features.py

Placeholder for building the top-15 feature dataset used in the dashboard's **Feature Insights** panel. Not yet implemented — the panel reads from `data/processed/dashboard_top15_features.csv` if it exists.

---

## run_dashboard.py

Launches the Streamlit dashboard by invoking `streamlit run` on `src/weldml/dashboard/app.py`.

### Usage

```bash
python scripts/run_dashboard.py --config configs/default.yaml
```

The dashboard will be available at `http://localhost:8501`.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Path passed to the Streamlit app via `--` separator |

### Equivalent direct command

```bash
streamlit run src/weldml/dashboard/app.py -- --config configs/default.yaml
```

---

## Typical Workflow

```bash
# 1. Build index from raw data
python scripts/build_index.py --config configs/default.yaml

# 2. Extract features
python scripts/extract_features.py --config configs/default.yaml

# 3. Launch dashboard
python scripts/run_dashboard.py --config configs/default.yaml
```

Or, after a full pipeline run:

```bash
# 1. Build index from pipeline manifest
python scripts/build_index.py --config configs/default.yaml --from-manifest

# 2. Launch dashboard (pipeline artifacts are loaded automatically)
python scripts/run_dashboard.py --config configs/default.yaml
```
