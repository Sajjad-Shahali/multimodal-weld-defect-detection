# weldml

Python package powering the weld defect detection dashboard. Provides run discovery, feature extraction, derived signal computation, and the Streamlit multi-page app.

---

## Package Structure

```text
weldml/
├── __init__.py
├── dashboard/
│   ├── __init__.py
│   └── app.py             # Streamlit multi-page dashboard (~1300 lines)
├── features/
│   ├── __init__.py
│   ├── extract.py          # Feature extraction driver
│   └── derived_signals.py  # Real-time audio/video signal computation
├── data/
│   ├── __init__.py
│   └── indexer.py          # Run discovery, metadata, JSONL/CSV output
├── models/
│   └── __init__.py         # (Reserved for model loading helpers)
└── utils/
    ├── __init__.py
    └── config.py           # YAML config loader
```

---

## Submodules

### `weldml.dashboard`

Multi-page Streamlit application. See [`dashboard/README.md`](dashboard/README.md).

Entry point:
```bash
streamlit run src/weldml/dashboard/app.py -- --config configs/default.yaml
```

### `weldml.features`

Feature extraction and derived signal computation. See [`features/README.md`](features/README.md).

Key functions:
- `extract.extract_features(cfg)` — compute per-run features from the index, write to parquet
- `derived_signals.get_derived_signals(audio_path, video_path, cache_dir, run_id)` — compute or load cached audio/video overlays

### `weldml.data`

Run discovery and inventory generation. See [`data/README.md`](data/README.md).

Key functions:
- `indexer.discover_runs(root_dir, split)` — scan folder structure, return list of `RunRecord`
- `indexer.discover_runs_from_manifest(manifest_path, split_dict_path)` — build from pipeline manifest
- `indexer.enrich_media_meta(record)` — populate sensor/audio/video duration and metadata
- `indexer.write_jsonl(records, out_path)` — write JSONL index
- `indexer.write_inventory_csv(records, out_path)` — write CSV inventory

### `weldml.models`

Reserved for model loading and inference helpers. Currently empty — the live inference in `app.py` loads the model directly via PyTorch.

### `weldml.utils`

Configuration utilities. See [`utils/README.md`](utils/README.md).

Key functions:
- `config.load_config(path)` — load a YAML file and return a dict

---

## Data Flow

```text
Raw run folders  ──► indexer.discover_runs()
                  ──► indexer.enrich_media_meta()
                  ──► write_jsonl() / write_inventory_csv()
                              │
                        run_index.jsonl
                              │
                  ──► features.extract_features()
                              │
                        features.parquet
                              │
                  ──► dashboard/app.py
                       ├── reads all outputs/  artifacts
                       ├── derived_signals for Live Sync
                       └── renders 14 analysis pages
```
