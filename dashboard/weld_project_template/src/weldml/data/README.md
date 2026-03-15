# weldml/data/

Run discovery, media metadata enrichment, and index/inventory generation.

---

## Files

| File | Description |
|---|---|
| `indexer.py` | Core module — scan run folders or manifest, produce JSONL index and CSV inventory |

---

## indexer.py

### Data Model: `RunRecord`

A dataclass representing a single weld run with all associated file paths and metadata.

```python
@dataclass
class RunRecord:
    run_id: str          # e.g. "08-17-22-0011-00"
    split: str           # "train", "val", or "test"
    label: str | None    # label code e.g. "00", "07" (None for test runs)
    sensor_csv: str      # absolute path to sensor.csv
    audio_flac: str      # absolute path to weld.flac
    video_avi: str       # absolute path to weld.avi
    images_dir: str      # absolute path to images/ folder
    media_meta: dict     # populated by enrich_media_meta()
    alignment: dict      # reserved for alignment metadata
```

---

### Functions

#### `discover_runs(root_dir, split, label_from_folder=True) -> list[RunRecord]`

Scans a directory of run folders and returns a list of `RunRecord` objects.

**Label extraction:** The label is read from the last two characters of the folder name (e.g., `08-17-22-0011-00` → label `"00"`). Set `label_from_folder=False` for test folders without labels.

**Expected folder contents:**

| File | Description |
|---|---|
| `sensor.csv` | Sensor data |
| `weld.flac` | Audio |
| `weld.avi` | Video |
| `images/` | Static image snapshots |

```python
from weldml.data.indexer import discover_runs
from pathlib import Path

records = discover_runs(Path("data/raw/train/good_weld"), split="train")
```

---

#### `discover_runs_from_manifest(manifest_path, split_dict_path, data_root=None) -> list[RunRecord]`

Builds run records from a `manifest.csv` + `split_dict.json` produced by the main pipeline. Preserves the exact train/val/test split assignments from the pipeline run.

**Manifest columns required:**
- `run_id` — run identifier
- `avi_path` — path to the AVI file
- `label_code` or `label` — defect class code
- `split` — fallback split if run is not in split_dict

**Split dict format:**
```json
{
  "train": ["run_id_1", "run_id_2", ...],
  "val":   ["run_id_3", ...],
  "test":  ["run_id_4", ...]
}
```

**Path rewriting:** If `data_root` is provided and manifest paths are absolute (starting with `/`), paths are rewritten by finding the `good_weld/`, `defect_data_weld/`, or `test_data/` marker in the path and prepending `data_root`.

```python
from weldml.data.indexer import discover_runs_from_manifest
from pathlib import Path

records = discover_runs_from_manifest(
    manifest_path=Path("data/interim/manifest.csv"),
    split_dict_path=Path("data/interim/split_dict.json"),
    data_root=Path("/new/data/location"),
)
```

---

#### `enrich_media_meta(record: RunRecord) -> None`

Populates `record.media_meta` with duration and sample rate information by reading the actual media files.

**Populated fields:**

| Field | Source | Description |
|---|---|---|
| `sensor_duration_s` | CSV `Date`+`Time` columns | Total sensor recording duration (seconds) |
| `sensor_sample_rate_hz` | CSV timestamp deltas | Median sample rate (Hz) |
| `audio_duration_s` | soundfile header | Audio duration (seconds) |
| `audio_sample_rate_hz` | soundfile header | Audio sample rate (Hz) |
| `video_fps` | cv2 CAP_PROP_FPS | Video frame rate |
| `video_n_frames` | cv2 CAP_PROP_FRAME_COUNT | Total frame count |
| `video_duration_s` | n_frames / fps | Video duration (seconds) |

```python
from weldml.data.indexer import enrich_media_meta

for record in records:
    enrich_media_meta(record)
    print(record.media_meta)
```

---

#### `write_jsonl(records, out_path) -> None`

Writes one JSON object per line to the JSONL index file.

```python
write_jsonl(records, Path("data/interim/index/run_index.jsonl"))
```

---

#### `write_inventory_csv(records, out_path) -> None`

Writes a CSV inventory with one row per run.

**Output columns:**

| Column | Description |
|---|---|
| `run_id` | Run identifier |
| `split` | train / val / test |
| `label` | Defect class code |
| `has_sensor` | True if sensor.csv exists |
| `has_audio` | True if weld.flac exists |
| `has_video` | True if weld.avi exists |
| `n_images` | Number of JPG files in images/ |
| `sensor_duration_s` | Sensor duration (seconds) |
| `audio_duration_s` | Audio duration (seconds) |
| `video_duration_s` | Video duration (seconds) |

```python
write_inventory_csv(records, Path("data/interim/index/inventory.csv"))
```

---

## Typical Usage

```python
from weldml.data.indexer import (
    discover_runs, discover_runs_from_manifest,
    enrich_media_meta, write_jsonl, write_inventory_csv
)
from pathlib import Path

# Discover
records = discover_runs(Path("data/raw/train/good_weld"), split="train")
records += discover_runs(Path("data/raw/train/defect_weld"), split="train")

# Enrich with media metadata
for r in records:
    enrich_media_meta(r)

# Write outputs
write_jsonl(records, Path("data/interim/index/run_index.jsonl"))
write_inventory_csv(records, Path("data/interim/index/inventory.csv"))
```
