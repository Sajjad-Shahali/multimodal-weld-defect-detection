# assets/

Static assets bundled with the dashboard template.

---

## Structure

```text
assets/
└── sample_data/          # 10 pre-loaded weld run samples for Live Sync
    ├── 08-17-22-0011-00/ # good_weld sample
    ├── 08-17-22-0012-00/
    ├── 08-17-22-0013-00/
    ├── 08-17-22-0014-00/
    ├── 08-17-22-0015-00/
    ├── 08-17-22-0016-00/
    ├── 08-17-22-0017-00/
    ├── 08-17-22-0018-00/
    ├── 08-17-22-0019-00/
    └── 08-18-22-0020-00/
```

---

## sample_data/

Ten pre-loaded weld run folders used by the **Live Sync** dashboard page. Each run folder contains:

| File | Description |
|---|---|
| `weld.avi` | Video of the weld pool and arc |
| `sensor.csv` | Timestamped sensor readings with `Date` and `Time` columns |
| `weld.flac` | Audio recording of the weld process |
| `images/` | Static JPG snapshots of the weld |

### Run ID Format

```
{MM}-{DD}-{YY}-{sequence}-{label_code}
```

| Part | Example | Description |
|---|---|---|
| Date | `08-17-22` | Month-Day-Year |
| Sequence | `0011` | Run sequence number |
| Label | `00` | Defect type code (see label map below) |

### Label Map

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

## Adding New Sample Runs

Any folder placed under `assets/sample_data/` that contains a `.avi` and a `sensor.csv` with `Date` and `Time` columns will automatically appear in the **Live Sync** dropdown.

The dashboard config key that controls the sample data path is:

```yaml
dashboard:
  sample_data: assets/sample_data
```

Update this path in `configs/default.yaml` to point to a different sample directory.

---

## Notes on Video Playback

Browsers do not natively support AVI. The dashboard automatically converts AVI files to MP4 when **ffmpeg** is installed. If ffmpeg is not available, the video will not play in the browser.

Install ffmpeg:
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
