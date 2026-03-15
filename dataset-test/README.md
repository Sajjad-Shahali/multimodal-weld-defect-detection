# Dataset Test Sample

This folder contains a small representative sample of the weld dataset used for local development, debugging, demo runs, and reproducible examples.

The full dataset is much larger, so only several items are included in the repository to keep the project portable and Git-friendly.

## Why this subset exists

- The original raw dataset is too large to commit safely to GitHub.
- A smaller sample is enough to validate preprocessing, feature extraction, training logic, and dashboard behavior.
- The default `config.yaml` points to this folder so the pipeline can be exercised without requiring the full competition dataset.

## Current contents

- `good-weld-test/` contains sample good weld runs.
- `defect-weld-test/` contains sample defect weld runs.
- The current repository snapshot includes `24` run folders in total.
- `12` runs are good weld examples.
- `12` runs are defect examples.

## Defect categories represented here

- `burn_through`
- `crater_cracks`
- `excessive_convexity`
- `lack_of_fusion`
- `overlap`

This sample is useful for smoke tests, but it should not be treated as the full training or evaluation distribution.

## Run folder format

Each weld run folder follows the same raw-data pattern expected by the pipeline:

```text
<run_id>/
|- <run_id>.csv
|- <run_id>.flac
|- <run_id>.avi
|- images/
   |- *.jpg
```

The three main modalities are:

- sensor CSV
- audio FLAC
- video AVI

## How the pipeline uses this folder

- `pipeline.step1_validate` inventories the files and checks modality health.
- `pipeline.step2_sensor` reads the CSV data and derives sensor features.
- `pipeline.step3_audio` extracts MFCC and spectral features from FLAC audio.
- `pipeline.step4_video` extracts exploratory video frames and simple frame statistics.
- `pipeline.step6_dataset` aligns all modalities into chunked training examples.

If you switch to the full dataset later, update `data_root` in `config.yaml`.
