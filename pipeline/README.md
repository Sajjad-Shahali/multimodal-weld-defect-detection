# Pipeline

This package contains the end-to-end machine learning workflow for WeldFusionNet, from raw multimodal weld runs to calibrated predictions and final submission files.

It is organized as a step-based pipeline so data preparation, modeling, evaluation, and inference can be run independently or as one reproducible flow.

## Core stack

- `Python` for orchestration and data processing
- `PyTorch` for the multimodal neural model
- `torchvision` for the MobileNetV3 video backbone
- `LightGBM` for the tabular baseline
- `scikit-learn` for splits and evaluation utilities
- `OpenCV` for video metadata, frame extraction, and lazy decoding
- `librosa` and `soundfile` for audio feature extraction
- `pandas` and `numpy` for tabular and numerical processing

## What the pipeline does

- validates raw runs across sensor, audio, video, and image files
- builds enriched sensor features and run-level sensor statistics
- extracts audio features such as MFCCs and spectral descriptors
- indexes video frames and computes lightweight video statistics
- aligns modalities on a common timeline
- chunks runs into model-ready training windows
- trains a LightGBM baseline on run-level tabular features
- trains WeldFusionNet for multimodal defect classification
- calibrates confidence with temperature scaling
- evaluates validation performance and generates inference outputs

## Model overview

The main neural model is `WeldFusionNet`, a multimodal architecture for automated weld defect classification.

### Input branches

- Sensor branch: multivariate time-series data from weld process CSV logs
- Audio branch: frame-level acoustic features derived from FLAC recordings
- Video branch: sampled weld-pool frames decoded from AVI recordings

### Feature representation

- Sensor tensors are aligned to a fixed master timeline and stored as `(25, N_sensor)` chunks.
- Audio tensors are built from `13` MFCC coefficients plus `5` frame-level spectral features, giving `18` channels per time step.
- Video is represented lazily through frame indices during preprocessing, then decoded at training or inference time.

### WeldFusionNet architecture

- Sensor encoder: `Conv1dEncoder` for short multivariate time series
- Audio encoder: `Conv1dEncoder` for compact temporal acoustic features
- Video encoder: `MobileNetV3-Small` with temporal attention pooling
- Fusion head: default concatenation plus MLP
- Optional fusion upgrade: transformer-based token fusion
- Output head 1: multiclass prediction for weld defect type
- Output head 2: binary auxiliary prediction for good-vs-defect supervision

The current class set used for training is:

- `good_weld`
- `excessive_penetration`
- `burn_through`
- `overlap`
- `lack_of_fusion`
- `excessive_convexity`
- `crater_cracks`

## Machine learning design choices

- `LightGBM` provides a fast tabular baseline over run-level sensor summaries.
- `FocalLoss` emphasizes harder and minority examples in the multiclass task.
- `BCEWithLogitsLoss` is combined with focal loss for multi-task learning.
- `AdamW` is used as the optimizer for the neural model.
- `OneCycleLR` is used for training-time learning-rate scheduling.
- `Early stopping` is driven by validation score.
- `Temperature scaling` is applied after training to reduce calibration error.
- `StratifiedGroupKFold` is used to split by run while preserving label balance.
- `Lazy video decoding` avoids storing all frames inside dataset chunks and keeps storage manageable.

## Data flow by phase

### Phase 1: preprocessing and dataset building

- `step1_validate.py` inventories raw runs and checks modality health.
- `step2_sensor.py` parses CSV files, detects the weld-active window, and creates derived sensor features.
- `step3_audio.py` extracts mel, MFCC, and frame-level spectral features from FLAC files.
- `step4_video.py` extracts preview frames and simple video statistics for analysis.
- `step5_align.py` aligns modalities in time.
- `step6_dataset.py` builds the chunked multimodal dataset and train-validation split.

### Phase 2: training

- `step7_tabular_baseline.py` trains the LightGBM baseline from run-level sensor statistics.
- `step8_dataset_torch.py` builds PyTorch datasets and dataloaders from chunk files.
- `step9_model.py` defines WeldFusionNet and the optional fusion variants.
- `step10_losses.py` defines focal loss and the multi-task loss wrapper.
- `step11_train.py` runs end-to-end neural training.

### Phase 3: calibration, evaluation, and inference

- `step12_calibrate.py` learns a temperature scalar on validation logits.
- `step13_evaluate.py` computes validation reports and confusion matrices.
- `step14_inference.py` runs predictions on test or external runs.
- `step15_sanity_leakage.py` performs leakage checks and audit-style validation.

## Important technical details

- The master fusion timeline is built at `25 Hz`.
- Each prediction chunk represents `1 second`, or `25` aligned frames.
- Audio features are aligned by nearest feature frame.
- Sensor features are aligned by linear interpolation.
- Video is aligned by storing source frame indices rather than pre-decoded frame tensors.
- The default video backbone is `MobileNetV3-Small`, with a tiny CNN fallback for lighter environments.

## Running the pipeline

Common entry point:

```bash
python -m pipeline.run_all
```

Typical examples:

```bash
python -m pipeline.run_all --steps 1 2 3 4 5 6
python -m pipeline.run_all --steps 7 11 12 13 14
python -m pipeline.run_all --steps 11 12 13 --use-video
python -m pipeline.run_all --steps 14 --test-dir sampleData/08-17-22-0011-00
```

## Outputs

Most generated artifacts are written under the repository `output/` folder.

For a step-to-file mapping, see [`../docs/outputs_map.md`](../docs/outputs_map.md).

## Notes for contributors

- Steps `8`, `9`, and `10` are library-style modules that support step `11`.
- The pipeline is designed so each stage can be rerun independently after upstream outputs exist.
- Configuration lives in the repository root `config.yaml`, which controls paths, model settings, calibration, and inference behavior.
