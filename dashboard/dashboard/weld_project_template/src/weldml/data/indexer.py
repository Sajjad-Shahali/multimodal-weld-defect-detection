from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import csv
import pandas as pd

@dataclass
class RunRecord:
    run_id: str
    split: str  # train|test
    label: str | None
    sensor_csv: str
    audio_flac: str
    video_avi: str
    images_dir: str
    media_meta: dict
    alignment: dict

def discover_runs(root_dir: Path, split: str, label_from_folder: bool = True) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not root_dir.exists():
        return records
    for run_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        run_id = run_dir.name
        label = None
        if label_from_folder and "-" in run_id:
            label = run_id.split("-")[-1]
        records.append(RunRecord(
            run_id=run_id,
            split=split,
            label=label if split == "train" else None,
            sensor_csv=str(run_dir / "sensor.csv"),
            audio_flac=str(run_dir / "weld.flac"),
            video_avi=str(run_dir / "weld.avi"),
            images_dir=str(run_dir / "images"),
            media_meta={},
            alignment={},
        ))
    return records


def discover_runs_from_manifest(
    manifest_path: Path,
    split_dict_path: Path,
    data_root: str | Path | None = None,
) -> list[RunRecord]:
    """
    Build run records from manifest.csv and split_dict.json.
    Expects manifest columns: run_id, label_code, avi_path, split.
    Raw files per run: {run_id}.csv, {run_id}.flac, {run_id}.avi, images/*.jpg
    """
    import pandas as pd
    records: list[RunRecord] = []
    if not manifest_path.exists():
        return records

    df = pd.read_csv(manifest_path)
    if "run_id" not in df.columns or "avi_path" not in df.columns:
        return records

    # One row per run (take first chunk's info)
    run_info = df.groupby("run_id").first().reset_index()
    run_info["label_code"] = run_info.get("label_code", run_info.get("label", "")).astype(str).str.zfill(2)

    split_dict: dict = {}
    if split_dict_path.exists():
        with split_dict_path.open("r", encoding="utf-8") as f:
            split_dict = json.load(f)

    data_root = Path(data_root) if data_root else None

    for _, row in run_info.iterrows():
        run_id = str(row["run_id"])
        avi_path = Path(str(row["avi_path"]).strip())
        label = str(row.get("label_code", row.get("label", ""))).zfill(2) if pd.notna(row.get("label_code", row.get("label"))) else ""
        manifest_split = str(row.get("split", "train"))

        if data_root and str(avi_path).startswith("/"):
            # Replace path: find good_weld/defect_data_weld in path, use from there
            s = str(avi_path)
            for marker in ["good_weld/", "defect_data_weld/", "test_data/"]:
                if marker in s:
                    idx = s.index(marker)
                    rel = s[idx:]
                    avi_path = data_root / rel
                    break

        run_dir = avi_path.parent
        sensor_csv = run_dir / f"{run_id}.csv"
        audio_flac = run_dir / f"{run_id}.flac"
        video_avi = str(avi_path)
        images_dir = run_dir / "images"

        split = "train"
        if split_dict:
            if run_id in split_dict.get("val", []):
                split = "val"
            elif run_id in split_dict.get("train", []):
                split = "train"
            elif run_id in split_dict.get("test", []):
                split = "test"
            else:
                split = manifest_split

        records.append(RunRecord(
            run_id=run_id,
            split=split,
            label=label or None,
            sensor_csv=str(sensor_csv),
            audio_flac=str(audio_flac),
            video_avi=video_avi,
            images_dir=str(images_dir),
            media_meta={},
            alignment={},
        ))
    return records


def _get_sensor_meta(sensor_path: Path) -> dict:
    meta = {}
    try:
        if not sensor_path.exists() or sensor_path.stat().st_size == 0:
            return meta
        df = pd.read_csv(sensor_path)
        if "Date" in df.columns and "Time" in df.columns:
            ts = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
            t0 = ts.dropna().iloc[0] if ts.notna().any() else None
            if t0 is not None:
                t_sec = (ts - t0).dt.total_seconds()
                meta["sensor_duration_s"] = float(t_sec.max())
                dt = t_sec.diff().dropna()
                if len(dt) > 0 and (dt > 0).any():
                    meta["sensor_sample_rate_hz"] = float(1.0 / dt[dt > 0].median())
        if "sensor_duration_s" not in meta and len(df) > 0:
            meta["sensor_duration_s"] = float(len(df))  # fallback: assume 1 sample/sec
    except Exception:
        pass
    return meta


def _get_audio_meta(audio_path: Path) -> dict:
    meta = {}
    try:
        import soundfile as sf
        if audio_path.exists() and audio_path.stat().st_size > 0:
            info = sf.info(str(audio_path))
            meta["audio_duration_s"] = info.duration
            meta["audio_sample_rate_hz"] = info.samplerate
    except Exception:
        pass
    return meta


def _get_video_meta(video_path: Path) -> dict:
    meta = {}
    try:
        import cv2
        if video_path.exists() and video_path.stat().st_size > 0:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps and fps > 0:
                meta["video_fps"] = float(fps)
            if n_frames and n_frames > 0:
                meta["video_n_frames"] = int(n_frames)
                if "video_fps" in meta:
                    meta["video_duration_s"] = n_frames / meta["video_fps"]
    except Exception:
        pass
    return meta


def enrich_media_meta(record: RunRecord) -> None:
    """Populate media_meta and alignment with computed metadata."""
    record.media_meta = {
        **_get_sensor_meta(Path(record.sensor_csv)),
        **_get_audio_meta(Path(record.audio_flac)),
        **_get_video_meta(Path(record.video_avi)),
    }


def write_jsonl(records: list[RunRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")

def write_inventory_csv(records: list[RunRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "split", "label", "has_sensor", "has_audio", "has_video", "n_images",
        "sensor_duration_s", "audio_duration_s", "video_duration_s",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            run_dir = Path(r.sensor_csv).parent
            images_dir = Path(r.images_dir)
            meta = r.media_meta or {}
            w.writerow({
                "run_id": r.run_id,
                "split": r.split,
                "label": r.label or "",
                "has_sensor": Path(r.sensor_csv).exists(),
                "has_audio": Path(r.audio_flac).exists(),
                "has_video": Path(r.video_avi).exists(),
                "n_images": len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0,
                "sensor_duration_s": meta.get("sensor_duration_s", ""),
                "audio_duration_s": meta.get("audio_duration_s", ""),
                "video_duration_s": meta.get("video_duration_s", ""),
            })
