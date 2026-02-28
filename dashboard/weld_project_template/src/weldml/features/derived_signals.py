"""Compute derived signals from audio/video for dashboard overlays."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def compute_audio_rms(audio_path: Path, sr: int | None = None, hop_ms: float = 50) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_sec, rms) arrays. t is center time of each hop window."""
    try:
        import soundfile as sf
        import librosa
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return np.array([]), np.array([])
        y, file_sr = sf.read(str(audio_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr is None:
            sr = file_sr
        if sr != file_sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        hop = max(1, int(sr * hop_ms / 1000))
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        t = np.arange(len(rms)) * hop / sr
        return t, rms.astype(np.float32)
    except Exception:
        return np.array([]), np.array([])


def compute_audio_spectral_centroid(audio_path: Path, sr: int | None = None, hop_ms: float = 50) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_sec, centroid_hz) arrays."""
    try:
        import soundfile as sf
        import librosa
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return np.array([]), np.array([])
        y, file_sr = sf.read(str(audio_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr is None:
            sr = file_sr
        if sr != file_sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        hop = max(1, int(sr * hop_ms / 1000))
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        t = np.arange(len(cent)) * hop / sr
        return t, cent.astype(np.float32)
    except Exception:
        return np.array([]), np.array([])


def compute_video_brightness(video_path: Path, sample_fps: float = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_sec, brightness) arrays. Brightness = mean grayscale value per frame."""
    try:
        import cv2
        if not video_path.exists() or video_path.stat().st_size == 0:
            return np.array([]), np.array([])
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_frames == 0:
            cap.release()
            return np.array([]), np.array([])
        step = max(1, int(fps / sample_fps))
        times, bright = [], []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bright.append(float(np.mean(gray)))
                times.append(idx / fps)
            idx += 1
        cap.release()
        return np.array(times, dtype=np.float32), np.array(bright, dtype=np.float32)
    except Exception:
        return np.array([]), np.array([])


def compute_video_motion(video_path: Path, sample_fps: float = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_sec, motion) arrays. Motion = mean absolute frame difference."""
    try:
        import cv2
        if not video_path.exists() or video_path.stat().st_size == 0:
            return np.array([]), np.array([])
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        step = max(1, int(fps / sample_fps))
        times, motion = [], []
        prev_gray = None
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is not None and idx % step == 0:
                diff = np.abs(gray - prev_gray)
                motion.append(float(np.mean(diff)))
                times.append(idx / fps)
            prev_gray = gray
            idx += 1
        cap.release()
        return np.array(times, dtype=np.float32), np.array(motion, dtype=np.float32)
    except Exception:
        return np.array([]), np.array([])


def get_derived_signals(
    audio_path: Path,
    video_path: Path,
    cache_dir: Path,
    run_id: str,
    hop_ms: float = 50,
    video_sample_fps: float = 5,
) -> dict:
    """
    Compute or load from cache: audio_rms, audio_centroid, video_brightness, video_motion.
    Returns dict with keys -> (t, values) tuples.
    """
    cache_dir = cache_dir / run_id.replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    def _cached(name: str, compute_fn, *args, **kwargs):
        cache_file = cache_dir / f"{name}.npz"
        if cache_file.exists():
            try:
                data = np.load(cache_file)
                return data["t"], data["v"]
            except Exception:
                pass
        t, v = compute_fn(*args, **kwargs)
        if len(t) > 0:
            np.savez_compressed(cache_file, t=t, v=v)
        return t, v

    if audio_path.exists() and audio_path.stat().st_size > 0:
        out["audio_rms"] = _cached("audio_rms", compute_audio_rms, audio_path, None, hop_ms)
        out["audio_centroid"] = _cached("audio_centroid", compute_audio_spectral_centroid, audio_path, None, hop_ms)

    if video_path.exists() and video_path.stat().st_size > 0:
        out["video_brightness"] = _cached("video_brightness", compute_video_brightness, video_path, video_sample_fps)
        out["video_motion"] = _cached("video_motion", compute_video_motion, video_path, video_sample_fps)

    return out
