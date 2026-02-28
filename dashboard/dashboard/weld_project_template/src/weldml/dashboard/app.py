from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import base64
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

from weldml.utils.config import load_config
from weldml.features.derived_signals import get_derived_signals

# UI add-ons
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Label map for defect classes (dashboard2 alignment)
LABEL_MAP = {
    "00": "good_weld",
    "01": "excessive_penetration",
    "02": "burn_through",
    "06": "overlap",
    "07": "lack_of_fusion",
    "08": "excessive_convexity",
    "11": "crater_cracks",
}

# Distinct colors per label for 3D Explorer (avoids collisions from hue stepping)
LABEL_COLORS = {
    "good_weld": "#22c55e",           # green
    "burn_through": "#ef4444",        # red
    "excessive_penetration": "#f59e0b",  # amber
    "overlap": "#8b5cf6",             # violet
    "lack_of_fusion": "#06b6d4",      # cyan
    "excessive_convexity": "#ec4899", # pink
    "crater_cracks": "#eab308",       # yellow
}


def _exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def _label_name(code: str) -> str:
    return LABEL_MAP.get(str(code).zfill(2), f"class_{code}")

# Custom CSS for AgGrid (injected into iframe - required for styling to apply)
AGGRID_CUSTOM_CSS = """
.ag-root-wrapper { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.3); font-family: 'Inter', -apple-system, sans-serif; }
.ag-header, .ag-header-row { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important; color: #0f172a !important; }
.ag-header-cell { font-weight: 600 !important; font-size: 0.85rem !important; padding: 12px 16px !important; }
.ag-row.ag-row-odd { background: rgba(30, 41, 59, 0.6) !important; }
.ag-row.ag-row-even { background: rgba(15, 23, 42, 0.8) !important; }
.ag-row:hover { background: rgba(245, 158, 11, 0.15) !important; }
.ag-cell { border-color: rgba(245, 158, 11, 0.15) !important; padding: 10px 16px !important; font-size: 0.9rem !important; }
"""

# -----------------------------
# Helpers
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def load_inventory(inv_path: Path) -> pd.DataFrame:
    return pd.read_csv(inv_path)


@st.cache_data(show_spinner=False)
def load_index_jsonl(index_path: Path) -> pd.DataFrame:
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_dataset_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_label_name(meta: dict | None, label_code: str) -> str:
    """Map label code (e.g. '00') to class name using dataset_meta."""
    if not meta or not label_code:
        return str(label_code)
    codes = meta.get("class_codes", [])
    names = meta.get("class_names", [])
    try:
        idx = codes.index(str(label_code).zfill(2))
        return names[idx] if idx < len(names) else str(label_code)
    except (ValueError, IndexError):
        return str(label_code)


def resolve_sensor_path(run_dir: Path, run_id: str) -> Path | None:
    """Resolve sensor CSV path - supports sensor.csv or run_id.csv."""
    for name in ["sensor.csv", f"{run_id}.csv"]:
        p = run_dir / name
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_sensor_csv(sensor_path: Path) -> pd.DataFrame:
    # Keep it forgiving; data may vary
    df = pd.read_csv(sensor_path)
    # Try to create elapsed time if Date/Time exist
    if "Date" in df.columns and "Time" in df.columns:
        try:
            ts = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
            df["_ts"] = ts
            t0 = ts.dropna().iloc[0] if ts.notna().any() else None
            if t0 is not None:
                df["t_sec"] = (ts - t0).dt.total_seconds()
        except Exception:
            pass
    if "t_sec" not in df.columns:
        df["t_sec"] = np.arange(len(df), dtype=float)
    return df


def find_active_window(df: pd.DataFrame, current_col: str = "Primary Weld Current", thr: float = 5.0, pad_s: float = 0.75):
    if current_col not in df.columns:
        return 0.0, float(df["t_sec"].max())

    t = df["t_sec"].to_numpy()
    x = pd.to_numeric(df[current_col], errors="coerce").fillna(0).to_numpy()
    active = x > thr

    if not active.any():
        return 0.0, float(t.max())

    # Largest contiguous active segment
    idx = np.where(active)[0]
    # find breaks
    breaks = np.where(np.diff(idx) > 1)[0]
    segments = []
    start = idx[0]
    for b in breaks:
        end = idx[b]
        segments.append((start, end))
        start = idx[b + 1]
    segments.append((start, idx[-1]))

    # choose longest in time
    best = max(segments, key=lambda ab: t[ab[1]] - t[ab[0]])
    t0 = max(0.0, float(t[best[0]] - pad_s))
    t1 = min(float(t.max()), float(t[best[1]] + pad_s))
    return t0, t1


PLOTLY_TEMPLATE = "plotly_dark"

def plot_timeseries_with_cursor(df: pd.DataFrame, xcol: str, ycols: list[str], cursor_t: float | None, title: str):
    fig = go.Figure()
    x = df[xcol]
    for c in ycols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=x, y=pd.to_numeric(df[c], errors="coerce"), mode="lines", name=c))
    if cursor_t is not None:
        fig.add_vline(x=cursor_t, line_width=2, line_dash="dash")
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
    )
    st.plotly_chart(fig, use_container_width=True)


def safe_media_exists(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def get_audio_duration_sec(audio_path: Path) -> float | None:
    try:
        import soundfile as sf
        if audio_path.exists() and audio_path.stat().st_size > 0:
            info = sf.info(str(audio_path))
            return info.duration
    except Exception:
        pass
    return None


def ensure_web_playable_video(video_path: Path, cache_dir: Path | None = None) -> tuple[Path, bool]:
    """
    Return (path, converted) to a browser-playable video (MP4). Converts AVI to MP4 if needed.
    Browsers do not support AVI; they need MP4 (H.264) or WebM.
    """
    if not video_path.exists():
        return video_path, False
    if video_path.suffix.lower() in (".mp4", ".webm", ".ogg"):
        return video_path, False
    # AVI or other format - convert to MP4 for browser playback
    cache_dir = cache_dir or Path("outputs/reports/dashboards/cache")
    mp4_path = cache_dir / "video_cache" / f"{video_path.stem}_playable.mp4"
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    if mp4_path.exists() and mp4_path.stat().st_mtime >= video_path.stat().st_mtime:
        return mp4_path, True
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-c:v", "libx264", "-preset", "fast", "-crf", "23", str(mp4_path)],
            check=True,
            capture_output=True,
        )
        return mp4_path, True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return video_path, False  # Fallback to original (may not play in browser)


def get_video_duration_sec(video_path: Path) -> float | None:
    try:
        import cv2
        if video_path.exists() and video_path.stat().st_size > 0:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            if fps > 0 and frames > 0:
                return frames / fps
    except Exception:
        pass
    return None


def _paths(cfg: dict) -> dict:
    """Resolve dashboard paths from config."""
    p = cfg.get("paths", {})
    d = cfg.get("dashboard", {})
    ds = Path(d.get("dataset", "data/interim"))
    return {
        "tabular": Path(d.get("tabular", "outputs/tabular")),
        "checkpoints": Path(d.get("checkpoints", "outputs/checkpoints")),
        "eval": Path(d.get("eval", "outputs/eval")),
        "predictions": Path(d.get("predictions", "outputs/predictions")),
        "reports": Path(d.get("reports", "outputs/reports")),
        "manifest": Path(p.get("manifest", str(ds / "manifest.csv"))),
        "split_dict": Path(p.get("split_dict", str(ds / "split_dict.json"))),
        "norm_stats": Path(p.get("norm_stats", str(ds / "norm_stats.json"))),
        "dataset_meta": Path(p.get("dataset_meta", str(ds / "dataset_meta.json"))),
        "inventory": Path(p.get("inventory", str(ds / "inventory.csv"))),
        "interim": ds,
        "sensor_stats": Path(d.get("sensor_stats", str(ds / "sensor_stats.csv"))),
    }


# -----------------------------
# Result page renderers (dashboard2 aligned)
# -----------------------------
def _render_dataset_stats_d2(cfg: dict, manifest: pd.DataFrame | None, split_dict: dict, dataset_meta: dict | None, paths: dict) -> None:
    st.header("Dataset Stats")
    if manifest is None:
        st.warning("No manifest.csv found.")
        st.stop()
    st.subheader("Manifest Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Chunks", f"{len(manifest):,}")
    c2.metric("Unique Runs", manifest["run_id"].nunique())
    c3.metric("Splits", manifest["split"].nunique())
    st.subheader("Class Distribution (by chunk)")
    dist = manifest["label_code"].astype(str).str.zfill(2).value_counts().sort_index()
    dist.index = [f"{c} — {_label_name(c)}" for c in dist.index]
    st.bar_chart(dist)
    if _exists(paths["norm_stats"]):
        st.subheader("Normalization Statistics")
        norm = json.loads(paths["norm_stats"].read_text())
        st.json({k: v[:5] if isinstance(v, list) and len(v) > 5 else v for k, v in norm.items()})
        st.caption("Showing first 5 values per array.")
    if dataset_meta:
        st.subheader("Dataset Metadata")
        st.json(dataset_meta)
    st.subheader("Manifest Table (first 500 rows)")
    display = manifest.head(500).copy()
    display["label_name"] = display["label_code"].astype(str).str.zfill(2).map(LABEL_MAP).fillna("unknown")
    st.dataframe(display, use_container_width=True, height=400)


def _render_sensor_stats(cfg: dict, paths: dict) -> None:
    """Render Sensor Stats page from run-level sensor aggregations (mean, std, min, max per run)."""
    sensor_stats_path = paths.get("sensor_stats", paths["interim"] / "sensor_stats.csv")
    if not _exists(sensor_stats_path):
        st.warning("No sensor_stats.csv found.")
        st.info("Set dashboard.sensor_stats in config to your sensor_stats.csv path.")
        st.stop()

    ss = pd.read_csv(sensor_stats_path)
    ss["label_name"] = ss["label_code"].astype(str).str.zfill(2).map(LABEL_MAP).fillna("unknown")

    st.header("Sensor Stats")
    st.caption("Run-level aggregations: mean, std, min, max per sensor. Use this to compare defect types and spot outliers.")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Runs", f"{len(ss):,}")
    c2.metric("Unique Labels", ss["label_code"].nunique())
    c3.metric("Sensors", 6, help="Pressure, CO2 Flow, Feed, Current, Wire, Voltage")
    if "weld_active_duration_sec" in ss.columns:
        c4.metric("Avg Weld Duration", f"{ss['weld_active_duration_sec'].mean():.1f}s")

    st.divider()

    # 1. Weld duration distribution
    if "weld_active_duration_sec" in ss.columns:
        st.subheader("Weld-Active Duration")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_dur = px.histogram(ss, x="weld_active_duration_sec", nbins=50, labels={"weld_active_duration_sec": "Duration (s)"}, color_discrete_sequence=["#f59e0b"])
            fig_dur.update_layout(height=320, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_dur, use_container_width=True)
        with col_b:
            fig_dur_label = px.box(ss, x="label_name", y="weld_active_duration_sec", color="label_name", color_discrete_sequence=px.colors.qualitative.Set2)
            fig_dur_label.update_layout(height=320, xaxis_title="", yaxis_title="Duration (s)", showlegend=False, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_dur_label, use_container_width=True)

    # 2. Sensor means by label (box plots for key sensors)
    mean_cols = [c for c in ss.columns if c.endswith("_mean")]
    std_cols = [c for c in ss.columns if c.endswith("_std")]
    if mean_cols:
        st.subheader("Sensor Means by Defect Type")
        sensor_names = sorted(set(c.replace("_mean", "") for c in mean_cols))
        sel_sensors = st.multiselect("Select sensors", sensor_names, default=sensor_names[:3])
        if sel_sensors:
            cols_to_plot = [f"{s}_mean" for s in sel_sensors if f"{s}_mean" in ss.columns]
            if cols_to_plot:
                melted = ss[["label_name"] + cols_to_plot].melt(id_vars="label_name", var_name="sensor", value_name="mean_value")
                melted["sensor"] = melted["sensor"].str.replace("_mean", "")
                fig_means = px.box(melted, x="label_name", y="mean_value", color="sensor", color_discrete_sequence=px.colors.qualitative.Set2)
                fig_means.update_layout(height=400, yaxis_title="Mean value", xaxis_title="", template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig_means, use_container_width=True)

    # 2b. Sensor std by label (variability by defect type)
    if std_cols:
        st.subheader("Sensor Variability (Std) by Defect Type")
        sensor_names_std = sorted(set(c.replace("_std", "") for c in std_cols))
        sel_std = st.multiselect("Select sensors for std", sensor_names_std, default=sensor_names_std[:3], key="std_sensors")
        if sel_std:
            cols_std = [f"{s}_std" for s in sel_std if f"{s}_std" in ss.columns]
            if cols_std:
                melted_std = ss[["label_name"] + cols_std].melt(id_vars="label_name", var_name="sensor", value_name="std_value")
                melted_std["sensor"] = melted_std["sensor"].str.replace("_std", "")
                fig_std = px.box(melted_std, x="label_name", y="std_value", color="sensor", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_std.update_layout(height=400, yaxis_title="Std dev", xaxis_title="", template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig_std, use_container_width=True)

    # 3. Correlation heatmap of sensor means
    if mean_cols:
        st.subheader("Correlation of Sensor Means (across runs)")
        mean_df = ss[mean_cols].copy()
        mean_df.columns = [c.replace("_mean", "") for c in mean_df.columns]
        corr = mean_df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto", zmin=-1, zmax=1)
        fig_corr.update_layout(height=400, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_corr, use_container_width=True)

    # 4. Summary table (percentiles per sensor mean)
    if mean_cols:
        st.subheader("Summary: Sensor Mean Percentiles (across runs)")
        summary = ss[mean_cols].describe().loc[["min", "25%", "50%", "75%", "max"]]
        summary.columns = [c.replace("_mean", "") for c in summary.columns]
        st.dataframe(summary.round(3), use_container_width=True)

    # 5. Export
    st.subheader("Export")
    csv_bytes = ss.to_csv(index=False).encode("utf-8")
    st.download_button("Download full sensor_stats.csv", data=csv_bytes, file_name="sensor_stats.csv", mime="text/csv", key="dl_sensor_stats")

    # 6. Run table (filterable)
    st.subheader("Run-Level Stats (first 200 rows)")
    display_cols = ["run_id", "label_name"] + ([c for c in ss.columns if c != "run_id" and c != "label_code" and c != "label_name"])[:12]
    display_cols = [c for c in display_cols if c in ss.columns]
    st.dataframe(ss[display_cols].head(200), use_container_width=True, height=400)


def _render_tabular_baseline_d2(cfg: dict, paths: dict) -> None:
    st.header("Tabular Baseline — LightGBM (Step 7)")
    tab_dir = paths["tabular"]
    tab_metrics_path = tab_dir / "val_metrics.json"
    tab_preds_path = tab_dir / "val_predictions.csv"
    feat_imp_path = tab_dir / "feature_importance.csv"
    if not _exists(tab_metrics_path):
        st.info("No tabular results found. Run Step 7 first.")
        st.stop()
    m = json.loads(tab_metrics_path.read_text())
    st.subheader("Validation Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Binary F1", f"{m.get('binary_f1', 0):.4f}")
    c2.metric("Macro F1", f"{m.get('macro_f1', 0):.4f}")
    c3.metric("ECE", f"{m.get('ece', 0):.4f}")
    c4.metric("FinalScore", f"{m.get('final_score', 0):.4f}")
    c5.metric("ROC AUC", f"{m.get('roc_auc', 0):.4f}")
    if _exists(feat_imp_path):
        st.subheader("Feature Importance (Top 15)")
        fi = pd.read_csv(feat_imp_path).sort_values("importance", ascending=False).head(15)
        fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", color_discrete_sequence=["#3498db"])
        fig_fi.update_layout(height=420, yaxis=dict(autorange="reversed"), template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_fi, use_container_width=True)
    if _exists(tab_preds_path):
        st.subheader("Validation Predictions")
        preds = pd.read_csv(tab_preds_path)
        st.dataframe(preds, use_container_width=True, height=350)


def _render_neural_training_d2(cfg: dict, paths: dict) -> None:
    st.header("WeldFusionNet Training (Step 11)")
    ckpt_dir = paths["checkpoints"]
    log_path = ckpt_dir / "training_log.json"
    summary_path = ckpt_dir / "training_summary.json"
    if not _exists(log_path):
        st.info("No training_log.json found. Run Step 11 first.")
        st.stop()
    tlog = json.loads(log_path.read_text())
    tdf = pd.DataFrame(tlog) if isinstance(tlog, list) else pd.DataFrame(tlog)
    if _exists(summary_path):
        summary = json.loads(summary_path.read_text())
        st.subheader("Training Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Best FinalScore", f"{summary.get('best_final_score', 0):.4f}")
        s2.metric("Best Epoch", summary.get("best_epoch", "—"))
        s3.metric("Total Epochs", summary.get("total_epochs", "—"))
        s4.metric("Training Time", f"{summary.get('training_time_sec', 0):.0f}s")
    st.subheader("Training Curves")
    col_loss, col_f1, col_lr = st.columns(3)
    with col_loss:
        fig_loss = go.Figure()
        if "train_loss" in tdf.columns:
            fig_loss.add_trace(go.Scatter(x=tdf["epoch"], y=tdf["train_loss"], mode="lines+markers", name="Train Loss", marker=dict(size=4)))
        if "val_loss" in tdf.columns:
            fig_loss.add_trace(go.Scatter(x=tdf["epoch"], y=tdf["val_loss"], mode="lines+markers", name="Val Loss", marker=dict(size=4)))
        fig_loss.update_layout(title="Loss", height=350, xaxis_title="Epoch", yaxis_title="Loss", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_loss, use_container_width=True)
    with col_f1:
        fig_f1 = go.Figure()
        for col, name in [("binary_f1", "Binary F1"), ("macro_f1", "Macro F1"), ("final_score", "FinalScore")]:
            if col in tdf.columns:
                fig_f1.add_trace(go.Scatter(x=tdf["epoch"], y=tdf[col], mode="lines+markers", name=name, marker=dict(size=4)))
        fig_f1.update_layout(title="F1 Metrics", height=350, xaxis_title="Epoch", yaxis_title="Score", yaxis=dict(range=[0.5, 1.02]), template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_f1, use_container_width=True)
    with col_lr:
        if "lr" in tdf.columns:
            fig_lr = px.line(tdf, x="epoch", y="lr", markers=True, color_discrete_sequence=["#e67e22"])
            fig_lr.update_layout(title="Learning Rate", height=350, xaxis_title="Epoch", yaxis_title="LR", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_lr, use_container_width=True)
    st.subheader("Per-Epoch Metrics Table")
    display_cols = [c for c in tdf.columns if c != "per_class_f1"]
    st.dataframe(tdf[display_cols], use_container_width=True, height=400)


def _render_calibration_d2(cfg: dict, paths: dict) -> None:
    st.header("Temperature Calibration (Step 12)")
    cal_path = paths["checkpoints"] / "calibration_report.json"
    if not _exists(cal_path):
        st.info("No calibration_report.json found. Run Step 12 first.")
        st.stop()
    cal = json.loads(cal_path.read_text())
    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", f"{cal.get('temperature', 0):.4f}")
    c2.metric("ECE Before", f"{cal.get('ece_before', 0):.4f}", delta=f"-{cal.get('ece_improvement', 0):.4f}", delta_color="inverse")
    c3.metric("ECE After", f"{cal.get('ece_after', 0):.4f}")
    st.markdown("")
    st.markdown(f"""
**How it works:** Temperature scaling divides the model's logits by a learned
temperature **T = {cal.get('temperature', 0):.4f}** before softmax, sharpening
the probability distribution and reducing calibration error.

- **{cal.get('n_val_samples', '?'):,}** validation samples used for fitting.
- ECE improved by **{cal.get('ece_improvement', 0):.4f}** ({cal.get('ece_before', 0):.4f} → {cal.get('ece_after', 0):.4f}).
""")
    fig_ece = go.Figure()
    fig_ece.add_trace(go.Bar(
        x=["Before Calibration", "After Calibration"],
        y=[cal.get("ece_before", 0), cal.get("ece_after", 0)],
        text=[f"{cal.get('ece_before', 0):.4f}", f"{cal.get('ece_after', 0):.4f}"],
        textposition="outside",
        marker_color=["#e74c3c", "#27ae60"],
    ))
    fig_ece.update_layout(title="Expected Calibration Error (ECE)", height=350, yaxis_title="ECE", template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_ece, use_container_width=True)


def _render_validation_d2(cfg: dict, paths: dict) -> None:
    st.header("Validation Results (Step 13)")
    eval_dir = paths["eval"]
    val_metrics_path = eval_dir / "val_metrics.json"
    val_preds_path = eval_dir / "val_predictions.csv"
    cm_mc_path = eval_dir / "confusion_matrix_mc.png"
    cm_bin_path = eval_dir / "confusion_matrix_binary.png"
    pcr_path = eval_dir / "per_class_report.csv"
    if not _exists(val_metrics_path):
        st.info("No validation metrics found. Run Step 13 first.")
        st.stop()
    vm = json.loads(val_metrics_path.read_text())
    st.subheader("Metrics")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Binary F1", f"{vm.get('binary_f1', 0):.4f}")
    v2.metric("Macro F1", f"{vm.get('macro_f1', 0):.4f}")
    v3.metric("FinalScore", f"{vm.get('final_score', 0):.4f}")
    v4.metric("ECE", f"{vm.get('ece', 0):.4f}")
    pcf1 = vm.get("per_class_f1", {})
    if pcf1:
        st.subheader("Per-Class F1 (Validation)")
        pcf1_data = [{"Class": k.replace("code_", ""), "Label": _label_name(k.replace("code_", "")), "F1": v} for k, v in pcf1.items()]
        pcf1_df = pd.DataFrame(pcf1_data)
        fig_pcf1 = px.bar(pcf1_df, x="F1", y="Label", orientation="h", color="F1", color_continuous_scale="RdYlGn", range_color=[0.9, 1.0], text="F1")
        fig_pcf1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_pcf1.update_layout(height=350, yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 1.08]), template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_pcf1, use_container_width=True)
    cm_l, cm_r = st.columns(2)
    with cm_l:
        if _exists(cm_mc_path):
            st.subheader("Multi-Class Confusion Matrix")
            st.image(str(cm_mc_path), use_container_width=True)
    with cm_r:
        if _exists(cm_bin_path):
            st.subheader("Binary Confusion Matrix")
            st.image(str(cm_bin_path), use_container_width=True)
    if _exists(pcr_path):
        st.subheader("Per-Class Classification Report")
        st.dataframe(pd.read_csv(pcr_path), use_container_width=True)
    if _exists(val_preds_path):
        st.subheader("Validation Predictions")
        vp = pd.read_csv(val_preds_path)
        vp["correct"] = vp["true_code"].astype(str).str.zfill(2) == vp["pred_code"].astype(str).str.zfill(2)
        st.dataframe(vp, use_container_width=True, height=350)


def _render_holdout_test_d2(cfg: dict, paths: dict) -> None:
    st.header("Holdout Test — Unseen Runs (Step 14)")
    pred_dir = paths["predictions"]
    test_metrics_path = pred_dir / "metrics.json"
    test_preds_path = pred_dir / "predictions.csv"
    test_cm_path = pred_dir / "confusion_matrix.png"
    if not _exists(test_metrics_path):
        st.info("No holdout test results found. Run Step 14 first.")
        st.stop()
    tm = json.loads(test_metrics_path.read_text())
    st.subheader("Holdout Test Metrics")
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    t1.metric("Accuracy", f"{tm.get('accuracy', 0):.4f}")
    t2.metric("Binary F1", f"{tm.get('binary_f1', 0):.4f}")
    t3.metric("Macro F1", f"{tm.get('macro_f1', 0):.4f}")
    t4.metric("FinalScore", f"{tm.get('final_score', 0):.4f}")
    t5.metric("ECE", f"{tm.get('ece', 0):.4f}")
    t6.metric("Temperature", f"{tm.get('temperature', 0):.4f}")
    pcf1 = tm.get("per_class_f1", {})
    if pcf1:
        st.subheader("Per-Class F1 (Holdout Test)")
        pcf1_data = [{"Class": k.replace("code_", ""), "Label": _label_name(k.replace("code_", "")), "F1": v} for k, v in pcf1.items()]
        pcf1_df = pd.DataFrame(pcf1_data)
        bar_colors = ["#27ae60" if r["Class"] == "00" else "#e74c3c" for _, r in pcf1_df.iterrows()]
        fig_pcf1 = px.bar(pcf1_df, x="F1", y="Label", orientation="h", text="F1", color_discrete_sequence=["#e74c3c"])
        fig_pcf1.update_traces(texttemplate="%{text:.4f}", textposition="outside", marker_color=bar_colors)
        fig_pcf1.update_layout(height=350, yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 1.08]), template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_pcf1, use_container_width=True)
    if _exists(test_cm_path):
        st.subheader("Confusion Matrix (Holdout Test)")
        st.image(str(test_cm_path), use_container_width=True)
    fig_dir = paths["reports"] / "figures"
    f1_fig_path = fig_dir / "f1_scores_summary.png"
    if _exists(f1_fig_path):
        st.subheader("F1 Scores Summary")
        st.image(str(f1_fig_path), use_container_width=True)
    if _exists(test_preds_path):
        st.subheader("Predictions")
        tp = pd.read_csv(test_preds_path)
        tp["correct"] = tp["true_code"].astype(str).str.zfill(2) == tp["pred_code"].astype(str).str.zfill(2)
        st.dataframe(tp, use_container_width=True, height=350)


def _render_model_comparison_d2(cfg: dict, paths: dict) -> None:
    st.header("Model Comparison")
    rows = []
    tab_m_path = paths["tabular"] / "val_metrics.json"
    eval_m_path = paths["eval"] / "val_metrics.json"
    test_m_path = paths["predictions"] / "metrics.json"
    if _exists(tab_m_path):
        m = json.loads(tab_m_path.read_text())
        rows.append({"Model": "LightGBM (Tabular)", "Split": "Validation", "Binary F1": m.get("binary_f1"), "Macro F1": m.get("macro_f1"), "FinalScore": m.get("final_score"), "ECE": m.get("ece")})
    if _exists(eval_m_path):
        m = json.loads(eval_m_path.read_text())
        rows.append({"Model": "WeldFusionNet", "Split": "Validation", "Binary F1": m.get("binary_f1"), "Macro F1": m.get("macro_f1"), "FinalScore": m.get("final_score"), "ECE": m.get("ece")})
    if _exists(test_m_path):
        m = json.loads(test_m_path.read_text())
        rows.append({"Model": "WeldFusionNet", "Split": "Holdout Test", "Binary F1": m.get("binary_f1"), "Macro F1": m.get("macro_f1"), "FinalScore": m.get("final_score"), "ECE": m.get("ece")})
    if not rows:
        st.info("Run Steps 7, 13, and 14 to compare models.")
        st.stop()
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.subheader("Visual Comparison")
    fig_comp = go.Figure()
    for _, row in comp_df.iterrows():
        label = f"{row['Model']} ({row['Split']})"
        vals = [row["Binary F1"], row["Macro F1"], row["FinalScore"]]
        fig_comp.add_trace(go.Bar(name=label, x=["Binary F1", "Macro F1", "FinalScore"], y=vals, text=[f"{v:.4f}" for v in vals], textposition="outside"))
    fig_comp.update_layout(barmode="group", height=400, yaxis=dict(range=[0.95, 1.01]), legend=dict(orientation="h", yanchor="bottom", y=1.02), template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_comp, use_container_width=True)


def _render_3d_explorer(cfg: dict, paths: dict) -> None:
    """3D Explorer using sensor_stats with meaningful axes: welding regime or outlier detection."""
    sensor_stats_path = paths.get("sensor_stats", paths["interim"] / "sensor_stats.csv")
    if not _exists(sensor_stats_path):
        st.warning("No sensor_stats.csv found.")
        st.info("Set dashboard.sensor_stats in config. The 3D Explorer uses run-level sensor aggregations.")
        st.stop()

    ss = pd.read_csv(sensor_stats_path)
    ss["label_name"] = ss["label_code"].astype(str).str.zfill(2).map(LABEL_MAP).fillna("unknown")

    st.header("3D Explorer")
    st.caption("Explore runs in 3D using meaningful axes. Rotate to find clusters and outliers.")

    mean_cols = [c for c in ss.columns if c.endswith("_mean")]
    std_cols = [c for c in ss.columns if c.endswith("_std")]
    sensor_names = sorted(set(c.replace("_mean", "") for c in mean_cols))

    view_mode = st.radio(
        "View mode",
        ["Welding regime (Current, Voltage, Duration)", "Outlier detection (Mean, Std, Duration)"],
        horizontal=True,
    )

    if view_mode == "Welding regime (Current, Voltage, Duration)":
        x_col = "Primary Weld Current_mean" if "Primary Weld Current_mean" in ss.columns else (mean_cols[0] if mean_cols else None)
        y_col = "Secondary Weld Voltage_mean" if "Secondary Weld Voltage_mean" in ss.columns else (mean_cols[1] if len(mean_cols) > 1 else None)
        z_col = "weld_active_duration_sec" if "weld_active_duration_sec" in ss.columns else None
        x_title = "Mean Current (A)"
        y_title = "Mean Voltage (V)"
        z_title = "Weld Duration (s)"
    else:
        if not sensor_names:
            st.error("No sensor mean columns found in sensor_stats.")
            st.stop()
        default_idx = min(2, len(sensor_names) - 1)
        outlier_sensor = st.selectbox("Sensor for outlier view", sensor_names, index=default_idx)
        x_col = f"{outlier_sensor}_mean"
        y_col = f"{outlier_sensor}_std"
        z_col = "weld_active_duration_sec" if "weld_active_duration_sec" in ss.columns else None
        x_title = f"{outlier_sensor} mean"
        y_title = f"{outlier_sensor} std (variability)"
        z_title = "Weld Duration (s)"

    if not all(c in ss.columns for c in [x_col, y_col, z_col] if c):
        st.error(f"Missing columns for {view_mode}. Need: {x_col}, {y_col}, {z_col}")
        st.stop()

    labels = ss["label_name"].astype(str).unique()
    colors = {lb: LABEL_COLORS.get(lb, f"hsl({i * 137 % 360}, 70%, 50%)") for i, lb in enumerate(labels)}

    fig = go.Figure()
    for lb in labels:
        sub = ss[ss["label_name"] == lb]
        fig.add_trace(go.Scatter3d(
            x=sub[x_col],
            y=sub[y_col],
            z=sub[z_col],
            mode="markers",
            name=lb,
            marker=dict(size=5, color=colors[lb]),
            text=sub["run_id"] if "run_id" in sub.columns else sub.index.astype(str),
            hovertemplate="<b>%{text}</b><br>" + f"{x_title}: %{{x:.2f}}<br>{y_title}: %{{y:.2f}}<br>{z_title}: %{{z:.2f}}<extra></extra>",
        ))

    if view_mode == "Outlier detection (Mean, Std, Duration)":
        q99 = ss[y_col].quantile(0.99)
        n_out = (ss[y_col] >= q99).sum()
        st.caption(f"Points with high std (extending along the variability axis) are outliers. {n_out} runs ≥ 99th percentile std.")

    fig.update_layout(
        height=600,
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_live_sync(cfg: dict) -> None:
    st.title("Live Sync")
    st.caption("Video playback synced with sensor values. Press play — values update in real time.")

    sample_path = Path(cfg.get("dashboard", {}).get("sample_data", "/Users/kiana/Desktop/08-17-22-0011-00"))
    if not sample_path.exists():
        st.error(f"Sample data folder not found: {sample_path}")
        st.info("Set dashboard.sample_data in config to your run folder (e.g. /path/to/08-17-22-0011-00)")
        return

    # Find video and CSV (support run_id.avi/csv or weld.avi/sensor.csv)
    run_id = sample_path.name
    video_path = sample_path / f"{run_id}.avi"
    if not video_path.exists():
        video_path = sample_path / "weld.avi"
    sensor_path = resolve_sensor_path(sample_path, run_id)
    if not sensor_path:
        sensor_path = sample_path / "sensor.csv"

    if not video_path.exists():
        st.error(f"No video found. Expected {run_id}.avi or weld.avi")
        return
    if not sensor_path.exists():
        st.error(f"No sensor CSV found. Expected {run_id}.csv or sensor.csv")
        return

    df = load_sensor_csv(sensor_path)
    if "t_sec" not in df.columns:
        st.error("Could not parse timestamps from CSV (need Date + Time columns)")
        return

    # Sensor columns to display (exclude metadata)
    sensor_cols = [c for c in ["Pressure", "CO2 Weld Flow", "Feed", "Primary Weld Current", "Wire Consumed", "Secondary Weld Voltage"] if c in df.columns]
    if not sensor_cols:
        sensor_cols = [c for c in df.columns if c not in ["Date", "Time", "_ts", "Part No", "Remarks"]]

    # Build sensor data for JS (list of {t_sec, col1, col2, ...})
    data_rows = []
    for _, row in df.iterrows():
        r = {"t_sec": float(row["t_sec"])}
        for c in sensor_cols:
            v = row.get(c)
            try:
                r[c] = float(v) if pd.notna(v) else 0.0
            except (ValueError, TypeError):
                r[c] = 0.0
        data_rows.append(r)

    # Build Plotly chart HTML (sensor timeline)
    # Custom line colors for the chart (edit these to change graph colors)
    CHART_COLORS = ["#f59e0b", "#22d3ee", "#a78bfa", "#34d399"]  # amber, cyan, violet, emerald
    chart_cols = sensor_cols[:4]
    fig = go.Figure()
    x = df["t_sec"]
    for i, c in enumerate(chart_cols):
        if c in df.columns:
            color = CHART_COLORS[i % len(CHART_COLORS)]
            fig.add_trace(go.Scatter(x=x, y=pd.to_numeric(df[c], errors="coerce"), mode="lines", name=c, line=dict(color=color, width=2)))
    fig.update_layout(
        title="Sensor timeline",
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", font=dict(size=10)),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
        shapes=[dict(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1, line=dict(color="#f59e0b", width=2, dash="dash"))],
    )
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": True, "responsive": True}, div_id="sensor-chart")

    # Video + live sensor display (convert AVI to MP4 for browser playback)
    cache_dir = Path(cfg.get("dashboard", {}).get("cache_dir", "outputs/reports/dashboards/cache"))
    playable_path, was_converted = ensure_web_playable_video(video_path, cache_dir)
    if video_path.suffix.lower() == ".avi" and not was_converted:
        st.warning("AVI format is not supported in browsers. Install ffmpeg and restart, or convert to MP4 manually.")

    video_bytes = playable_path.read_bytes()
    video_size_mb = len(video_bytes) / (1024 * 1024)
    if video_size_mb > 50:
        st.warning(f"Video is {video_size_mb:.1f} MB — too large for embedded live sync. Showing video only.")
        col_vid, col_info = st.columns([2, 1])
        with col_vid:
            st.video(str(playable_path))
        with col_info:
            st.info("Use the sensor timeline below. Live sync requires video < 50 MB.")
        st.subheader("Sensor timeline")
        plot_timeseries_with_cursor(df, "t_sec", sensor_cols[:4], None, "Full sensor trace")
    else:
        video_b64 = base64.b64encode(video_bytes).decode("ascii")
        cols_html = "".join(
            f'<div class="sensor-card"><div class="sensor-label">{c}</div><div id="live-{i}" class="sensor-value">—</div></div>'
            for i, c in enumerate(sensor_cols)
        )
        col_ids = [f"live-{i}" for i in range(len(sensor_cols))]
        col_keys = sensor_cols
        n_cols = min(6, len(sensor_cols))
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* {{ box-sizing: border-box; }}
body {{ font-family: 'Inter', -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 12px; overflow: hidden; height: 100%; letter-spacing: -0.01em; }}
#top {{ display: grid; grid-template-columns: 380px 1fr; gap: 16px; height: 280px; min-height: 0; align-items: center; }}
#video-wrap {{ min-height: 0; display: flex; align-items: center; }}
#video-wrap video {{ max-width: 100%; max-height: 280px; object-fit: contain; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
#values-wrap {{ display: flex; flex-direction: column; justify-content: center; min-height: 0; gap: 12px; }}
#live-time {{ font-size: 0.8rem; font-weight: 500; color: #fbbf24; margin-bottom: 2px; letter-spacing: 0.02em; }}
#live-sensor-grid {{ display: grid; grid-template-columns: repeat({n_cols}, 1fr); gap: 10px; }}
.sensor-card {{ background: linear-gradient(145deg,#1e293b,#334155); padding: 0.75rem 0.9rem; border-radius: 10px; text-align: left; min-width: 0; border: 1px solid rgba(245,158,11,0.25); box-shadow: 0 2px 10px rgba(0,0,0,0.2); transition: box-shadow 0.2s; }}
.sensor-card:hover {{ box-shadow: 0 4px 16px rgba(245,158,11,0.12); }}
.sensor-label {{ font-size: 0.72rem; font-weight: 500; color: #94a3b8; line-height: 1.25; word-break: break-word; min-height: 2.2em; letter-spacing: 0.02em; }}
.sensor-value {{ font-size: 1.25rem; font-weight: 700; margin-top: 4px; color: #f1f5f9; letter-spacing: -0.02em; }}
#chart-wrap {{ height: 280px; min-height: 0; margin-top: 12px; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.2); }}
#chart-wrap > div {{ height: 100% !important; }}
</style>
</head>
<body>
<div id="top">
  <div id="video-wrap">
    <video id="live-vid" src="data:video/mp4;base64,{video_b64}" controls></video>
  </div>
  <div id="values-wrap">
    <div id="live-time">Time: <span id="live-time-val">0.00</span> s</div>
    <div id="live-sensor-grid">{cols_html}</div>
  </div>
</div>
<div id="chart-wrap">{chart_html}</div>
<script>
(function() {{
  const data = {json.dumps(data_rows)};
  const colKeys = {json.dumps(col_keys)};
  const colIds = {json.dumps(col_ids)};
  function interpolate(t) {{
    if (!data.length) return null;
    if (t <= data[0].t_sec) return data[0];
    if (t >= data[data.length-1].t_sec) return data[data.length-1];
    let i = 0;
    while (i < data.length && data[i].t_sec < t) i++;
    const r0 = data[i-1], r1 = data[i];
    const frac = (r1.t_sec - r0.t_sec) ? (t - r0.t_sec) / (r1.t_sec - r0.t_sec) : 0;
    const row = {{}};
    for (const k of Object.keys(r0)) {{
      row[k] = typeof r0[k]==='number' ? r0[k] + frac*(r1[k]-r0[k]) : r0[k];
    }}
    return row;
  }}
  function update(t) {{
    const row = interpolate(t);
    if (!row) return;
    const timeEl = document.getElementById('live-time-val');
    if (timeEl) timeEl.textContent = t.toFixed(2);
    colKeys.forEach((k, i) => {{
      const el = document.getElementById(colIds[i]);
      if (el) el.textContent = (row[k] != null ? Number(row[k]).toFixed(2) : '—');
    }});
    const chartEl = document.getElementById('sensor-chart');
    if (chartEl && typeof Plotly !== 'undefined') {{
      Plotly.relayout('sensor-chart', {{ shapes: [{{ type: 'line', xref: 'x', yref: 'paper', x0: t, x1: t, y0: 0, y1: 1, line: {{ color: '#f59e0b', width: 2, dash: 'dash' }} }}] }});
    }}
  }}
  const vid = document.getElementById('live-vid');
  if (vid) {{
    vid.addEventListener('timeupdate', () => update(vid.currentTime));
    vid.addEventListener('loadedmetadata', () => update(0));
    update(vid.currentTime || 0);
  }}
}})();
</script>
</body>
</html>
"""
        components.html(html_content, height=580, scrolling=False)


# -----------------------------
# App
# -----------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    st.set_page_config(page_title="Weld Dashboard", layout="wide", initial_sidebar_state="expanded")

    # Typography and layout polish
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Base typography */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            letter-spacing: -0.01em;
        }
        h1, h2, h3 {
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }
        h1 { font-size: 1.75rem !important; margin-bottom: 0.5rem !important; }
        h2 { font-size: 1.35rem !important; margin-top: 1.5rem !important; margin-bottom: 0.75rem !important; }
        h3 { font-size: 1.1rem !important; }

        /* Main content area spacing */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1400px !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] {
            padding: 0.75rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] img {
            object-fit: contain !important;
            object-position: center !important;
            max-height: 90px !important;
            width: 100% !important;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] > div {
            display: flex !important;
            justify-content: center !important;
            background: transparent !important;
        }

        /* Option menu - cleaner nav */
        [data-testid="stHorizontalBlock"] .stSelectbox label { font-weight: 500 !important; }

        /* Metric cards - aligned, consistent */
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #1e293b 0%, #334155 100%) !important;
            padding: 1.25rem 1.5rem !important;
            border-radius: 12px !important;
            border: 1px solid rgba(245, 158, 11, 0.2) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25) !important;
            transition: transform 0.2s, box-shadow 0.2s !important;
            text-align: center !important;
        }
        [data-testid="stMetric"] label {
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            color: #94a3b8 !important;
            letter-spacing: 0.02em !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            color: #f1f5f9 !important;
            letter-spacing: -0.02em !important;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(245, 158, 11, 0.15) !important;
        }

        /* DataFrames */
        div[data-testid="stDataFrame"] {
            border-radius: 12px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        }
        div[data-testid="stDataFrame"] thead tr th {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
            color: #0f172a !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 12px 16px !important;
            text-align: left !important;
        }
        div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
            background: rgba(30, 41, 59, 0.5) !important;
        }
        div[data-testid="stDataFrame"] tbody tr:hover {
            background: rgba(245, 158, 11, 0.12) !important;
        }
        div[data-testid="stDataFrame"] td {
            padding: 10px 16px !important;
            font-size: 0.9rem !important;
        }

        /* Plotly charts */
        .js-plotly-plot, .plotly {
            border-radius: 12px !important;
            overflow: hidden !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.2) !important;
        }

        /* Buttons */
        .stDownloadButton button, .stButton button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.2s !important;
        }
        .stDownloadButton button:hover, .stButton button:hover {
            transform: scale(1.02) !important;
        }

        /* Captions - muted, consistent */
        [data-testid="stCaptionContainer"] {
            font-size: 0.85rem !important;
            color: #94a3b8 !important;
            margin-top: 0.25rem !important;
        }

        /* Expanders & alerts */
        .streamlit-expanderHeader { border-radius: 8px !important; font-weight: 500 !important; }
        [data-testid="stExpander"] { border-radius: 10px !important; overflow: hidden !important; }
        [data-testid="stAlert"] { border-radius: 10px !important; }

        /* Divider */
        hr { border-color: rgba(148, 163, 184, 0.2) !important; margin: 1.5rem 0 !important; }

        /* Column layout - consistent gaps */
        [data-testid="column"] { gap: 0.5rem !important; }

        /* Info / warning boxes */
        .stAlert { padding: 1rem 1.25rem !important; margin: 1rem 0 !important; }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        logo_path = Path("assets/therness.png")
        if logo_path.exists():
            # Horizontal layout: logo in centered flex container
            lc1, lc2, lc3 = st.columns([1, 3, 1])
            with lc2:
                st.image(str(logo_path), use_container_width=True)
        #st.markdown("## Weld Dashboard")
        nav = option_menu(
            menu_title=None,
            options=[
                "Overview",
                "Sensor Stats",
                "Dataset Stats",
                "Tabular Baseline",
                "Neural Training",
                "Calibration",
                "Validation",
                "Holdout Test",
                "Model Comparison",
                "3D Explorer",
                "Live Sync",
            ],
            icons=[
                "speedometer2",
                "graph-up",
                "database",
                "table",
                "cpu",
                "thermometer",
                "clipboard-data",
                "trophy",
                "arrow-left-right",
                "globe2",
                "play-circle",
            ],
            default_index=0,
        )
        st.divider()

    paths = _paths(cfg)
    manifest_path = paths["manifest"]
    split_path = paths["split_dict"]
    meta_path = paths["dataset_meta"]
    inv_path = paths["inventory"]

    manifest = pd.read_csv(manifest_path) if _exists(manifest_path) else None
    split_dict = json.loads(split_path.read_text()) if _exists(split_path) else {}
    dataset_meta = load_dataset_meta(meta_path)
    index_df = None
    if Path(cfg["paths"]["index"]).exists():
        index_df = load_index_jsonl(Path(cfg["paths"]["index"]))

    # -------------------------
    # Overview (dashboard2 aligned)
    # -------------------------
    if nav == "Overview":
        st.title("Overview")
        st.caption("Dataset health and run inventory")
        st.divider()

        if manifest is None:
            st.warning("No manifest.csv found. Run the data pipeline first.")
            st.stop()

        n_train = len(split_dict.get("train", []))
        n_val = len(split_dict.get("val", []))
        n_test = len(split_dict.get("test", []))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Chunks", f"{len(manifest):,}")
        c2.metric("Unique Runs", manifest["run_id"].nunique())
        c3.metric("Train Runs", f"{n_train:,}")
        c4.metric("Val Runs", f"{n_val:,}")
        c5.metric("Test Runs", f"{n_test:,}")

        st.markdown("")
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Defect-Type Distribution")
            if _exists(inv_path):
                inv = pd.read_csv(inv_path)
                if "label_name" in inv.columns:
                    dist = inv.groupby("label_name").size().sort_values(ascending=True)
                else:
                    dist = inv.groupby("label").size().sort_values(ascending=True)
                fig = px.bar(x=dist.values, y=dist.index, orientation="h", labels={"x": "Number of Runs", "y": ""}, color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=350, showlegend=False, template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig, use_container_width=True)
            else:
                label_dist = manifest["label_code"].astype(str).str.zfill(2).map(LABEL_MAP).fillna("unknown").value_counts()
                st.bar_chart(label_dist)

        with col_r:
            st.subheader("Good vs Defect")
            if _exists(inv_path):
                inv = pd.read_csv(inv_path)
                lbl_col = "label_code" if "label_code" in inv.columns else "label"
                good = int((inv[lbl_col].astype(str).str.zfill(2) == "00").sum())
                defect = len(inv) - good
            else:
                good = int((manifest["label_code"].astype(str).str.zfill(2) == "00").sum())
                defect = len(manifest) - good
            fig_pie = px.pie(values=[good, defect], names=["Good Weld", "Defect"], color_discrete_sequence=["#27ae60", "#e74c3c"])
            fig_pie.update_layout(height=350, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Split Distribution (Chunks)")
        split_counts = manifest.groupby("split").size().reset_index(name="chunks")
        split_counts["runs"] = split_counts["split"].map({"train": n_train, "val": n_val, "test": n_test})
        st.dataframe(split_counts.set_index("split"), use_container_width=True)

        sensor_stats_path = paths.get("sensor_stats", paths["interim"] / "sensor_stats.csv")
        if _exists(sensor_stats_path):
            ss = pd.read_csv(sensor_stats_path)
            if "weld_active_duration_sec" in ss.columns:
                st.subheader("Weld-Active Duration Distribution")
                fig_dur = px.histogram(ss, x="weld_active_duration_sec", nbins=40, labels={"weld_active_duration_sec": "Duration (s)"}, color_discrete_sequence=["#3498db"])
                fig_dur.update_layout(height=300, template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig_dur, use_container_width=True)

    # -------------------------
    # Sensor Stats (run-level sensor aggregations)
    # -------------------------
    elif nav == "Sensor Stats":
        _render_sensor_stats(cfg, paths)

    # -------------------------
    # Dataset Stats (dashboard2 aligned)
    # -------------------------
    elif nav == "Dataset Stats":
        _render_dataset_stats_d2(cfg, manifest, split_dict, dataset_meta, paths)

    # -------------------------
    # Tabular Baseline (dashboard2 aligned)
    # -------------------------
    elif nav == "Tabular Baseline":
        _render_tabular_baseline_d2(cfg, paths)

    # -------------------------
    # Neural Training (dashboard2 aligned)
    # -------------------------
    elif nav == "Neural Training":
        _render_neural_training_d2(cfg, paths)

    # -------------------------
    # Calibration (dashboard2 aligned)
    # -------------------------
    elif nav == "Calibration":
        _render_calibration_d2(cfg, paths)

    # -------------------------
    # Validation (dashboard2 aligned)
    # -------------------------
    elif nav == "Validation":
        _render_validation_d2(cfg, paths)

    # -------------------------
    # Holdout Test (dashboard2 aligned)
    # -------------------------
    elif nav == "Holdout Test":
        _render_holdout_test_d2(cfg, paths)

    # -------------------------
    # Model Comparison (dashboard2 aligned)
    # -------------------------
    elif nav == "Model Comparison":
        _render_model_comparison_d2(cfg, paths)

    # -------------------------
    # 3D Explorer (sensor_stats-based, meaningful axes + outlier detection)
    # -------------------------
    elif nav == "3D Explorer":
        _render_3d_explorer(cfg, paths)

    elif nav == "Live Sync":
        _render_live_sync(cfg)


if __name__ == "__main__":
    main()