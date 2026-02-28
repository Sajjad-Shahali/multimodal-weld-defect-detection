"""\
step15_sanity_leakage.py — Sanity checks for data leakage.

Why this exists
--------------
If validation results look "too good", the most common root cause is leakage:
the same weld run (or the same raw file) appearing in both train and val/test.

This step audits the Step 6 dataset outputs:
  - output/dataset/manifest.csv
  - output/dataset/split_dict.json (if present)

It checks:
  1) Run-level overlap between splits (hard leakage)
  2) Chunk filename overlap between splits (hard leakage)
  3) AVI path overlap across splits (usually implies run-level overlap)
  4) Manifest "split" column consistency (each run_id should map to one split)
  5) Duplicate chunk file rows in the manifest
    6) Run-level label consistency (each run_id should map to one label_code)
    7) Split/label distribution summary (missing labels, imbalance)
    8) Chunk files exist on disk (output/dataset/chunks)
    9) Exact duplicate chunk content across splits (SHA1 of .npz bytes; sampled)
    10) Near-duplicate chunk signatures across splits (quantized sensor+audio; sampled)
    11) Spot-check manifest ↔ .npz metadata consistency (sampled)

Outputs
-------
Writes:
  output/leakage_report.json

Usage
-----
  python -m pipeline.step15_sanity_leakage --config config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from pipeline.utils import load_config, ensure_dir

log = logging.getLogger(__name__)


@dataclass
class LeakageFinding:
    name: str
    status: str  # "ok" | "warn" | "fail"
    details: Dict[str, Any]


def _set_intersection(a: Set[str], b: Set[str]) -> List[str]:
    return sorted(a.intersection(b))


def _limited(items: List[str], limit: int = 25) -> Dict[str, Any]:
    return {
        "count": len(items),
        "examples": items[:limit],
        "truncated": len(items) > limit,
    }


def _get_split_runs_from_manifest(manifest: pd.DataFrame, split_name: str) -> Set[str]:
    if "split" not in manifest.columns:
        return set()
    if "run_id" not in manifest.columns:
        return set()
    return set(manifest.loc[manifest["split"] == split_name, "run_id"].astype(str).unique())


def _get_split_files_from_manifest(manifest: pd.DataFrame, split_name: str) -> Set[str]:
    if "split" not in manifest.columns or "file" not in manifest.columns:
        return set()
    return set(manifest.loc[manifest["split"] == split_name, "file"].astype(str).unique())


def _get_split_avi_paths_from_manifest(manifest: pd.DataFrame, split_name: str) -> Set[str]:
    if "split" not in manifest.columns or "avi_path" not in manifest.columns:
        return set()
    # Normalize path string casing/separators a bit for Windows.
    paths = (
        manifest.loc[manifest["split"] == split_name, "avi_path"]
        .astype(str)
        .str.replace("\\\\", "/", regex=False)
        .str.lower()
    )
    return set(paths.unique())


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _quantized_signature_npz(path: str, max_elems_per_array: int = 4096) -> str:
    """Lightweight signature for near-duplicate detection.

    Not cryptographically unique and can collide in theory, so we treat cross-split
    matches as a warning signal rather than an automatic "fail".
    """

    z = np.load(path, allow_pickle=False)
    sensor = np.asarray(z["sensor"], dtype=np.float32)
    audio = np.asarray(z["audio"], dtype=np.float32)

    def _downsample_flat(a: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(a, copy=False)
        flat = a.reshape(-1)
        if flat.size <= max_elems_per_array:
            # Clip before float16 cast to avoid overflow.
            return np.clip(flat, -65504.0, 65504.0)
        stride = max(1, int(flat.size // max_elems_per_array))
        sample = flat[::stride][:max_elems_per_array]
        return np.clip(sample, -65504.0, 65504.0)

    s = _downsample_flat(sensor).astype(np.float16, copy=False)
    a = _downsample_flat(audio).astype(np.float16, copy=False)

    h = hashlib.sha1()
    h.update(str(sensor.shape).encode("utf-8"))
    h.update(str(audio.shape).encode("utf-8"))
    h.update(s.tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def _sample_list(items: List[str], max_items: int, seed: int) -> List[str]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    rng = random.Random(seed)
    return rng.sample(items, k=max_items)


def run(config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    out_root = cfg.get("output_root", "output")

    sanity_cfg = cfg.get("sanity_checks", {}) or {}
    seed = int(sanity_cfg.get("random_seed", 0))
    chunk_hash_max_files = int(sanity_cfg.get("chunk_hash_max_files", 5000))
    chunk_signature_max_files = int(sanity_cfg.get("chunk_signature_max_files", 2000))
    manifest_npz_spotcheck_files = int(sanity_cfg.get("manifest_npz_spotcheck_files", 250))

    ds_dir = os.path.join(out_root, "dataset")
    manifest_path = os.path.join(ds_dir, "manifest.csv")
    split_dict_path = os.path.join(ds_dir, "split_dict.json")
    report_path = os.path.join(out_root, "leakage_report.json")

    chunks_dir = os.path.join(ds_dir, "chunks")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path} (run step6 first)")

    ensure_dir(out_root)

    manifest = pd.read_csv(manifest_path)
    findings: List[LeakageFinding] = []

    # ---- Basic columns presence ----
    required_cols = ["file", "run_id", "chunk_idx", "label_code", "split"]
    missing_cols = [c for c in required_cols if c not in manifest.columns]
    findings.append(
        LeakageFinding(
            name="manifest_required_columns_present",
            status="fail" if len(missing_cols) > 0 else "ok",
            details={"missing_columns": missing_cols, "required_columns": required_cols},
        )
    )

    # ---- Basic manifest integrity ----
    dup_rows = int(manifest.duplicated().sum())
    findings.append(
        LeakageFinding(
            name="manifest_duplicate_rows",
            status="warn" if dup_rows > 0 else "ok",
            details={"duplicate_row_count": dup_rows, "n_rows": int(len(manifest))},
        )
    )

    if "file" in manifest.columns:
        dup_files = manifest["file"].astype(str).duplicated().sum()
        findings.append(
            LeakageFinding(
                name="manifest_duplicate_chunk_files",
                status="fail" if dup_files > 0 else "ok",
                details={"duplicate_file_count": int(dup_files)},
            )
        )

    if "run_id" in manifest.columns and "label_code" in manifest.columns:
        run_label_nunique = manifest.groupby("run_id")["label_code"].nunique(dropna=False)
        multi_label_runs = run_label_nunique[run_label_nunique > 1]
        findings.append(
            LeakageFinding(
                name="run_id_has_multiple_labels",
                status="fail" if len(multi_label_runs) > 0 else "ok",
                details=_limited([str(i) for i in multi_label_runs.index.tolist()]),
            )
        )

    # ---- Split availability ----
    if "split" not in manifest.columns:
        findings.append(
            LeakageFinding(
                name="manifest_has_split_column",
                status="fail",
                details={"reason": "manifest.csv has no 'split' column"},
            )
        )
        report = {
            "config_path": config_path,
            "manifest_path": manifest_path,
            "split_dict_path": split_dict_path,
            "findings": [asdict(f) for f in findings],
            "overall": "fail",
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  ❌ Leakage report written: {report_path}")
        return report

    split_counts = manifest["split"].value_counts(dropna=False).to_dict()
    findings.append(
        LeakageFinding(
            name="manifest_split_counts",
            status="ok",
            details={"split_counts": {str(k): int(v) for k, v in split_counts.items()}},
        )
    )

    # ---- Split/label distribution ----
    if "label_code" in manifest.columns:
        dist = (
            manifest.groupby(["split", "label_code"]).size().unstack(fill_value=0).sort_index()
        )
        label_codes = [str(c) for c in dist.columns.tolist()]
        split_names = [str(i) for i in dist.index.tolist()]
        split_label_counts = {
            str(split): {str(lbl): int(dist.loc[split, lbl]) for lbl in dist.columns}
            for split in dist.index
        }
        # Warn if any split is missing a label entirely.
        missing_labels = []
        for split in dist.index:
            for lbl in dist.columns:
                if int(dist.loc[split, lbl]) == 0:
                    missing_labels.append(f"split={split} label_code={lbl}")
        findings.append(
            LeakageFinding(
                name="split_label_distribution",
                status="warn" if len(missing_labels) > 0 else "ok",
                details={
                    "splits": split_names,
                    "label_codes": label_codes,
                    "counts": split_label_counts,
                    "missing_labels": _limited(missing_labels),
                },
            )
        )

    # ---- Chunk files exist on disk ----
    if "file" in manifest.columns:
        files_unique = manifest["file"].astype(str).unique().tolist()
        missing = []
        for f in files_unique:
            p = os.path.join(chunks_dir, f)
            if not os.path.exists(p):
                missing.append(f)
        findings.append(
            LeakageFinding(
                name="chunk_files_exist_on_disk",
                status="fail" if len(missing) > 0 else "ok",
                details={
                    "chunks_dir": chunks_dir,
                    "missing": _limited(sorted(missing)),
                    "n_unique_files": int(len(files_unique)),
                },
            )
        )

    # ---- One run_id must map to one split ----
    run_split_nunique = (
        manifest.groupby("run_id")["split"].nunique(dropna=False)
        if "run_id" in manifest.columns
        else pd.Series(dtype=int)
    )
    multi_split_runs = run_split_nunique[run_split_nunique > 1]
    findings.append(
        LeakageFinding(
            name="run_id_in_multiple_splits",
            status="fail" if len(multi_split_runs) > 0 else "ok",
            details=_limited([str(i) for i in multi_split_runs.index.tolist()]),
        )
    )

    # ---- Load split_dict if available (run-level) ----
    split_dict: Optional[Dict[str, List[str]]] = None
    if os.path.exists(split_dict_path):
        with open(split_dict_path, "r", encoding="utf-8") as f:
            split_dict = json.load(f)
    findings.append(
        LeakageFinding(
            name="split_dict_present",
            status="ok" if split_dict is not None else "warn",
            details={"path": split_dict_path, "present": split_dict is not None},
        )
    )

    # Determine which splits to compare.
    splits = sorted(set(manifest["split"].astype(str).unique().tolist()))
    # Common expected values are train/val/test.
    pairs = []
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            pairs.append((splits[i], splits[j]))

    # ---- Leakage checks per split-pair ----
    for a, b in pairs:
        a_runs = _get_split_runs_from_manifest(manifest, a)
        b_runs = _get_split_runs_from_manifest(manifest, b)
        run_overlap = _set_intersection(a_runs, b_runs)
        findings.append(
            LeakageFinding(
                name=f"run_id_overlap__{a}__{b}",
                status="fail" if len(run_overlap) > 0 else "ok",
                details=_limited(run_overlap),
            )
        )

        a_files = _get_split_files_from_manifest(manifest, a)
        b_files = _get_split_files_from_manifest(manifest, b)
        file_overlap = _set_intersection(a_files, b_files)
        findings.append(
            LeakageFinding(
                name=f"chunk_file_overlap__{a}__{b}",
                status="fail" if len(file_overlap) > 0 else "ok",
                details=_limited(file_overlap),
            )
        )

        a_avi = _get_split_avi_paths_from_manifest(manifest, a)
        b_avi = _get_split_avi_paths_from_manifest(manifest, b)
        avi_overlap = _set_intersection(a_avi, b_avi)
        # If there is avi overlap, it's highly suspicious. Mark warn (fail if also run overlap).
        status = "ok"
        if len(avi_overlap) > 0:
            status = "fail" if len(run_overlap) > 0 else "warn"
        findings.append(
            LeakageFinding(
                name=f"avi_path_overlap__{a}__{b}",
                status=status,
                details=_limited(avi_overlap),
            )
        )

    # ---- Exact duplicate chunk-content hashes (sampled) ----
    # This detects cases where content is duplicated under different filenames.
    if "file" in manifest.columns and "split" in manifest.columns and os.path.isdir(chunks_dir):
        file_to_split = (
            manifest[["file", "split"]]
            .dropna()
            .astype({"file": str, "split": str})
            .drop_duplicates(subset=["file"])
            .set_index("file")["split"]
            .to_dict()
        )
        all_files = sorted(file_to_split.keys())
        sampled = _sample_list(all_files, chunk_hash_max_files, seed)

        hash_to_splits: Dict[str, Set[str]] = {}
        hash_to_examples: Dict[str, List[str]] = {}

        for f in sampled:
            p = os.path.join(chunks_dir, f)
            if not os.path.exists(p):
                continue
            try:
                hx = _sha1_file(p)
            except Exception as e:
                findings.append(
                    LeakageFinding(
                        name="chunk_content_hashing_error",
                        status="warn",
                        details={"file": f, "error": f"{type(e).__name__}: {str(e).splitlines()[0]}"},
                    )
                )
                continue

            split = str(file_to_split.get(f, ""))
            hash_to_splits.setdefault(hx, set()).add(split)
            if hx not in hash_to_examples:
                hash_to_examples[hx] = []
            if len(hash_to_examples[hx]) < 5:
                hash_to_examples[hx].append(f)

        cross_split_hashes = []
        within_split_hashes = []
        for hx, splits in hash_to_splits.items():
            if len(hash_to_examples.get(hx, [])) <= 1:
                continue
            if len([s for s in splits if s]) > 1:
                cross_split_hashes.append({"sha1": hx, "splits": sorted(splits), "examples": hash_to_examples[hx]})
            else:
                within_split_hashes.append({"sha1": hx, "split": sorted(splits)[0] if splits else "", "examples": hash_to_examples[hx]})

        findings.append(
            LeakageFinding(
                name="exact_duplicate_chunk_content_across_splits",
                status="fail" if len(cross_split_hashes) > 0 else "ok",
                details={
                    "chunks_dir": chunks_dir,
                    "sampled_files": int(len(sampled)),
                    "max_files": int(chunk_hash_max_files),
                    "cross_split_duplicates": {"count": len(cross_split_hashes), "examples": cross_split_hashes[:25]},
                },
            )
        )
        findings.append(
            LeakageFinding(
                name="exact_duplicate_chunk_content_within_split",
                status="warn" if len(within_split_hashes) > 0 else "ok",
                details={
                    "chunks_dir": chunks_dir,
                    "sampled_files": int(len(sampled)),
                    "max_files": int(chunk_hash_max_files),
                    "within_split_duplicates": {"count": len(within_split_hashes), "examples": within_split_hashes[:25]},
                },
            )
        )

    # ---- Near-duplicate chunk signatures (sampled; sensor+audio quantized) ----
    if "file" in manifest.columns and "split" in manifest.columns and os.path.isdir(chunks_dir):
        file_to_split = (
            manifest[["file", "split"]]
            .dropna()
            .astype({"file": str, "split": str})
            .drop_duplicates(subset=["file"])
            .set_index("file")["split"]
            .to_dict()
        )
        all_files = sorted(file_to_split.keys())
        sampled = _sample_list(all_files, chunk_signature_max_files, seed + 1)

        sig_to_splits: Dict[str, Set[str]] = {}
        sig_to_examples: Dict[str, List[str]] = {}
        for f in sampled:
            p = os.path.join(chunks_dir, f)
            if not os.path.exists(p):
                continue
            try:
                sx = _quantized_signature_npz(p)
            except Exception as e:
                findings.append(
                    LeakageFinding(
                        name="chunk_signature_error",
                        status="warn",
                        details={"file": f, "error": f"{type(e).__name__}: {str(e).splitlines()[0]}"},
                    )
                )
                continue
            split = str(file_to_split.get(f, ""))
            sig_to_splits.setdefault(sx, set()).add(split)
            if sx not in sig_to_examples:
                sig_to_examples[sx] = []
            if len(sig_to_examples[sx]) < 5:
                sig_to_examples[sx].append(f)

        cross_split_sigs = []
        for sx, splits in sig_to_splits.items():
            if len(sig_to_examples.get(sx, [])) <= 1:
                continue
            if len([s for s in splits if s]) > 1:
                cross_split_sigs.append({"signature": sx, "splits": sorted(splits), "examples": sig_to_examples[sx]})

        findings.append(
            LeakageFinding(
                name="near_duplicate_chunk_signatures_across_splits",
                status="warn" if len(cross_split_sigs) > 0 else "ok",
                details={
                    "chunks_dir": chunks_dir,
                    "sampled_files": int(len(sampled)),
                    "max_files": int(chunk_signature_max_files),
                    "cross_split_matches": {"count": len(cross_split_sigs), "examples": cross_split_sigs[:25]},
                },
            )
        )

    # ---- Spot-check manifest ↔ npz metadata consistency (sampled) ----
    if (
        "file" in manifest.columns
        and "run_id" in manifest.columns
        and "chunk_idx" in manifest.columns
        and "label_code" in manifest.columns
        and os.path.isdir(chunks_dir)
    ):
        df_small = manifest[["file", "run_id", "chunk_idx", "label_code"]].dropna().copy()
        df_small["file"] = df_small["file"].astype(str)
        df_small["run_id"] = df_small["run_id"].astype(str)
        df_small["chunk_idx"] = df_small["chunk_idx"].astype(int)
        df_small["label_code"] = df_small["label_code"].astype(int)
        rows = list(df_small.itertuples(index=False, name=None))
        sampled_rows = _sample_list(rows, manifest_npz_spotcheck_files, seed + 2)

        mismatches = []
        checked = 0
        for f, run_id, chunk_idx, label_code in sampled_rows:
            p = os.path.join(chunks_dir, f)
            if not os.path.exists(p):
                continue
            try:
                z = np.load(p, allow_pickle=False)
                npz_run = str(z["run_id"].item())
                npz_idx = int(z["chunk_idx"].item())
                npz_lbl = int(z["label"].item())
            except Exception as e:
                mismatches.append(
                    f"file={f} error={type(e).__name__}:{str(e).splitlines()[0]}"
                )
                continue
            checked += 1
            if npz_run != str(run_id) or npz_idx != int(chunk_idx) or npz_lbl != int(label_code):
                mismatches.append(
                    f"file={f} manifest(run_id={run_id},chunk_idx={chunk_idx},label_code={label_code}) npz(run_id={npz_run},chunk_idx={npz_idx},label={npz_lbl})"
                )

        findings.append(
            LeakageFinding(
                name="manifest_npz_metadata_consistency_spotcheck",
                status="fail" if len(mismatches) > 0 else "ok",
                details={
                    "checked": int(checked),
                    "requested": int(manifest_npz_spotcheck_files),
                    "mismatches": _limited(mismatches),
                },
            )
        )

    # ---- Compare manifest runs vs split_dict runs (if provided) ----
    if split_dict is not None and "run_id" in manifest.columns:
        for split_name, ids in split_dict.items():
            ids_set = set(map(str, ids))
            mf_set = _get_split_runs_from_manifest(manifest, split_name)
            missing_in_manifest = sorted(ids_set - mf_set)
            missing_in_split_dict = sorted(mf_set - ids_set)

            status = "ok" if (len(missing_in_manifest) == 0 and len(missing_in_split_dict) == 0) else "warn"
            findings.append(
                LeakageFinding(
                    name=f"split_dict_consistency__{split_name}",
                    status=status,
                    details={
                        "missing_in_manifest": _limited(missing_in_manifest),
                        "missing_in_split_dict": _limited(missing_in_split_dict),
                    },
                )
            )

        # Hard check: run overlap in split_dict itself
        keys = list(split_dict.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                overlap = _set_intersection(set(map(str, split_dict[a])), set(map(str, split_dict[b])))
                findings.append(
                    LeakageFinding(
                        name=f"split_dict_run_id_overlap__{a}__{b}",
                        status="fail" if len(overlap) > 0 else "ok",
                        details=_limited(overlap),
                    )
                )

    overall = "ok"
    if any(f.status == "fail" for f in findings):
        overall = "fail"
    elif any(f.status == "warn" for f in findings):
        overall = "warn"

    report: Dict[str, Any] = {
        "config_path": config_path,
        "manifest_path": manifest_path,
        "split_dict_path": split_dict_path,
        "overall": overall,
        "findings": [asdict(f) for f in findings],
        "sanity_config": {
            "random_seed": seed,
            "chunk_hash_max_files": chunk_hash_max_files,
            "chunk_signature_max_files": chunk_signature_max_files,
            "manifest_npz_spotcheck_files": manifest_npz_spotcheck_files,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if overall == "ok":
        print(f"  ✅ Leakage checks passed. Report: {report_path}")
    elif overall == "warn":
        print(f"  ⚠ Leakage checks completed with warnings. Report: {report_path}")
    else:
        print(f"  ❌ Leakage checks FAILED. Report: {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 15: Leakage sanity checks")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args.config)
