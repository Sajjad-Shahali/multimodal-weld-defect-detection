#!/usr/bin/env python3
"""
optimize_postprocessing.py — Grid-search post-processing hyper-parameters
on the VALIDATION set, then auto-apply the best combo to the 115 test
folders to produce submission_optimized.csv.

Sweep axes
----------
  blind_video      : [True, False]
  threshold        : np.linspace(0.50, 0.95, 46)      — precision knob
  class_11_penalty : np.linspace(0.1, 1.0, 19)        — crater-cracks damper

Metrics (same as competition grader)
------------------------------------
  FinalScore = 0.6 * Binary_F1 + 0.4 * Macro_F1

Strategy
--------
  1. Cache TWO sets of per-chunk softmax probs on val:
       a) full   — video as-is
       b) blind  — video zeroed
  2. Pure-Numpy grid search on the cached probs (≈1 760 combos, <10 s).
  3. Print Top-5 combos.
  4. Re-run generate_final_submission with the winning combo.

Usage
-----
  python optimize_postprocessing.py                         # defaults
  python optimize_postprocessing.py --config config.yaml    # explicit config
"""

import argparse
import json
import os
import subprocess
import sys
import time
from itertools import product

import numpy as np
import torch
from sklearn.metrics import f1_score

# Make sure pipeline/ is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.utils import load_config
from pipeline.step8_dataset_torch import build_dataloaders, CLASSES_WITH_DATA
from pipeline.step9_model import build_model, NUM_CLASSES

# ── Label mapping (same as step11_train) ────────────────────────────
CODE_TO_IDX = {c: i for i, c in enumerate(CLASSES_WITH_DATA)}
IDX_TO_CODE = {i: c for c, i in CODE_TO_IDX.items()}


def remap_labels(labels_original: torch.Tensor) -> torch.Tensor:
    device = labels_original.device
    mapping = torch.full((max(CLASSES_WITH_DATA) + 1,), -1, dtype=torch.long, device=device)
    for code, idx in CODE_TO_IDX.items():
        mapping[code] = idx
    return mapping[labels_original]


# ── Collect per-chunk probs on val ──────────────────────────────────

@torch.no_grad()
def cache_val_probs(model, val_loader, device, temperature, *, blind_video: bool):
    """
    Run one pass over val_loader, return
      probs  : np.ndarray (N, 7) — calibrated softmax probs per chunk
      labels : np.ndarray (N,)   — contiguous class indices (0-6)
      run_ids: list[str]         — run_id per chunk
    """
    model.eval()
    all_probs, all_labels, all_run_ids = [], [], []

    for batch in val_loader:
        audio = batch["audio"].to(device)
        video = batch["video"].to(device) if model.use_video else None

        if blind_video and video is not None:
            video = torch.zeros_like(video)

        sensor = None  # use_sensor=False in our best checkpoint

        logits_mc, _ = model(sensor, audio, video)

        # Temperature scaling
        probs = torch.softmax(logits_mc / temperature, dim=1)

        labels_orig = batch["label"].to(device)
        labels = remap_labels(labels_orig)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_run_ids.extend(batch["run_id"])

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        all_run_ids,
    )


# ── Aggregate chunks → per-run probs ───────────────────────────────

def aggregate_to_runs(probs, labels, run_ids):
    """
    Combine chunk-level probs into run-level predictions.
    Returns dict: run_id → {probs: (7,), label: int}
    """
    from collections import defaultdict
    acc = defaultdict(lambda: {"probs": [], "label": None})
    for i, rid in enumerate(run_ids):
        acc[rid]["probs"].append(probs[i])
        acc[rid]["label"] = labels[i]  # all chunks share the same label
    out = {}
    for rid, v in acc.items():
        out[rid] = {
            "probs": np.mean(v["probs"], axis=0),
            "label": v["label"],
        }
    return out


# ── Evaluate one post-processing combo ──────────────────────────────

def evaluate_combo(run_data, threshold, class_11_penalty):
    """
    Apply hallucination penalty, threshold shifting, consistency enforcer
    and compute FinalScore.

    Parameters
    ----------
    run_data          : dict  run_id → {probs: (7,), label: int}
    threshold         : float decision boundary (mapped to grader's 0.5)
    class_11_penalty  : float multiply crater_cracks prob

    Returns
    -------
    final_score, binary_f1, macro_f1
    """
    CLASS_11_IDX = 6  # crater_cracks
    y_true_bin, y_pred_bin = [], []
    y_true_mc, y_pred_mc = [], []

    for rd in run_data.values():
        probs = rd["probs"].copy()
        label = rd["label"]

        # 1) Hallucination penalty
        probs[CLASS_11_IDX] *= class_11_penalty
        s = probs.sum()
        if s > 0:
            probs /= s

        # 2) p_defect
        p_defect = 1.0 - probs[0]

        # 3) Predicted class
        pred_idx = int(probs.argmax())
        pred_code = IDX_TO_CODE[pred_idx]

        # 4) Threshold shifting (precision booster)
        if p_defect < threshold:
            pred_code = 0  # force good_weld

        # 5) Binary truth & pred
        y_true_bin.append(int(label != 0))
        y_pred_bin.append(int(pred_code != 0))

        # 6) Multi-class truth & pred (use contiguous indices)
        y_true_mc.append(label)
        if pred_code == 0:
            y_pred_mc.append(0)
        else:
            y_pred_mc.append(pred_idx)

    binary_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    macro_f1 = f1_score(
        y_true_mc, y_pred_mc,
        labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0,
    )
    final_score = 0.6 * binary_f1 + 0.4 * macro_f1
    return final_score, binary_f1, macro_f1


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid-search post-processing parameters on validation set")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=os.path.join("output", "checkpoints", "best_model.pt"))
    parser.add_argument("--calibration", default=os.path.join("output", "checkpoints", "calibration_report.json"))
    parser.add_argument("--test-dir", default=os.path.join(
        "test_data-20260228T060326Z-1-001", "test_data"))
    parser.add_argument("--output", default="submission_optimized.csv",
                        help="Final optimized submission output path")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Only run grid search, skip generating submission_optimized.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load temperature ────────────────────────────────────────────
    with open(args.calibration) as f:
        cal = json.load(f)
    temperature = cal["temperature"]
    print(f"[1/5] Temperature from calibration: T={temperature:.5f}")

    # ── Load model ──────────────────────────────────────────────────
    print(f"[2/5] Loading model from {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    use_video = ckpt.get("use_video", True)
    use_sensor = ckpt.get("use_sensor", False)
    model = build_model(cfg, use_video=use_video, use_sensor=use_sensor).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}, use_video={use_video}, use_sensor={use_sensor}")

    # ── Build val_loader ────────────────────────────────────────────
    print("[3/5] Building validation DataLoader …")
    _, val_loader, _, _ = build_dataloaders(
        cfg, load_video=use_video, use_sensor=use_sensor,
    )

    # ── Two inference passes (full + blind) ─────────────────────────
    print("[4/5] Running two inference passes on validation set …")
    t0 = time.time()

    probs_full, labels_full, rids_full = cache_val_probs(
        model, val_loader, device, temperature, blind_video=False,
    )
    run_data_full = aggregate_to_runs(probs_full, labels_full, rids_full)
    print(f"  Full video pass: {len(probs_full)} chunks → {len(run_data_full)} runs ({time.time()-t0:.1f}s)")

    t1 = time.time()
    probs_blind, labels_blind, rids_blind = cache_val_probs(
        model, val_loader, device, temperature, blind_video=True,
    )
    run_data_blind = aggregate_to_runs(probs_blind, labels_blind, rids_blind)
    print(f"  Blind video pass: {len(probs_blind)} chunks → {len(run_data_blind)} runs ({time.time()-t1:.1f}s)")

    # ── Grid search ─────────────────────────────────────────────────
    print("[5/5] Grid search …")
    blind_opts = [True, False]
    thresholds = np.round(np.linspace(0.50, 0.95, 46), 4)
    penalties = np.round(np.linspace(0.1, 1.0, 19), 4)

    total = len(blind_opts) * len(thresholds) * len(penalties)
    print(f"  Sweeping {total} combos  (blind×thresh×penalty = {len(blind_opts)}×{len(thresholds)}×{len(penalties)})")

    results = []
    t2 = time.time()
    for blind in blind_opts:
        rd = run_data_blind if blind else run_data_full
        for thr in thresholds:
            for pen in penalties:
                fs, bf1, mf1 = evaluate_combo(rd, float(thr), float(pen))
                results.append({
                    "blind_video": blind,
                    "threshold": float(thr),
                    "class_11_penalty": float(pen),
                    "final_score": fs,
                    "binary_f1": bf1,
                    "macro_f1": mf1,
                })

    elapsed = time.time() - t2
    print(f"  Grid search done in {elapsed:.1f}s")

    # Sort by final_score descending, then binary_f1 as tiebreaker
    results.sort(key=lambda r: (r["final_score"], r["binary_f1"]), reverse=True)

    # ── Print Top-10 ────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  TOP-10 POST-PROCESSING COMBOS ON VALIDATION SET")
    print("=" * 85)
    print(f"{'#':>3}  {'blind':>5}  {'thresh':>7}  {'c11_pen':>7}  "
          f"{'FinalScore':>11}  {'Binary_F1':>10}  {'Macro_F1':>9}")
    print("-" * 85)

    for i, r in enumerate(results[:10], 1):
        bstr = "YES" if r["blind_video"] else "NO"
        print(f"{i:>3}  {bstr:>5}  {r['threshold']:>7.4f}  {r['class_11_penalty']:>7.2f}  "
              f"{r['final_score']:>11.6f}  {r['binary_f1']:>10.6f}  {r['macro_f1']:>9.6f}")
    print("=" * 85)

    # ── Save full grid to JSON ──────────────────────────────────────
    grid_path = "postprocessing_grid_results.json"
    with open(grid_path, "w") as f:
        json.dump(results[:50], f, indent=2)
    print(f"\nTop-50 results saved to {grid_path}")

    # ── Best combo ──────────────────────────────────────────────────
    best = results[0]
    print(f"\n>>> BEST COMBO:")
    print(f"    blind_video      = {best['blind_video']}")
    print(f"    threshold        = {best['threshold']:.4f}")
    print(f"    class_11_penalty = {best['class_11_penalty']:.2f}")
    print(f"    FinalScore       = {best['final_score']:.6f}")
    print(f"    Binary_F1        = {best['binary_f1']:.6f}")
    print(f"    Macro_F1         = {best['macro_f1']:.6f}")

    if args.skip_generate:
        print("\n--skip-generate: skipping submission_optimized.csv generation.")
        return

    # ── Generate optimized submission ───────────────────────────────
    print(f"\n>>> Generating {args.output} with best combo …")
    cmd = [
        sys.executable, "generate_final_submission.py",
        "--config", args.config,
        "--test-dir", args.test_dir,
        "--output", args.output,
        "--threshold", str(best["threshold"]),
        "--class-11-penalty", str(best["class_11_penalty"]),
    ]
    if best["blind_video"]:
        cmd.append("--blind-video")
    else:
        cmd.append("--no-blind-video")

    print(f"  CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\n✓ Optimized submission written to {args.output}")


if __name__ == "__main__":
    main()
