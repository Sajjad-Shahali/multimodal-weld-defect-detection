#!/usr/bin/env python
"""
fix_submission.py — Fix the threshold contradiction bug in submission.csv.

Problem
-------
The grader penalises logical inconsistencies between pred_label_code and p_defect:
  • pred_label_code == '00' (good weld) BUT p_defect >= 0.5  → False Positive
  • pred_label_code != '00' (defect)     BUT p_defect <  0.5  → False Negative

This script reads submission.csv, corrects contradictory rows, and writes
submission_fixed.csv for immediate re-submission.

Usage
-----
  python fix_submission.py
  python fix_submission.py --input submission.csv --output submission_fixed.csv
"""

import argparse
import csv


def fix_submission(input_path="submission.csv", output_path="submission_fixed.csv"):
    rows = []
    n_fp_fix = 0   # pred=00 but p_defect >= 0.5  →  scale down to 0.49
    n_fn_fix = 0   # pred!=00 but p_defect <  0.5  →  scale up   to 0.51

    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id       = row["sample_id"]
            pred_label_code = row["pred_label_code"].strip()
            p_defect        = float(row["p_defect"])

            # ── Fix 1: Good weld predicted but p_defect says defect ──
            if pred_label_code == "00" and p_defect >= 0.5:
                p_defect = 0.49
                n_fp_fix += 1

            # ── Fix 2: Defect predicted but p_defect says good ───────
            elif pred_label_code != "00" and p_defect < 0.5:
                p_defect = 0.51
                n_fn_fix += 1

            rows.append({
                "sample_id":       sample_id,
                "pred_label_code": pred_label_code,
                "p_defect":        round(p_defect, 4),
            })

    # ── Write fixed CSV ──────────────────────────────────────────────
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "pred_label_code", "p_defect"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Fixed {n_fp_fix} FP contradictions (pred=00, p_defect>=0.5 → 0.49)")
    print(f"Fixed {n_fn_fix} FN contradictions (pred!=00, p_defect<0.5 → 0.51)")
    print(f"Total rows: {len(rows)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix threshold contradictions in submission CSV")
    parser.add_argument("--input",  default="submission.csv")
    parser.add_argument("--output", default="submission_fixed.csv")
    args = parser.parse_args()
    fix_submission(args.input, args.output)
