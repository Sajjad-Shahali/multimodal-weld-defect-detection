from __future__ import annotations
from pathlib import Path
import pandas as pd
from io import StringIO

def extract_features_from_run(run_record: dict) -> dict:
    # Placeholder: implement per-modality extraction later.
    return {
        "run_id": run_record["run_id"],
        "label": run_record.get("label"),
        "t_start_s": None,
        "t_end_s": None,
        "active_duration_s": None,
    }

def extract_features(index_jsonl: Path, out_path: Path) -> None:
    rows = []
    with index_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            run = pd.read_json(StringIO(line), typ="series").to_dict()
            rows.append(extract_features_from_run(run))
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
