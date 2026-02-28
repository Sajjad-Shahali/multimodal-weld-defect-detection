from __future__ import annotations
from pathlib import Path
from weldml.utils.config import load_config
from weldml.data.indexer import (
    discover_runs,
    discover_runs_from_manifest,
    enrich_media_meta,
    write_jsonl,
    write_inventory_csv,
)
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--from-manifest", action="store_true", help="Build index from manifest.csv + split_dict.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]

    if args.from_manifest and paths.get("manifest"):
        manifest_path = Path(paths["manifest"])
        split_path = Path(paths.get("split_dict", "data/interim/split_dict.json"))
        data_root = paths.get("data_root")
        recs = discover_runs_from_manifest(manifest_path, split_path, data_root)
        print(f"Loaded {len(recs)} runs from manifest")
    else:
        recs = []
        recs += discover_runs(Path(paths["raw_train_good"]), split="train", label_from_folder=True)
        recs += discover_runs(Path(paths["raw_train_defect"]), split="train", label_from_folder=True)
        recs += discover_runs(Path(paths["raw_test"]), split="test", label_from_folder=False)

    for r in recs:
        enrich_media_meta(r)

    write_jsonl(recs, Path(paths["index"]))
    write_inventory_csv(recs, Path(paths["inventory"]))
    print(f"Wrote index: {paths['index']}")
    print(f"Wrote inventory: {paths['inventory']}")

if __name__ == "__main__":
    main()
