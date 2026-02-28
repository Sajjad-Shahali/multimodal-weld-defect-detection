from __future__ import annotations
from pathlib import Path
from weldml.utils.config import load_config
from weldml.features.extract import extract_features
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    extract_features(Path(cfg["paths"]["index"]), Path(cfg["paths"]["features"]))
    print(f"Wrote features: {cfg['paths']['features']}")

if __name__ == "__main__":
    main()
