from __future__ import annotations
import subprocess, sys

def main():
    config = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else "configs/default.yaml"
    cmd = ["streamlit", "run", "src/weldml/dashboard/app.py", "--", "--config", config]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
