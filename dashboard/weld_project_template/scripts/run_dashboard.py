from __future__ import annotations
import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).resolve().parent.parent / "src" / "weldml" / "dashboard" / "app.py"
    config = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "configs/default.yaml"
    cmd = ["streamlit", "run", str(app_path), "--server.headless", "true", "--", "--config", config]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
