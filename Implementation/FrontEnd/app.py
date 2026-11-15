"""Compatibility launcher for the complete analyzer, including its APIs."""

from pathlib import Path
import runpy
import sys

backend_folder = Path(__file__).resolve().parent.parent / "BackEnd"
sys.path.insert(0, str(backend_folder))
app = runpy.run_path(str(backend_folder / "app.py"))["app"]


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
