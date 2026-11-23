"""Exercise committed models over HTTP with a private database and server restart."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import sklearn

from signal_generator import SIGNAL_TYPES, generate_signal


ROOT = Path(__file__).resolve().parent.parent


def request_json(base, route, payload=None, expected=200):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    try:
        response = urlopen(Request(base + route, data=body, headers=headers), timeout=20)
    except HTTPError as error:
        response = error
    with response:
        if response.code != expected:
            raise AssertionError(f"{route}: expected {expected}, received {response.code}")
        return json.load(response)


def start_server(database):
    code = (
        "import sys; from app import app; from werkzeug.serving import make_server; "
        "app.config['DATABASE_PATH']=sys.argv[1]; "
        "server=make_server('127.0.0.1',0,app); "
        "print(server.server_port,flush=True); server.serve_forever()"
    )
    process = subprocess.Popen([sys.executable, "-u", "-B", "-c", code, str(database)],
                               cwd=ROOT / "BackEnd", stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True)
    try:
        port = int(process.stdout.readline().strip())
    except (ValueError, OSError):
        process.terminate()
        process.wait(timeout=10)
        raise RuntimeError("Verification server failed to start.") from None
    return process, f"http://127.0.0.1:{port}"


def stop_server(process):
    process.terminate()
    process.wait(timeout=10)
    process.stdout.close()


def verify(output=None):
    model_path = ROOT / "Models" / "classifiers.joblib"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report = {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
              "model_bundle_sha256": model_hash, "checks": [], "classifications": []}
    # Keep all verification history separate from the user's application history.
    with tempfile.TemporaryDirectory(prefix="power-quality-verify-") as temporary:
        database = Path(temporary) / "verification.sqlite3"
        process, base = start_server(database)
        try:
            for route, marker in (("/", b"<html"), ("/style.css", b"{"), ("/script.js", b"function")):
                with urlopen(base + route, timeout=20) as response:
                    assert response.code == 200 and marker in response.read(), route
            assert request_json(base, "/api/history")["history"] == []
            report["checks"].append("HTML, CSS and JavaScript served; private history initially empty")
            for label in SIGNAL_TYPES:
                times, values = generate_signal(label)
                # Round-trip through the documented two-column CSV before the API request.
                stream = io.StringIO()
                np.savetxt(stream, np.column_stack((times, values)), delimiter=",",
                           header="time_seconds,voltage_relative", comments="")
                rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
                payload = {
                    "time_seconds": [float(row["time_seconds"]) for row in rows],
                    "voltage_relative": [float(row["voltage_relative"]) for row in rows],
                    "sample_rate_hz": 6000, "cleaned_data_confirmed": True,
                    "added_noise_snr_db": None,
                }
                for name in ("random_forest", "svm"):
                    payload["model"] = name
                    prediction = request_json(base, "/api/classify", payload)["prediction"]
                    assert prediction == label, (name, label, prediction)
                    report["classifications"].append({"class": label, "model": name, "prediction": prediction})
            report["checks"].append("Five CSV classes classified correctly by both committed models")
            request_json(base, "/api/classify", {**payload, "cleaned_data_confirmed": False}, 422)
            request_json(base, "/api/classify", {**payload, "time_seconds": payload["time_seconds"][:100]}, 422)
            request_json(base, "/api/history?limit=0", expected=400)
            page = request_json(base, "/api/history?limit=5")
            assert len(page["history"]) == 5 and page["has_more"]
            older = request_json(base, f"/api/history?limit=5&before_id={page['next_before_id']}")
            assert len(older["history"]) == 5 and not older["has_more"]
            ids = [record["id"] for record in page["history"] + older["history"]]
            assert len(set(ids)) == 10
            report["checks"].append("Invalid requests rejected without saved records; cursor pages complete without duplicates")
        finally:
            stop_server(process)
        process, base = start_server(database)
        try:
            restored = request_json(base, "/api/history")
            assert [record["id"] for record in restored["history"]] == ids
            report["checks"].append("All ten saved analyses survive a separate server process restart")
        finally:
            stop_server(process)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_hash
    report["checks"].append("Serving model bundle unchanged; verification database removed")
    report["status"] = "passed"
    if output:
        path = Path(output)
        if path.exists():
            raise FileExistsError("Verification output already exists. Choose another --output file.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify(args.output)
