"""Serve the web interface and classify validated voltage recordings."""

from functools import lru_cache
from pathlib import Path
import sqlite3
from time import perf_counter

from flask import Flask, jsonify, request
import joblib
import numpy as np
import sklearn
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge, UnsupportedMediaType

from feature_extraction import (
    FEATURE_NAMES, FEATURE_VERSION, SAMPLE_COUNT, SAMPLE_RATE_HZ, extract_features,
)
from signal_generator import SIGNAL_TYPES
from history import read_history, save_analysis


frontend_folder = Path(__file__).resolve().parent.parent / "FrontEnd"

app = Flask(
    __name__,
    static_folder=str(frontend_folder),
    static_url_path="",
)
app.config.update(
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
    MODEL_BUNDLE=Path(__file__).resolve().parent.parent / "Models" / "classifiers.joblib",
    DATABASE_PATH=Path(__file__).resolve().parent.parent / "Data" / "analysis_history.sqlite3",
)


def validate_recording(payload):
    if not isinstance(payload, dict):
        raise ValueError("Send a JSON object containing the recording.")
    if payload.get("cleaned_data_confirmed") is not True:
        raise ValueError("Confirm that the recording contains cleaned data.")
    model = payload.get("model")
    if model not in ("random_forest", "svm"):
        raise ValueError("Select Random Forest or Support Vector Machine.")
    arrays = []
    for key in ("time_seconds", "voltage_relative"):
        samples = payload.get(key)
        if not isinstance(samples, list) or len(samples) != SAMPLE_COUNT:
            raise ValueError("This model requires 1,200 samples covering 0.2 seconds at 6,000 Hz.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in samples):
            raise ValueError("Timestamps and voltage samples must be numbers.")
        try:
            values = np.asarray(samples, dtype=np.float64)
        except (ValueError, OverflowError):
            raise ValueError("Timestamps and voltage samples must be finite numbers.") from None
        if not np.isfinite(values).all():
            raise ValueError("Timestamps and voltage samples must be finite numbers.")
        arrays.append(values)
    times, voltage = arrays
    steps = np.diff(times)
    if not np.allclose(steps, 1 / SAMPLE_RATE_HZ, rtol=0.001, atol=1e-10):
        raise ValueError("Use increasing, evenly spaced timestamps sampled at 6,000 Hz.")
    rate = payload.get("sample_rate_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError("The declared sample rate must match the 6,000 Hz recording.")
    try:
        rate = float(rate)
    except OverflowError:
        raise ValueError("The declared sample rate must match the 6,000 Hz recording.") from None
    if not np.isfinite(rate) or not np.isclose(rate, SAMPLE_RATE_HZ, rtol=0.001):
        raise ValueError("The declared sample rate must match the 6,000 Hz recording.")
    if np.max(np.abs(voltage)) > 5:
        raise ValueError("Use relative voltage with nominal peak 1.0. Magnitudes above 5 are unsupported.")
    if np.ptp(voltage) == 0:
        raise ValueError("A flat recording is outside the trained examples. Include the waveform before and after an interruption.")
    snr = payload.get("added_noise_snr_db")
    if snr is not None and (
        isinstance(snr, bool)
        or not isinstance(snr, (int, float))
        or snr not in (20, 30, 40)
    ):
        raise ValueError("Added noise must be off, 40 dB, 30 dB, or 20 dB.")
    return voltage, model


@lru_cache(maxsize=2)
def load_bundle(path, modified_ns):
    # The server loads only this local training output, never an uploaded model.
    bundle = joblib.load(path)
    if (
        bundle["feature_version"] != FEATURE_VERSION
        or bundle["feature_names"] != list(FEATURE_NAMES)
        or bundle["sklearn_version"] != sklearn.__version__
        or bundle["sample_count"] != SAMPLE_COUNT
        or bundle["sample_rate_hz"] != SAMPLE_RATE_HZ
        or bundle["classes"] != list(SIGNAL_TYPES)
        or set(bundle["models"]) != {"random_forest", "svm"}
    ):
        raise ValueError("The model bundle does not match this application.")
    return bundle


@app.get("/")
def home():
    return app.send_static_file("index.html")


@app.post("/api/classify")
def classify():
    try:
        payload = request.get_json()
    except (BadRequest, UnsupportedMediaType):
        return jsonify(error="Send valid JSON with Content-Type application/json."), 400
    try:
        voltage, model_name = validate_recording(payload)
    except ValueError as error:
        return jsonify(error=str(error)), 422
    path = Path(app.config["MODEL_BUNDLE"])
    analysis_started = perf_counter()
    try:
        bundle = load_bundle(str(path.resolve()), path.stat().st_mtime_ns)
        features = extract_features(voltage, SAMPLE_RATE_HZ)
        prediction = str(bundle["models"][model_name].predict(features.reshape(1, -1))[0])
        if prediction not in SIGNAL_TYPES:
            raise ValueError("Unsupported model result.")
    except Exception:
        app.logger.exception("Classification model could not be used")
        return jsonify(error="The classification service needs its trained models. Contact the application operator."), 503
    analysis_time_ms = (perf_counter() - analysis_started) * 1000
    try:
        save_analysis(
            app.config["DATABASE_PATH"], prediction=prediction, model=model_name,
            added_noise_snr_db=payload.get("added_noise_snr_db"),
            analysis_time_ms=analysis_time_ms,
        )
    except (sqlite3.Error, OSError):
        app.logger.exception("Classification history could not be saved")
        return jsonify(error="The analysis could not be saved. Please try again later."), 503
    return jsonify(prediction=prediction)


@app.get("/api/history")
def history():
    try:
        limit = int(request.args.get("limit", "20"))
        records = read_history(app.config["DATABASE_PATH"], limit)
    except ValueError:
        return jsonify(error="History limit must be a whole number between 1 and 100."), 400
    except (sqlite3.Error, OSError):
        app.logger.exception("Classification history could not be read")
        return jsonify(error="Analysis history is unavailable. Please try again later."), 503
    return jsonify(history=records)


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(error):
    return jsonify(error="The recording exceeds the 5 MB request limit."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
