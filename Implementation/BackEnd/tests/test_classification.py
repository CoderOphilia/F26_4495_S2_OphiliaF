"""Check known features, partition isolation, persisted models, and API errors."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app, load_bundle
from feature_extraction import FEATURE_NAMES, calculate_peak_amplitude, calculate_rms, extract_features
from evaluate_noise import add_noise, run_noise_experiment
from generate_dataset import build_dataset
from signal_generator import SIGNAL_TYPES, generate_signal
from train_models import train_models


class FeatureTests(unittest.TestCase):
    def test_known_normal_and_harmonic_measurements(self):
        _, normal = generate_signal("normal")
        measured = dict(zip(FEATURE_NAMES, extract_features(normal, 6000)))
        self.assertAlmostEqual(measured["rms"], 1 / np.sqrt(2))
        self.assertAlmostEqual(measured["peak_amplitude"], 1.0)
        self.assertAlmostEqual(measured["crest_factor"], np.sqrt(2))
        self.assertAlmostEqual(measured["cycle_rms_min_ratio"], 1.0)
        self.assertAlmostEqual(measured["cycle_rms_max_ratio"], 1.0)
        self.assertEqual(measured["dominant_frequency_hz"], 60)
        _, harmonic = generate_signal("harmonic_distortion")
        measured = dict(zip(FEATURE_NAMES, extract_features(harmonic, 6000)))
        self.assertAlmostEqual(measured["third_harmonic_power_ratio"], 0.04 / 1.05, places=6)
        self.assertAlmostEqual(measured["fifth_harmonic_power_ratio"], 0.01 / 1.05, places=6)

    def test_cycle_features_capture_magnitude_events(self):
        for label in ("sag", "swell", "interruption"):
            with self.subTest(label=label):
                _, voltage = generate_signal(label)
                measured = dict(zip(FEATURE_NAMES, extract_features(voltage, 6000)))
                self.assertGreater(measured["cycle_rms_max_ratio"], measured["cycle_rms_min_ratio"])
                self.assertGreater(measured["cycle_rms_std_ratio"], 0)
        self.assertEqual(calculate_peak_amplitude([-1.5, 0.2, 1]), 1.5)
        self.assertAlmostEqual(calculate_rms([1e200, -1e200]), 1e200)
        np.testing.assert_array_equal(extract_features(np.zeros(1200), 6000), np.zeros(len(FEATURE_NAMES)))

    def test_bad_features_are_rejected(self):
        for invalid in ([], [[1, 2]], [np.nan], [np.inf], [1j], ["invalid"]):
            for function in (calculate_rms, calculate_peak_amplitude):
                with self.subTest(function=function.__name__, invalid=invalid):
                    with self.assertRaises(ValueError):
                        function(invalid)
        with self.assertRaises(ValueError):
            extract_features([1, 2], 6000)
        with self.assertRaises(ValueError):
            extract_features(np.ones(1200), 500)

    def test_added_noise_has_the_requested_measured_snr(self):
        _, signal = generate_signal("normal")
        for snr in (40, 30, 20):
            noisy = add_noise(signal, snr, np.random.default_rng(42))
            measured = 10 * np.log10(np.mean(signal ** 2) / np.mean((noisy - signal) ** 2))
            self.assertAlmostEqual(measured, snr, places=8)
            np.testing.assert_array_equal(noisy, add_noise(signal, snr, np.random.default_rng(42)))


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls.temporary.name) / "data"
        cls.models = Path(cls.temporary.name) / "models"
        build_dataset(cls.dataset, samples_per_class=20)
        cls.report = train_models(cls.dataset, cls.models)
        cls.bundle = joblib.load(cls.models / "classifiers.joblib")

    @classmethod
    def tearDownClass(cls):
        load_bundle.cache_clear()
        cls.temporary.cleanup()

    def setUp(self):
        self.previous_config = app.config.copy()
        self.request_folder = tempfile.TemporaryDirectory(dir=self.temporary.name)
        self.database_path = Path(self.request_folder.name) / "analysis_history.sqlite3"
        app.config.update(
            TESTING=True, MODEL_BUNDLE=self.models / "classifiers.joblib",
            DATABASE_PATH=self.database_path,
        )
        self.client = app.test_client()
        time, voltage = generate_signal("normal")
        self.payload = {
            "time_seconds": time.tolist(), "voltage_relative": voltage.tolist(),
            "sample_rate_hz": 6000, "model": "random_forest",
            "cleaned_data_confirmed": True, "added_noise_snr_db": None,
        }

    def tearDown(self):
        app.config.clear()
        app.config.update(self.previous_config)
        self.request_folder.cleanup()

    def test_saved_models_match_training_only_scaling_and_partitions(self):
        with np.load(self.dataset / "signals.npz") as archive:
            features = np.stack([extract_features(row, 6000) for row in archive["voltage_relative"]])
            train_mask = archive["splits"] == "train"
            scaler = self.bundle["models"]["svm"].named_steps["standardscaler"]
            np.testing.assert_allclose(scaler.mean_, features[train_mask].mean(axis=0))
            self.assertEqual(scaler.n_samples_seen_, 70)
        ids = self.report["partition_ids"]
        self.assertFalse(set(ids["train"]) & set(ids["test"]))
        self.assertFalse(set(ids["train"]) & set(ids["validation"]))
        for result in self.report["models"].values():
            self.assertEqual(np.sum(result["test"]["confusion_matrix"]), 15)
            self.assertGreaterEqual(result["test"]["macro_f1"], 0.7)
        self.assertEqual(json.loads((self.models / "evaluation.json").read_text()), self.report)
        with self.assertRaises(FileExistsError):
            train_models(self.dataset, self.models)

    def test_five_csv_uploads_work_with_both_models(self):
        for label in SIGNAL_TYPES:
            values = np.loadtxt(self.dataset / "examples" / f"{label}_signal.csv", delimiter=",", skiprows=1)
            for model in ("random_forest", "svm"):
                with self.subTest(label=label, model=model):
                    payload = dict(self.payload, time_seconds=values[:, 0].tolist(), voltage_relative=values[:, 1].tolist(), model=model)
                    response = self.client.post("/api/classify", json=payload)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json, {"prediction": label})

    def test_noise_preview_is_classified_without_adding_noise_twice(self):
        for snr in (40, 30, 20):
            payload = copy.deepcopy(self.payload)
            values = np.asarray(payload["voltage_relative"])
            noise = np.random.default_rng(42).normal(size=len(values)) * np.sqrt(np.mean(values ** 2) / 10 ** (snr / 10))
            payload.update(voltage_relative=(values + noise).tolist(), added_noise_snr_db=snr)
            response = self.client.post("/api/classify", json=payload)
            self.assertEqual(response.status_code, 200)
            expected = str(self.bundle["models"]["random_forest"].predict(extract_features(values + noise, 6000).reshape(1, -1))[0])
            self.assertEqual(response.json["prediction"], expected)

    def test_noise_experiment_preserves_models_and_uses_test_parents(self):
        path = self.models / "classifiers.joblib"
        model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        output = Path(self.temporary.name) / "noise"
        report = run_noise_experiment(self.dataset, self.models, output)
        self.assertEqual(set(report["test_signal_ids"]), set(self.report["partition_ids"]["test"]))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), model_hash)
        self.assertEqual(set(report["conditions"]), {"clean", "40_db", "30_db", "20_db"})
        for name in ("random_forest", "svm"):
            self.assertEqual(report["conditions"]["clean"][name], self.report["models"][name]["test"])
        self.assertTrue((output / "noise_comparison.png").is_file())
        self.assertEqual(json.loads((output / "results.json").read_text()), report)
        with self.assertRaises(FileExistsError):
            run_noise_experiment(self.dataset, self.models, output)

    def test_invalid_requests_return_actionable_json(self):
        cases = [
            {"cleaned_data_confirmed": False}, {"cleaned_data_confirmed": "true"},
            {"model": "../other"}, {"model": ["svm"]},
            {"voltage_relative": [0]}, {"voltage_relative": [True] * 1200},
            {"voltage_relative": [None] * 1200}, {"voltage_relative": ["1"] * 1200},
            {"voltage_relative": [np.nan] * 1200}, {"voltage_relative": [0] * 1200},
            {"voltage_relative": [230] * 1200}, {"time_seconds": [0] * 1200},
            {"time_seconds": [index / 5000 for index in range(1200)]},
            {"sample_rate_hz": 5000}, {"sample_rate_hz": True},
            {"sample_rate_hz": 10 ** 400},
            {"added_noise_snr_db": 10}, {"added_noise_snr_db": []},
        ]
        for updates in cases:
            with self.subTest(fields=list(updates)):
                response = self.client.post("/api/classify", json=dict(self.payload, **updates))
                self.assertEqual(response.status_code, 422)
                self.assertIn("error", response.json)
        for body in ([], None, "text"):
            response = self.client.post("/api/classify", data=json.dumps(body), content_type="application/json")
            self.assertEqual(response.status_code, 422)
        response = self.client.post("/api/classify", data="{", content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post("/api/classify", data="text").status_code, 400)
        self.assertFalse(self.database_path.exists())

    def test_successful_analyses_persist_in_history(self):
        self.assertEqual(self.client.get("/api/history").json, {"history": []})
        self.assertFalse(self.database_path.exists())
        for model, snr in (("random_forest", None), ("svm", 40), ("svm", 20)):
            response = self.client.post("/api/classify", json=dict(self.payload, model=model, added_noise_snr_db=snr))
            self.assertEqual(response.status_code, 200)
        # A new client reads data saved through earlier, closed connections.
        response = app.test_client().get("/api/history?limit=2")
        self.assertEqual(response.status_code, 200)
        records = response.json["history"]
        self.assertEqual(len(records), 2)
        self.assertEqual([row["added_noise_snr_db"] for row in records], [20, 40])
        self.assertGreater(records[0]["id"], records[1]["id"])
        for row in records:
            self.assertEqual(row["prediction"], "normal")
            self.assertEqual(row["model"], "svm")
            self.assertGreaterEqual(row["analysis_time_ms"], 0)
            self.assertTrue(row["analyzed_at_utc"].endswith("+00:00"))
            self.assertNotIn("voltage_relative", row)
        all_records = self.client.get("/api/history").json["history"]
        self.assertEqual(len(all_records), 3)
        self.assertIsNone(all_records[-1]["added_noise_snr_db"])

    def test_history_rejects_invalid_limits(self):
        for limit in ("0", "101", "-1", "1.5", "text", "", "1;DROP TABLE classification_history"):
            response = self.client.get("/api/history", query_string={"limit": limit})
            self.assertEqual(response.status_code, 400)
            self.assertIn("error", response.json)
        self.assertFalse(self.database_path.exists())

    def test_database_failures_return_service_errors(self):
        app.config["DATABASE_PATH"] = Path(self.request_folder.name)
        with self.assertLogs(app.logger, level="ERROR"):
            response = self.client.post("/api/classify", json=self.payload)
        self.assertEqual(response.status_code, 503)
        self.assertIn("error", response.json)
        with self.assertLogs(app.logger, level="ERROR"):
            response = self.client.get("/api/history")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(self.database_path.exists())

    def test_missing_or_incompatible_models_return_service_error(self):
        app.config["MODEL_BUNDLE"] = self.models / "missing.joblib"
        with self.assertLogs(app.logger, level="ERROR"):
            self.assertEqual(self.client.post("/api/classify", json=self.payload).status_code, 503)
        self.assertFalse(self.database_path.exists())
        incompatible = dict(self.bundle, feature_version=-1)
        path = Path(self.temporary.name) / "incompatible.joblib"
        joblib.dump(incompatible, path)
        app.config["MODEL_BUNDLE"] = path
        with self.assertLogs(app.logger, level="ERROR"):
            self.assertEqual(self.client.post("/api/classify", json=self.payload).status_code, 503)

    def test_static_entry_and_request_size_limit(self):
        for path in ("/", "/style.css", "/script.js"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            response.close()
        app.config["MAX_CONTENT_LENGTH"] = 100
        response = self.client.post("/api/classify", json=self.payload)
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.json)


if __name__ == "__main__":
    unittest.main()
