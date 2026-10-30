"""Check waveform mathematics, data provenance, and reproducible partitions."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_dataset import build_dataset
from signal_generator import SIGNAL_TYPES, generate_normal_signal, generate_signal


class SignalTests(unittest.TestCase):
    def test_original_normal_example(self):
        time, voltage = generate_normal_signal()
        self.assertEqual(len(voltage), 1200)
        self.assertAlmostEqual(float(np.diff(time).mean()), 1 / 6000)
        self.assertAlmostEqual(float(np.sqrt(np.mean(voltage ** 2))), 1 / np.sqrt(2))
        frequency_axis = np.fft.rfftfreq(len(voltage), d=1 / 6000)
        self.assertEqual(frequency_axis[np.argmax(np.abs(np.fft.rfft(voltage)))], 60)

    def test_magnitude_events_modify_only_the_requested_interval(self):
        time, reference = generate_normal_signal(amplitude=1.08, phase=0.73)
        event = (time >= 0.043) & (time < 0.137)
        for label, multiplier in (("sag", 0.45), ("swell", 1.65), ("interruption", 0.02)):
            with self.subTest(label=label):
                _, voltage = generate_signal(
                    label, amplitude=1.08, phase=0.73,
                    event_start_seconds=0.043, event_end_seconds=0.137,
                    event_multiplier=multiplier,
                )
                np.testing.assert_array_equal(voltage[~event], reference[~event])
                np.testing.assert_allclose(voltage[event], reference[event] * multiplier)

    def test_harmonic_frequencies_and_amplitudes(self):
        _, voltage = generate_signal("harmonic_distortion")
        spectrum = 2 * np.abs(np.fft.rfft(voltage)) / len(voltage)
        frequencies = np.fft.rfftfreq(len(voltage), d=1 / 6000)
        for frequency, expected in ((60, 1.0), (180, 0.2), (300, 0.1)):
            self.assertAlmostEqual(spectrum[np.flatnonzero(frequencies == frequency)[0]], expected)
        spectrum[np.isin(frequencies, [60, 180, 300])] = 0
        self.assertLess(float(spectrum.max()), 1e-12)

    def test_invalid_parameters_and_undersampling_are_rejected(self):
        cases = [
            ("unknown", {}), ("normal", {"amplitude": 0}),
            ("normal", {"phase": float("nan")}),
            ("normal", {"duration_seconds": 0}),
            ("harmonic_distortion", {"sample_rate_hz": 500}),
            ("harmonic_distortion", {"third_harmonic_ratio": -0.1}),
            ("sag", {"event_multiplier": 1.5}),
            ("swell", {"event_multiplier": 0.5}),
            ("interruption", {"event_multiplier": 0.5}),
            ("sag", {"event_start_seconds": -0.1}),
            ("sag", {"event_end_seconds": 0.2}),
        ]
        for label, arguments in cases:
            with self.subTest(label=label, arguments=arguments):
                with self.assertRaises(ValueError):
                    generate_signal(label, **arguments)


class DatasetTests(unittest.TestCase):
    def test_labels_partitions_metadata_and_examples(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dataset"
            manifest = build_dataset(output, samples_per_class=20, seed=42)
            self.assertEqual(manifest["split_counts"], {"train": 70, "validation": 15, "test": 15})
            self.assertEqual(manifest["signals_npz_sha256"], hashlib.sha256((output / "signals.npz").read_bytes()).hexdigest())
            with (output / "metadata.csv").open(newline="", encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            with np.load(output / "signals.npz", allow_pickle=False) as data:
                self.assertEqual(data["voltage_relative"].shape, (100, 1200))
                self.assertTrue(np.isfinite(data["voltage_relative"]).all())
                self.assertEqual(len(set(data["signal_ids"])), 100)
                self.assertEqual(len(np.unique(data["voltage_relative"], axis=0)), 100)
                for label in SIGNAL_TYPES:
                    class_rows = data["labels"] == label
                    self.assertEqual(int(class_rows.sum()), 20)
                    for split, count in (("train", 14), ("validation", 3), ("test", 3)):
                        self.assertEqual(int(np.count_nonzero(class_rows & (data["splits"] == split))), count)
                # Reconstruct every waveform using the saved parameters,
                # which also checks that shuffled metadata and arrays align.
                for index, row in enumerate(records):
                    self.assertEqual(row["signal_id"], data["signal_ids"][index])
                    self.assertEqual(row["label"], data["labels"][index])
                    self.assertEqual(row["split"], data["splits"][index])
                    parameters = {
                        key: float(value) for key, value in row.items()
                        if key not in ("signal_id", "label", "split", "sample_seed") and value
                    }
                    time, voltage = generate_signal(row["label"], **parameters)
                    np.testing.assert_array_equal(time, data["time_seconds"])
                    np.testing.assert_array_equal(voltage, data["voltage_relative"][index])
                for label, example in manifest["examples"].items():
                    csv_path = output / example["path"]
                    with csv_path.open(encoding="utf-8") as stream:
                        self.assertEqual(stream.readline().strip(), "time_seconds,voltage_relative")
                    values = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                    record_index = int(np.flatnonzero(data["signal_ids"] == example["signal_id"])[0])
                    self.assertEqual(data["labels"][record_index], label)
                    self.assertEqual(data["splits"][record_index], "train")
                    np.testing.assert_allclose(values[:, 0], data["time_seconds"], rtol=1e-10, atol=1e-12)
                    np.testing.assert_allclose(values[:, 1], data["voltage_relative"][record_index], rtol=1e-10, atol=1e-12)
            self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)

    def test_same_seed_reproduces_data_and_different_seed_changes_it(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ("first", "repeat", "different")]
            for path, seed in zip(paths, [42, 42, 43]):
                build_dataset(path, samples_per_class=20, seed=seed)
            with np.load(paths[0] / "signals.npz") as first, np.load(paths[1] / "signals.npz") as repeat, np.load(paths[2] / "signals.npz") as different:
                for name in first.files:
                    np.testing.assert_array_equal(first[name], repeat[name])
                self.assertFalse(np.array_equal(first["voltage_relative"], different["voltage_relative"]))
            self.assertEqual((paths[0] / "metadata.csv").read_bytes(), (paths[1] / "metadata.csv").read_bytes())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "notes.txt"
            path.write_text("Keep this work.")
            with self.assertRaises(FileExistsError):
                build_dataset(folder, samples_per_class=20)
            self.assertEqual(path.read_text(), "Keep this work.")
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_invalid_dataset_sizes_and_seeds_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for count, seed in ((0, 42), (19, 42), (20.5, 42), (True, 42), (20, -1), (20, True)):
                with self.subTest(count=count, seed=seed):
                    with self.assertRaises(ValueError):
                        build_dataset(Path(folder) / "unused", count, seed)
            self.assertFalse((Path(folder) / "unused").exists())


if __name__ == "__main__":
    unittest.main()
