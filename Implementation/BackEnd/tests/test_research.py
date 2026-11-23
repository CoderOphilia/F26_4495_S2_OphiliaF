"""Check noise pairing and that research archives cannot silently change."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_extraction import FEATURE_NAMES, extract_features
from generate_dataset import build_dataset
from research_validation import FEATURE_GROUPS, load_archive, noisy_features
from signal_generator import generate_signal
from train_models import model_candidates


class ResearchTests(unittest.TestCase):
    def test_noise_seed_changes_measurements_but_is_repeatable(self):
        _, signal = generate_signal("normal")
        first = noisy_features([signal], ["test-1"], 20, 2026)
        np.testing.assert_array_equal(first, noisy_features([signal], ["test-1"], 20, 2026))
        self.assertFalse(np.array_equal(first, noisy_features([signal], ["test-1"], 20, 2027)))
        np.testing.assert_allclose(noisy_features([signal], ["test-1"], None, 2026)[0], extract_features(signal, 6000))

    def test_groups_cover_features_and_estimators_are_independent(self):
        self.assertEqual(FEATURE_GROUPS["time"] + FEATURE_GROUPS["frequency"], list(range(len(FEATURE_NAMES))))
        first, second = model_candidates(), model_candidates()
        first["random_forest"][0].set_params(max_depth=1)
        self.assertEqual(second["random_forest"][0].max_depth, 8)
        self.assertFalse(hasattr(second["svm"][0].steps[0][1], "mean_"))

    def test_archive_checksum_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "dataset"
            build_dataset(destination, samples_per_class=20, seed=67)
            data, _, _ = load_archive(destination)
            partitions = {name: set(data["signal_ids"][data["splits"] == name]) for name in ("train", "validation", "test")}
            self.assertFalse(partitions["train"] & partitions["test"])
            self.assertFalse(partitions["validation"] & partitions["test"])
            archive = destination / "signals.npz"
            archive.write_bytes(archive.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_archive(destination)
