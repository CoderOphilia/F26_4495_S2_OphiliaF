"""Check SQLite transactions and concurrent result saving."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history import read_history, save_analysis


class HistoryTests(unittest.TestCase):
    def test_failed_insert_preserves_existing_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            record = dict(prediction="sag", model="svm", added_noise_snr_db=None, analysis_time_ms=1.25)
            save_analysis(path, **record)
            original = read_history(path)
            with self.assertRaises(sqlite3.IntegrityError):
                save_analysis(path, **dict(record, model="invalid"))
            self.assertEqual(read_history(path), original)
            save_analysis(path, **record)
            self.assertEqual(len(read_history(path)), 2)

    def test_parallel_saves_keep_every_complete_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "history.sqlite3"
            def save(index):
                save_analysis(
                    path, prediction="normal", model="random_forest",
                    added_noise_snr_db=30, analysis_time_ms=float(index),
                )
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(save, range(12)))
            records = read_history(path, limit=100)
            self.assertEqual(len(records), 12)
            self.assertEqual(len({row["id"] for row in records}), 12)
            self.assertEqual({row["analysis_time_ms"] for row in records}, set(range(12)))


if __name__ == "__main__":
    unittest.main()
