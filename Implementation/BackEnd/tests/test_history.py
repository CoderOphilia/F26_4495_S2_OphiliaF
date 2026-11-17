"""Check SQLite transactions and concurrent result saving."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history import read_history, read_history_page, save_analysis


class HistoryTests(unittest.TestCase):
    def test_pages_remain_complete_when_new_analyses_arrive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            record = dict(prediction="normal", model="svm", added_noise_snr_db=None, analysis_time_ms=1.0)
            for _ in range(45):
                save_analysis(path, **record)
            first = read_history_page(path)
            self.assertEqual([row["id"] for row in first["history"]], list(range(45, 25, -1)))
            self.assertTrue(first["has_more"])
            save_analysis(path, **record)
            second = read_history_page(path, before_id=int(first["next_before_id"]))
            third = read_history_page(path, before_id=int(second["next_before_id"]))
            ids = [row["id"] for page in (first, second, third) for row in page["history"]]
            self.assertEqual(ids, list(range(45, 0, -1)))
            self.assertFalse(third["has_more"])
            self.assertIsNone(third["next_before_id"])
            self.assertEqual(read_history(path)[0]["id"], 46)

    def test_a_full_final_page_has_no_extra_page(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            for _ in range(20):
                save_analysis(path, prediction="sag", model="svm", added_noise_snr_db=40, analysis_time_ms=1.0)
            page = read_history_page(path)
            self.assertEqual(len(page["history"]), 20)
            self.assertFalse(page["has_more"])
            self.assertIsNone(page["next_before_id"])

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
