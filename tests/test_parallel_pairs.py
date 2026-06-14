"""Tests for the cross-file id-join word-pair builder (real duckdb on tiny files)."""

import re
import tempfile
import unittest
from pathlib import Path

from eroll.parallel_pairs import SCRIPT_RANGES, build_parallel_word_map

MAL = re.compile(r"[ഀ-ൿ]+")


def _write(path: Path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


class TestParallelPairs(unittest.TestCase):
    def test_join_align_modal_and_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            nat = Path(tmp) / "mal.csv"
            eng = Path(tmp) / "eng.csv"
            # id=1,2 corroborate "radha"->radha twice (kept); id=3 is a single sighting
            # (dropped by min_count=2); id=4 native token "ര" is 1 char (dropped by min_len).
            _write(
                nat,
                ["id", "elector_name"],
                [["1", "രാധ"], ["2", "രാധ"], ["3", "കുമ"], ["4", "ര"]],
            )
            _write(
                eng,
                ["id", "elector_name"],
                [["1", "Radha"], ["2", "Radha"], ["3", "Kuma"], ["4", "Ra"]],
            )
            wm, stats = build_parallel_word_map(
                nat,
                eng,
                join_key="id",
                fields=["elector_name"],
                native_run=MAL,
                min_len=3,
                min_count=2,
            )
            self.assertEqual(
                wm, {"രാധ": "radha"}
            )  # only the corroborated >=3-char token
            self.assertEqual(stats["joined_rows"], 4)
            self.assertEqual(stats["kept"], 1)

    def test_word_count_mismatch_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            nat = Path(tmp) / "mal.csv"
            eng = Path(tmp) / "eng.csv"
            # native 1 run vs english 3 runs -> not aligned, no pair emitted.
            _write(nat, ["id", "elector_name"], [["1", "രാധ"]])
            _write(eng, ["id", "elector_name"], [["1", "Radha K Nair"]])
            wm, stats = build_parallel_word_map(
                nat,
                eng,
                join_key="id",
                fields=["elector_name"],
                native_run=MAL,
                min_len=3,
                min_count=1,
            )
            self.assertEqual(wm, {})
            self.assertEqual(stats["mismatched"], 1)

    def test_script_ranges_cover_target_languages(self):
        for lang in ("bengali", "gujarati", "malayalam", "kannada", "telugu"):
            self.assertIn(lang, SCRIPT_RANGES)


if __name__ == "__main__":
    unittest.main()
