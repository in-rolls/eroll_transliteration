"""Tests for (native, english) pair extraction, incl. the Punjabi regression gate."""

from __future__ import annotations

import gzip
import os
import re
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from eroll.extract_pairs import extract_pairs
from eroll.states import NAME_PLACE_COLUMNS

GURMUKHI = re.compile(r"[਀-੿]+")


def _decompress(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.read()


class TestExtractPairs(unittest.TestCase):
    def test_alignment_and_modal_dedup(self):
        # rows: two exact pairs, one multi-word aligned pair, one mismatch (no Latin).
        table = pa.table(
            {
                "elector_name": ["ਰਾਜ", "ਰਾਜ", "ਰਾਜ ਸਿੰਘ", "ਕੌਰ"],
                "elector_name_transliterated": ["raj", "raj", "Raj Singh", "ਕੌਰ"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "parallel.parquet"
            out = Path(tmp) / "punjabi.csv.gz"
            pq.write_table(table, src)
            stats = extract_pairs(
                source=src,
                out=out,
                native_run=GURMUKHI,
                fields=["elector_name"],
                header=("punjabi", "english"),
                translit_suffix="_transliterated",
                progress=False,
            )
            content = _decompress(out)

        self.assertEqual(stats["mismatched"], 1)  # ਕੌਰ has no Latin partner
        self.assertEqual(stats["aligned"], 3)
        # Sorted, lowercased, modal English; ਕੌਰ excluded.
        self.assertEqual(content, "punjabi,english\nਰਾਜ,raj\nਸਿੰਘ,singh\n")

    @unittest.skipUnless(
        os.environ.get("EROLL_RUN_REGRESSION"),
        "slow (19M rows); set EROLL_RUN_REGRESSION=1 to run",
    )
    def test_punjabi_regression_byte_for_byte(self):
        """extract_pairs reproduces indicate's committed punjabi.csv.gz (decompressed)."""
        indicate_data = Path(
            os.environ.get(
                "INDICATE_DATA_DIR",
                Path(__file__).resolve().parents[2] / "indicate" / "data",
            )
        )
        parquet = indicate_data / "punjab_transliteration_subset.parquet"
        gold = indicate_data / "punjabi.csv.gz"
        if not parquet.is_file() or not gold.is_file():
            self.skipTest(f"missing {parquet} or {gold}")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "punjabi.csv.gz"
            extract_pairs(
                source=parquet,
                out=out,
                native_run=re.compile(r"[਀-੿]+"),
                fields=list(NAME_PLACE_COLUMNS),
                header=("punjabi", "english"),
                translit_suffix="_transliterated",
                max_len=40,
                batch_size=100_000,
                progress=False,
            )
            self.assertEqual(_decompress(out), _decompress(gold))


if __name__ == "__main__":
    unittest.main()
