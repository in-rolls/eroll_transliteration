"""Tests for (native, english) pair extraction, incl. the Punjabi regression gate."""

import csv
import gzip
import os
import re
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from eroll.extract_pairs import (
    extract_pairs,
    load_corpus_native_keys,
    merge_word_map_into_corpus,
)
from eroll.states import NAME_PLACE_COLUMNS

GURMUKHI = re.compile(r"[਀-੿]+")


def _decompress(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.read()


def _write_corpus(path: Path, header, pairs) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(pairs)


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

    def test_load_corpus_native_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "bengali.csv.gz"
            _write_corpus(corpus, ("bengali", "english"), [("ক", "ka"), ("খ", "kha")])
            self.assertEqual(load_corpus_native_keys(corpus), {"ক", "খ"})
            # Missing file -> empty set (first harvest still works).
            self.assertEqual(load_corpus_native_keys(Path(tmp) / "nope.csv.gz"), set())

    def test_merge_appends_sorted_and_skips_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "bengali.csv.gz"
            _write_corpus(corpus, ("bengali", "english"), [("খ", "kha")])
            stats = merge_word_map_into_corpus(
                corpus_csv=corpus,
                # ক is net-new (lowercased on write); খ exists (not overwritten);
                # the long token is dropped by the max_len guard.
                word_map={"ক": "Ka", "খ": "DIFFERENT", "ক" * 50: "x"},
                header=("bengali", "english"),
                max_len=40,
            )
            self.assertEqual(stats["added"], 1)
            self.assertEqual(stats["skipped_existing"], 1)
            self.assertEqual(stats["skipped_len"], 1)
            self.assertEqual(stats["existing"], 1)
            self.assertEqual(stats["total"], 2)
            # Sorted union; new key added + lowercased; existing key preserved.
            self.assertEqual(_decompress(corpus), "bengali,english\nক,ka\nখ,kha\n")

    def test_merge_into_missing_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "bengali.csv.gz"
            stats = merge_word_map_into_corpus(
                corpus_csv=corpus,
                word_map={"ক": "ka"},
                header=("bengali", "english"),
            )
            self.assertEqual(stats["existing"], 0)
            self.assertEqual(stats["added"], 1)
            self.assertEqual(_decompress(corpus), "bengali,english\nক,ka\n")

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
