"""Tests for the per-state configuration registry."""

from __future__ import annotations

import unittest

from eroll.states import NAME_PLACE_COLUMNS, STATES, data_dir


class TestStates(unittest.TestCase):
    def test_registry_membership(self):
        self.assertIn("assam", STATES)
        self.assertIn("punjab", STATES)
        self.assertIn("west_bengal", STATES)
        # Andhra is intentionally excluded (already-romanized English).
        self.assertNotIn("andhra", STATES)

    def test_west_bengal_shares_bengali_corpus(self):
        wb = STATES["west_bengal"]
        self.assertEqual(wb.input_glob, "wb_all.csv.gz")
        self.assertEqual(wb.language, "bengali")
        # Same Bengali corpus as assam -- west_bengal appends net-new pairs to it.
        self.assertEqual(wb.corpus_csv.name, "bengali.csv.gz")
        self.assertEqual(wb.corpus_csv, STATES["assam"].corpus_csv)
        self.assertEqual(wb.columns, NAME_PLACE_COLUMNS)

    def test_columns_default_order(self):
        # Order must match indicate's extract_punjabi SOURCE_FIELDS for byte-exact reproduction.
        self.assertEqual(STATES["assam"].columns, NAME_PLACE_COLUMNS)
        self.assertEqual(NAME_PLACE_COLUMNS[0], "elector_name")
        self.assertEqual(len(NAME_PLACE_COLUMNS), 11)

    def test_assamese_native_run(self):
        run = STATES["assam"].native_run
        # Assamese/Bengali script kept, Latin dropped.
        self.assertEqual(run.findall("চঞ্চলা rani das"), ["চঞ্চলা"])

    def test_punjabi_native_run(self):
        run = STATES["punjab"].native_run
        self.assertEqual(run.findall("ਰਾਜ Singh ਸਿੰਘ"), ["ਰਾਜ", "ਸਿੰਘ"])

    def test_derived_paths(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"EROLL_DATA_DIR": "/tmp/rolls"}):
            self.assertEqual(str(data_dir()), "/tmp/rolls")
            assam = STATES["assam"]
            # Assam rolls are Bengali script -> the corpus is the shared bengali.csv.gz.
            self.assertEqual(assam.corpus_csv.name, "bengali.csv.gz")
            self.assertEqual(assam.parallel_parquet.name, "assam_parallel.parquet")
            # Punjab uses the original notebook's parquet/corpus names + suffix.
            punjab = STATES["punjab"]
            self.assertEqual(punjab.corpus_csv.name, "punjabi.csv.gz")
            self.assertEqual(
                punjab.parallel_parquet.name, "punjab_transliteration_subset.parquet"
            )
            self.assertEqual(punjab.translit_suffix, "_transliterated")


if __name__ == "__main__":
    unittest.main()
