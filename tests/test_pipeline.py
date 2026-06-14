"""Tests for unique-token extraction and word-map join-back."""

from __future__ import annotations

import re
import unittest

import pandas as pd

from eroll.pipeline import extract_unique_native_tokens, join_back

GURMUKHI = re.compile(r"[਀-੿]+")


class TestPipeline(unittest.TestCase):
    def test_extract_unique_native_tokens(self):
        df = pd.DataFrame(
            {
                "elector_name": ["ਰਾਜ ਸਿੰਘ", "ਰਾਜ", "Bob Smith", None],
                "ignored": ["ਪੰਜਾਬ", "x", "y", "z"],
            }
        )
        tokens = extract_unique_native_tokens([df], ["elector_name"], GURMUKHI)
        self.assertEqual(tokens, {"ਰਾਜ", "ਸਿੰਘ"})

    def test_join_back(self):
        df = pd.DataFrame({"elector_name": ["ਰਾਜ Singh", None, "ਪੰਜਾਬ", "ਅਣਜਾਣ"]})
        word_map = {"ਰਾਜ": "raj", "ਪੰਜਾਬ": "punjab"}
        join_back(df, ["elector_name"], word_map, GURMUKHI, suffix="_t13n_llm")
        out = df["elector_name_t13n_llm"].tolist()
        self.assertEqual(out[0], "raj Singh")  # native run replaced, Latin kept
        self.assertTrue(pd.isna(out[1]))  # NaN preserved
        self.assertEqual(out[2], "punjab")
        self.assertEqual(out[3], "ਅਣਜਾਣ")  # unknown run left unchanged

    def test_join_back_skips_missing_column(self):
        df = pd.DataFrame({"elector_name": ["ਰਾਜ"]})
        # "father_or_husband_name" absent -> skipped, no error.
        join_back(
            df, ["elector_name", "father_or_husband_name"], {"ਰਾਜ": "raj"}, GURMUKHI
        )
        self.assertIn("elector_name_t13n_llm", df.columns)
        self.assertNotIn("father_or_husband_name_t13n_llm", df.columns)


if __name__ == "__main__":
    unittest.main()
