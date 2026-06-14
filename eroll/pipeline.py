"""Roll pipeline primitives: unique-token extraction and word-map join-back.

The LLM step itself is delegated to ``indicate.batch.transliterate_tokens_batched``
(batch-mode, checkpointed). This module only does the cheap, deterministic data work:
collecting unique native-script tokens from roll columns, and substituting a resolved
``token -> english`` map back into each cell.
"""

import re
from collections.abc import Iterable

import pandas as pd


def extract_unique_native_tokens(
    chunks: Iterable[pd.DataFrame],
    columns: Iterable[str],
    native_run: re.Pattern[str],
) -> set[str]:
    """Collect every unique native-script run across ``columns`` of all chunks."""
    columns = list(columns)
    tokens: set[str] = set()
    for df in chunks:
        for col in columns:
            if col not in df.columns:
                continue
            for value in df[col].dropna().unique():
                tokens.update(native_run.findall(str(value)))
    return tokens


def join_back(
    df: pd.DataFrame,
    columns: Iterable[str],
    word_map: dict[str, str],
    native_run: re.Pattern[str],
    suffix: str = "_t13n_llm",
) -> pd.DataFrame:
    """Add ``<col><suffix>`` columns with native runs replaced via ``word_map``.

    Mirrors the notebook's fast path: transliterate each column's *unique* values
    once, then map back to every row. Unknown runs (not in ``word_map``) are left
    unchanged, so the output degrades gracefully. Returns ``df`` (mutated in place).
    """
    columns = list(columns)

    def _sub(text: object) -> object:
        if pd.isna(text):
            return text
        return native_run.sub(lambda m: word_map.get(m.group(0), m.group(0)), str(text))

    for col in columns:
        if col not in df.columns:
            continue
        value_map = {value: _sub(value) for value in df[col].dropna().unique()}
        df[f"{col}{suffix}"] = df[col].map(value_map)
    return df
