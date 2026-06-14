"""Extract clean ``(native, english)`` word pairs from a parallel-script parquet.

Generalized from ``indicate``'s ``training/extract_punjabi.py``. For every row/field
it pulls the native-script runs from ``<field>`` and the Latin runs from the
positionally-aligned ``<field><translit_suffix>``; when their counts match they zip
into word pairs. Ambiguous native words (more than one observed English) collapse to
the **modal** English. Output is a deduped, sorted ``(native, english)`` csv.gz.

Field order and ``batch_size`` are significant for exact reproduction of the original
Punjabi corpus: they fix the ``Counter`` insertion order (hence modal tie-breaking).
"""

from __future__ import annotations

import csv
import gzip
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TextIO, cast

import pyarrow.parquet as pq
from tqdm import tqdm

LATIN_RUN = re.compile(r"[A-Za-z]+")


def _open_text(path: Path, mode: str, *, gz: bool | None = None) -> TextIO:
    """Open a path text-mode, gzip-compressed when ``gz`` (default: path ends ``.gz``)."""
    if gz is None:
        gz = str(path).endswith(".gz")
    opener = gzip.open if gz else open
    return cast(TextIO, opener(path, mode, encoding="utf-8", newline=""))


def extract_pairs(
    *,
    source: str | Path,
    out: str | Path,
    native_run: re.Pattern[str],
    fields: list[str],
    header: tuple[str, str],
    translit_suffix: str = "_transliterated",
    latin_run: re.Pattern[str] = LATIN_RUN,
    max_len: int = 40,
    batch_size: int = 100_000,
    progress: bool = True,
) -> dict[str, int]:
    """Write ``(native, english)`` pairs from ``source`` parquet to ``out`` csv(.gz).

    Returns a stats dict: ``rows_seen, aligned, mismatched, pairs, ambiguous``.
    """
    source = Path(source)
    out = Path(out)
    parquet = pq.ParquetFile(source)
    columns = [c for f in fields for c in (f, f"{f}{translit_suffix}")]

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    rows_seen = aligned = mismatched = 0
    total = parquet.metadata.num_rows

    bar = tqdm(total=total, desc="rows", unit="row", disable=not progress)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        cols = batch.to_pydict()
        batch_len = len(cols[fields[0]])
        rows_seen += batch_len
        for field_name in fields:
            src_col = cols[field_name]
            tgt_col = cols[f"{field_name}{translit_suffix}"]
            for src, tgt in zip(src_col, tgt_col, strict=False):
                if not src or not tgt:
                    continue
                native = native_run.findall(src)
                if not native:
                    continue
                latin = latin_run.findall(tgt)
                if len(native) != len(latin):
                    mismatched += 1
                    continue
                aligned += 1
                for nat, lat in zip(native, latin, strict=False):
                    eng = lat.lower()
                    if 0 < len(nat) <= max_len and 0 < len(eng) <= max_len:
                        counts[nat][eng] += 1
        bar.update(batch_len)
    bar.close()

    ambiguous = 0
    pairs: list[tuple[str, str]] = []
    for native_word, eng_counts in counts.items():
        if len(eng_counts) > 1:
            ambiguous += 1
        pairs.append((native_word, eng_counts.most_common(1)[0][0]))
    pairs.sort()

    out.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(out).endswith(".gz") else open
    with opener(out, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(pairs)

    return {
        "rows_seen": rows_seen,
        "aligned": aligned,
        "mismatched": mismatched,
        "pairs": len(pairs),
        "ambiguous": ambiguous,
    }


def load_corpus_native_keys(corpus_csv: str | Path) -> set[str]:
    """Return the native-side keys already present in a ``(native, english)`` corpus.

    Reads the first column of an existing pairs ``csv(.gz)`` (skipping its header).
    Returns an empty set if the file does not exist, so a first harvest still works.
    """
    corpus_csv = Path(corpus_csv)
    if not corpus_csv.is_file():
        return set()
    keys: set[str] = set()
    with _open_text(corpus_csv, "rt") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if row:
                keys.add(row[0])
    return keys


def merge_word_map_into_corpus(
    *,
    corpus_csv: str | Path,
    word_map: dict[str, str],
    header: tuple[str, str],
    max_len: int = 40,
) -> dict[str, int]:
    """Append net-new ``token -> english`` pairs to a sorted/deduped corpus csv(.gz).

    Existing pairs are preserved (an existing native key is never overwritten); new keys
    are normalized exactly like :func:`extract_pairs` (english lowercased, both sides
    ``1..max_len`` chars). The full union is re-sorted and written atomically (temp file
    then ``os.replace``) so an interrupted write can never truncate the corpus.

    Returns stats: ``existing, added, skipped_existing, skipped_len, total``.
    """
    corpus_csv = Path(corpus_csv)
    merged: dict[str, str] = {}
    if corpus_csv.is_file():
        with _open_text(corpus_csv, "rt") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header
            for row in reader:
                if len(row) >= 2:
                    merged[row[0]] = row[1]
    existing = len(merged)

    added = skipped_existing = skipped_len = 0
    for native, english in word_map.items():
        eng = english.strip().lower()
        if not (0 < len(native) <= max_len and 0 < len(eng) <= max_len):
            skipped_len += 1
            continue
        if native in merged:
            skipped_existing += 1
            continue
        merged[native] = eng
        added += 1

    pairs = sorted(merged.items())
    corpus_csv.parent.mkdir(parents=True, exist_ok=True)
    gz = str(corpus_csv).endswith(".gz")
    tmp = corpus_csv.with_name(corpus_csv.name + ".tmp")
    with _open_text(tmp, "wt", gz=gz) as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(pairs)
    os.replace(tmp, corpus_csv)

    return {
        "existing": existing,
        "added": added,
        "skipped_existing": skipped_existing,
        "skipped_len": skipped_len,
        "total": len(pairs),
    }
