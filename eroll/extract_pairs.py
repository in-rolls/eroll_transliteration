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
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

LATIN_RUN = re.compile(r"[A-Za-z]+")


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
