"""Build word pairs by joining a native-script roll to a parallel English roll on a key.

Some states ship the *same* roll in two languages (e.g. Kerala: Malayalam yearly files +
an English ``all_clean``). Joining them on the elector ``id`` yields native->English pairs
straight from the official roll -- authoritative spellings, no LLM. This is the cross-file
analogue of :func:`extract_pairs`: per id-matched row we align each field's native runs
(``[script]+``) with the English Latin runs (``[A-Za-z]+``); equal counts zip into pairs,
ambiguous tokens collapse to the modal English.

Cross-file joins are noisier than the LLM (the English file is consolidated across years, so
an ``id`` can drift to a different person), so two filters earn their keep: ``min_len`` drops
OCR fragments, and ``min_count`` keeps only pairs whose modal English is corroborated across
records. The 25M-row join runs out-of-core in duckdb; alignment/dedup stays in Python.
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

import click
import duckdb

from .extract_pairs import LATIN_RUN, merge_word_map_into_corpus
from .states import data_dir

# Unicode block bounds per script (lo, hi), for the native-run regex.
SCRIPT_RANGES: dict[str, tuple[str, str]] = {
    "bengali": ("ঀ", "৿"),
    "gujarati": ("઀", "૿"),
    "malayalam": ("ഀ", "ൿ"),
    "kannada": ("ಀ", "೿"),
    "telugu": ("ఀ", "౿"),
    "tamil": ("஀", "௿"),
}


def build_parallel_word_map(
    native_path: str | Path,
    english_path: str | Path,
    *,
    join_key: str,
    fields: list[str],
    native_run: re.Pattern[str],
    latin_run: re.Pattern[str] = LATIN_RUN,
    min_len: int = 3,
    min_count: int = 2,
    batch_size: int = 200_000,
) -> tuple[dict[str, str], dict[str, int]]:
    """Join two rolls on ``join_key`` and align ``fields`` into a ``native -> english`` map.

    The English side is de-duplicated to one row per key (``any_value``) so a repeated id
    can't fan out the join. Returns ``(word_map, stats)`` where stats has ``joined_rows,
    aligned, mismatched, tokens, kept``.
    """
    # n<i>/l<i> = native/Latin value of fields[i] from the matched (m)alayalam / (e)nglish row.
    select = ", ".join(
        f'm."{f}" AS n{i}, e."{f}" AS l{i}' for i, f in enumerate(fields)
    )
    eng_cols = ", ".join(f'any_value("{f}") AS "{f}"' for f in fields)
    # union_by_name lets ``native_path`` be a glob spanning multiple yearly files whose
    # column order/presence varies slightly; matching is by column name, not position.
    opts = (
        "header = true, all_varchar = true, ignore_errors = true, union_by_name = true"
    )
    query = f"""
        WITH e AS (
            SELECT "{join_key}" AS k, {eng_cols}
            FROM read_csv(?, {opts})
            GROUP BY "{join_key}"
        )
        SELECT {select}
        FROM read_csv(?, {opts}) m
        JOIN e ON m."{join_key}" = e.k
    """
    con = duckdb.connect()
    rel = con.execute(query, [str(english_path), str(native_path)])

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    joined = aligned = mismatched = 0
    nfields = len(fields)
    while True:
        rows = rel.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            joined += 1
            for i in range(nfields):
                nv, lv = row[2 * i], row[2 * i + 1]
                if not nv or not lv:
                    continue
                native = native_run.findall(nv)
                if not native:
                    continue
                latin = latin_run.findall(lv)
                if len(native) != len(latin):
                    mismatched += 1
                    continue
                aligned += 1
                for nat, lat in zip(native, latin, strict=False):
                    if len(nat) >= min_len:
                        counts[nat][lat.lower()] += 1
    con.close()

    word_map: dict[str, str] = {}
    for nat, eng_counts in counts.items():
        eng, n = eng_counts.most_common(1)[0]
        if n >= min_count and len(eng) >= min_len:
            word_map[nat] = eng
    return word_map, {
        "joined_rows": joined,
        "aligned": aligned,
        "mismatched": mismatched,
        "tokens": len(counts),
        "kept": len(word_map),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option(
    "--native", "native_path", required=True, help="Native-script roll (csv[.gz])."
)
@click.option("--english", "english_path", required=True, help="Parallel English roll.")
@click.option("--script", required=True, type=click.Choice(sorted(SCRIPT_RANGES)))
@click.option("--lang", required=True, help="Corpus language (header + <lang>.csv.gz).")
@click.option("--join-key", default="id", show_default=True)
@click.option(
    "--fields",
    default="elector_name,father_or_husband_name",
    show_default=True,
    help="Comma-separated columns to align (present in both rolls).",
)
@click.option("--min-len", default=3, show_default=True, type=int)
@click.option("--min-count", default=2, show_default=True, type=int)
@click.option(
    "--dry-run", is_flag=True, help="Report + sample, do not write the corpus."
)
def cli(
    native_path,
    english_path,
    script,
    lang,
    join_key,
    fields,
    min_len,
    min_count,
    dry_run,
):
    """Join a native roll to a parallel English roll and append pairs to <lang>.csv.gz."""
    lo, hi = SCRIPT_RANGES[script]
    native_run = re.compile(f"[{lo}-{hi}]+")
    field_list = [f.strip() for f in fields.split(",") if f.strip()]
    click.echo(f"[{lang}] joining on '{join_key}', fields={field_list} ...")
    word_map, stats = build_parallel_word_map(
        native_path,
        english_path,
        join_key=join_key,
        fields=field_list,
        native_run=native_run,
        min_len=min_len,
        min_count=min_count,
    )
    click.echo(
        f"  joined {stats['joined_rows']:,} rows; aligned {stats['aligned']:,} cells "
        f"({stats['mismatched']:,} mismatched); {stats['tokens']:,} tokens -> "
        f"{stats['kept']:,} kept (min_len={min_len}, min_count={min_count})"
    )
    if dry_run:
        for nat, eng in sorted(word_map.items())[:: max(1, len(word_map) // 25)][:25]:
            click.echo(f"    {nat} -> {eng}")
        return
    corpus = data_dir() / f"{lang}.csv.gz"
    ms = merge_word_map_into_corpus(
        corpus_csv=corpus, word_map=word_map, header=(lang, "english")
    )
    click.echo(
        f"[{lang}] appended {ms['added']:,} -> {corpus} "
        f"(corpus {ms['existing']:,} -> {ms['total']:,})"
    )


if __name__ == "__main__":
    cli()
