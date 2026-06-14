"""Orchestrator CLI: profile / sample / run / extract a state's roll.

    python -m eroll.transliterate_rolls --state assam profile
    python -m eroll.transliterate_rolls --state assam sample --sample-n 300
    python -m eroll.transliterate_rolls --state assam run
    python -m eroll.transliterate_rolls --state assam extract
    python -m eroll.transliterate_rolls --state assam all --yes

Memory-aware: roll inputs are streamed in chunks (never fully materialized) and the
parallel parquet is written incrementally. The LLM step is batch-mode and resumable
via a checkpoint, so ``run`` can be re-invoked to resume an in-flight batch.
"""

import json
import math
import random
import subprocess
import sys
from collections.abc import Iterator

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .extract_pairs import (
    extract_pairs,
    load_corpus_native_keys,
    merge_word_map_into_corpus,
)
from .pipeline import extract_unique_native_tokens, join_back
from .pricing import MODEL_PRICING, estimate_cost
from .states import STATES, StateConfig, data_dir


# --------------------------------------------------------------------------- #
# Streaming helpers
# --------------------------------------------------------------------------- #
def _read_header(cfg: StateConfig) -> list[str]:
    if cfg.archive:
        proc = subprocess.Popen(
            ["7z", "e", "-so", str(data_dir() / cfg.archive), cfg.input_glob],
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        header = pd.read_csv(
            proc.stdout, nrows=0, encoding="utf-8-sig"
        ).columns.tolist()
        proc.stdout.close()
        proc.wait()
        return header
    return pd.read_csv(cfg.input_path, nrows=0, encoding="utf-8-sig").columns.tolist()


def available_columns(cfg: StateConfig) -> list[str]:
    """cfg.columns that actually exist in the roll, in cfg order."""
    header = set(_read_header(cfg))
    return [c for c in cfg.columns if c in header]


def iter_chunks(
    cfg: StateConfig, usecols: list[str], limit: int | None = None
) -> Iterator[pd.DataFrame]:
    """Yield roll chunks (all-string dtype) of ``usecols``, optionally capped."""
    read_kwargs = {
        "usecols": usecols,
        "chunksize": cfg.csv_chunksize,
        "dtype": str,
        "keep_default_na": True,
        "encoding": "utf-8-sig",  # strip a leading BOM (Daman rolls have one); no-op otherwise
    }
    proc = None
    if cfg.archive:
        proc = subprocess.Popen(
            ["7z", "e", "-so", str(data_dir() / cfg.archive), cfg.input_glob],
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        reader = pd.read_csv(proc.stdout, **read_kwargs)
    else:
        reader = pd.read_csv(cfg.input_path, **read_kwargs)

    seen = 0
    try:
        for chunk in reader:
            if limit is not None and seen + len(chunk) >= limit:
                yield chunk.iloc[: max(0, limit - seen)]
                break
            seen += len(chunk)
            yield chunk
    finally:
        if proc is not None and proc.stdout:
            proc.stdout.close()
            proc.wait()


def _batch_kwargs(ctx: click.Context) -> dict:
    return {
        "provider": ctx.obj["provider"],
        "model": ctx.obj["model"],
        "group_size": ctx.obj["group_size"],
        "poll_interval": ctx.obj["poll_interval"],
        "max_wait": ctx.obj["max_wait"],
    }


# --------------------------------------------------------------------------- #
# CLI group
# --------------------------------------------------------------------------- #
@click.group()
@click.option("--state", required=True, type=click.Choice(sorted(STATES)))
@click.option("--provider", default=None, help="LLM provider (default: auto-detect).")
@click.option("--model", default=None, help="LLM model (default: provider default).")
@click.option("--group-size", default=25, show_default=True, type=int)
@click.option("--poll-interval", default=60.0, show_default=True, type=float)
@click.option(
    "--max-wait", default=None, type=float, help="Seconds before giving up a poll."
)
@click.pass_context
def cli(ctx, state, provider, model, group_size, poll_interval, max_wait):
    """Transliterate electoral rolls (forward: native script -> English)."""
    ctx.ensure_object(dict)
    ctx.obj.update(
        cfg=STATES[state],
        provider=provider,
        model=model,
        group_size=group_size,
        poll_interval=poll_interval,
        max_wait=max_wait,
    )


@cli.command()
@click.option("--limit", default=None, type=int, help="Cap rows scanned (dry run).")
@click.pass_context
def profile(ctx, limit):
    """Count unique native tokens per column and print a batch-cost estimate."""
    cfg: StateConfig = ctx.obj["cfg"]
    cols = available_columns(cfg)
    click.echo(f"[{cfg.name}] columns: {', '.join(cols)}")

    per_col: dict[str, set[str]] = {c: set() for c in cols}
    for chunk in iter_chunks(cfg, cols, limit=limit):
        for col in cols:
            for value in chunk[col].dropna().unique():
                per_col[col].update(cfg.native_run.findall(str(value)))
    cumulative: set[str] = set()
    for tokens in per_col.values():
        cumulative |= tokens

    click.echo("\n  column                         unique native tokens")
    for col in cols:
        click.echo(f"  {col:<30} {len(per_col[col]):>12,}")
    n_unique = len(cumulative)
    click.echo(f"  {'TOTAL (deduped)':<30} {n_unique:>12,}")

    # Rough cost estimate. chars/4 ~ tokens; output ~1.5x input chars (romanization).
    total_chars = sum(len(t) for t in cumulative)
    n_requests = math.ceil(n_unique / max(1, ctx.obj["group_size"]))
    overhead_in = 220  # ~system+few-shot prompt tokens per request
    avg_in = (total_chars / 4 / n_requests if n_requests else 0) + overhead_in
    avg_out = total_chars * 1.5 / 4 / n_requests if n_requests else 0
    click.echo(
        f"\nEstimated batch requests: {n_requests:,} (group_size={ctx.obj['group_size']})"
    )
    click.echo("Estimated cost (batch rates -- VERIFY CURRENT PRICING):")
    for model in MODEL_PRICING:
        cost = estimate_cost(n_requests, avg_in, avg_out, model)
        if cost is not None:
            click.echo(f"  {model:<22} ~${cost:,.2f}")


@cli.command()
@click.option("--sample-n", default=300, show_default=True, type=int)
@click.option(
    "--limit",
    default=500_000,
    show_default=True,
    type=int,
    help="Rows scanned for uniques.",
)
@click.pass_context
def sample(ctx, sample_n, limit):
    """Transliterate a random sample of unique tokens for human review."""
    from indicate.batch import transliterate_tokens_batched

    cfg: StateConfig = ctx.obj["cfg"]
    cols = available_columns(cfg)
    tokens = extract_unique_native_tokens(
        iter_chunks(cfg, cols, limit=limit), cols, cfg.native_run
    )
    token_list = sorted(tokens)
    rng = random.Random(42)
    chosen = rng.sample(token_list, min(sample_n, len(token_list)))
    click.echo(
        f"[{cfg.name}] sampling {len(chosen)} of {len(token_list):,} unique tokens"
    )

    ckpt = data_dir() / "eval" / f"{cfg.name}_sample.jsonl"
    word_map = transliterate_tokens_batched(
        chosen, cfg.language, "english", checkpoint_path=ckpt, **_batch_kwargs(ctx)
    )

    out_tsv = data_dir() / "eval" / f"{cfg.name}_sample.tsv"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8") as handle:
        handle.write("native\tllm_english\n")
        for token in chosen:
            handle.write(f"{token}\t{word_map.get(token, '')}\n")
    click.echo(
        f"Wrote {out_tsv} ({len(word_map)} resolved) -- review before a full run."
    )


@cli.command()
@click.option("--limit", default=None, type=int, help="Cap rows (testing).")
@click.pass_context
def run(ctx, limit):
    """Transliterate all unique tokens (batch) and write the parallel parquet."""
    from indicate.batch import transliterate_tokens_batched

    cfg: StateConfig = ctx.obj["cfg"]
    cols = available_columns(cfg)

    click.echo(f"[{cfg.name}] pass 1/2: collecting unique native tokens ...")
    tokens = extract_unique_native_tokens(
        iter_chunks(cfg, cols, limit=limit), cols, cfg.native_run
    )
    click.echo(f"  {len(tokens):,} unique tokens")

    ckpt = data_dir() / f"{cfg.name}_tokens.jsonl"
    click.echo(f"[{cfg.name}] transliterating (batch, resumable) ...")
    word_map = transliterate_tokens_batched(
        sorted(tokens),
        cfg.language,
        "english",
        checkpoint_path=ckpt,
        **_batch_kwargs(ctx),
    )
    click.echo(f"  resolved {len(word_map):,}/{len(tokens):,} tokens")

    written_cols = [*cols, *(f"{c}{cfg.translit_suffix}" for c in cols)]
    schema = pa.schema([(c, pa.string()) for c in written_cols])
    cfg.parallel_parquet.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"[{cfg.name}] pass 2/2: join-back -> {cfg.parallel_parquet}")
    writer = pq.ParquetWriter(cfg.parallel_parquet, schema)
    try:
        for chunk in iter_chunks(cfg, cols, limit=limit):
            join_back(chunk, cols, word_map, cfg.native_run, suffix=cfg.translit_suffix)
            table = pa.Table.from_pandas(
                chunk[written_cols].astype("string"),
                schema=schema,
                preserve_index=False,
            )
            writer.write_table(table)
    finally:
        writer.close()
    click.echo("  done.")


def _read_checkpoint_map(checkpoint_path) -> dict[str, str]:
    """Load a resolved ``token -> english`` map from indicate's JSONL checkpoint."""
    word_map: dict[str, str] = {}
    if not checkpoint_path.exists():
        return word_map
    with open(checkpoint_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                word_map[record["token"]] = record["translit"]
    return word_map


@cli.command()
@click.option("--limit", default=None, type=int, help="Cap rows scanned (testing).")
@click.pass_context
def harvest(ctx, limit):
    """Transliterate only net-new native tokens (batch) and append pairs to the corpus.

    Unlike ``run``, this never builds the parallel parquet: it collects unique tokens,
    drops those already in the corpus, transliterates just the remainder via Gemini/LLM
    batch (so you pay only for net-new), then appends the new ``(native, english)`` pairs.
    """
    from indicate.batch import transliterate_tokens_batched

    cfg: StateConfig = ctx.obj["cfg"]
    cols = available_columns(cfg)

    click.echo(f"[{cfg.name}] pass 1/1: collecting unique native tokens ...")
    tokens = extract_unique_native_tokens(
        iter_chunks(cfg, cols, limit=limit), cols, cfg.native_run
    )
    existing = load_corpus_native_keys(cfg.corpus_csv)
    net_new = sorted(tokens - existing)
    click.echo(
        f"  {len(tokens):,} unique, {len(existing):,} already in corpus -> "
        f"{len(net_new):,} net-new"
    )
    if not net_new:
        click.echo("  nothing net-new; corpus unchanged.")
        return

    ckpt = data_dir() / f"{cfg.name}_tokens.jsonl"
    click.echo(f"[{cfg.name}] transliterating net-new (batch, resumable) ...")
    word_map = transliterate_tokens_batched(
        net_new,
        cfg.language,
        "english",
        checkpoint_path=ckpt,
        **_batch_kwargs(ctx),
    )
    click.echo(f"  resolved {len(word_map):,}/{len(net_new):,} tokens")

    stats = merge_word_map_into_corpus(
        corpus_csv=cfg.corpus_csv,
        word_map=word_map,
        header=cfg.corpus_header,
        max_len=cfg.max_len,
    )
    click.echo(
        f"[{cfg.name}] appended {stats['added']:,} new pairs -> {cfg.corpus_csv} "
        f"(corpus {stats['existing']:,} -> {stats['total']:,}; "
        f"{stats['skipped_existing']:,} already present, {stats['skipped_len']:,} over-length)"
    )


@cli.command()
@click.pass_context
def merge(ctx):
    """Append an already-resolved token checkpoint into the corpus (no API calls)."""
    cfg: StateConfig = ctx.obj["cfg"]
    ckpt = data_dir() / f"{cfg.name}_tokens.jsonl"
    word_map = _read_checkpoint_map(ckpt)
    if not word_map:
        click.echo(f"[{cfg.name}] no resolved tokens at {ckpt}; nothing to merge.")
        return
    stats = merge_word_map_into_corpus(
        corpus_csv=cfg.corpus_csv,
        word_map=word_map,
        header=cfg.corpus_header,
        max_len=cfg.max_len,
    )
    click.echo(
        f"[{cfg.name}] appended {stats['added']:,} new pairs -> {cfg.corpus_csv} "
        f"(corpus {stats['existing']:,} -> {stats['total']:,}; "
        f"{stats['skipped_existing']:,} already present, {stats['skipped_len']:,} over-length)"
    )


@cli.command()
@click.pass_context
def extract(ctx):
    """Extract (native, english) pairs from the parallel parquet to the corpus csv.gz."""
    cfg: StateConfig = ctx.obj["cfg"]
    schema_names = set(pq.ParquetFile(cfg.parallel_parquet).schema_arrow.names)
    fields = [
        c
        for c in cfg.columns
        if c in schema_names and f"{c}{cfg.translit_suffix}" in schema_names
    ]
    stats = extract_pairs(
        source=cfg.parallel_parquet,
        out=cfg.corpus_csv,
        native_run=cfg.native_run,
        fields=fields,
        header=cfg.corpus_header,
        translit_suffix=cfg.translit_suffix,
        max_len=cfg.max_len,
    )
    click.echo(
        f"[{cfg.name}] {stats['pairs']:,} unique pairs -> {cfg.corpus_csv} "
        f"(aligned {stats['aligned']:,}, skipped {stats['mismatched']:,} mismatched, "
        f"{stats['ambiguous']:,} ambiguous)"
    )


@cli.command()
@click.option("--yes", is_flag=True, help="Skip the interactive sample-review gate.")
@click.option("--limit", default=None, type=int)
@click.pass_context
def all(ctx, yes, limit):
    """sample-gate -> run -> extract."""
    if not yes:
        ctx.invoke(sample)
        if not click.confirm("Proceed with the full run?", default=False):
            click.echo("Aborted at sample gate.")
            sys.exit(1)
    ctx.invoke(run, limit=limit)
    ctx.invoke(extract)


if __name__ == "__main__":
    cli()
