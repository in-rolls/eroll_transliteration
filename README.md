# eroll_transliteration

Reusable pipeline for transliterating Indian **electoral rolls** from native script to
English, building high-quality `(native, english)` training corpora. The LLM step runs in
**batch mode** through the [`indicate`](https://github.com/in-rolls/indicate) core library;
this repo holds the roll-specific glue (per-state config, streaming IO, net-new harvest,
corpus extraction, and a cross-file id-join for rolls that ship an official English version).

**Forward direction only** (Indic → English). Rolls that are *only* romanized English (e.g.
Andhra) are out of scope as a native source — but an English roll that has a native-script
twin is useful as the English side of a join (see *Parallel id-join* below).

## Install

```bash
uv sync                # uses ../indicate as an editable path dependency (see pyproject)
```

Keys are read from the environment at runtime (never committed):

```bash
export GEMINI_API_KEY=...           # Gemini batch (recommended); or OPENAI_API_KEY, etc.
export DATAVERSE_API_TOKEN=...      # only to download rolls from Dataverse
export EROLL_DATA_DIR=/path/to/rolls   # defaults to ./data
```

## Get the rolls

Rolls live on a Harvard Dataverse dataset; pull them with the built-in downloader:

```bash
python -m eroll.dataverse list --doi doi:10.7910/DVN/MUEGDT
python -m eroll.dataverse download --doi doi:10.7910/DVN/MUEGDT --pattern 'guj_all_clean*'
```

Only the `data/<language>.csv.gz` corpora are committed. The large raw rolls, batch
checkpoints (`<state>_tokens.tar.gz`), parallel parquets, and key files (`.gemini_key`,
`.dataverse_token`) are gitignored — delete a roll once its harvest is done; it's
re-downloadable. Andhra is excluded as a *source* (already romanized English, no native side).

## Harvest a corpus (LLM)

`harvest` collects unique native tokens, keeps only those **not already** in the language's
corpus, transliterates just those via batch, and **appends** the new pairs to the shared
`data/<language>.csv.gz` (kept sorted + deduped). It never builds a parallel parquet, so you
pay the LLM only for net-new tokens. Multiple states can feed one corpus (e.g. `assam`,
`west_bengal`, `tripura` all → `bengali.csv.gz`).

```bash
# unique-token counts + batch-cost estimate (no API calls)
python -m eroll.transliterate_rolls --state west_bengal profile

# (optional) sample-review 300 random transliterations before spending
python -m eroll.transliterate_rolls --state west_bengal \
    --provider gemini --model gemini/gemini-2.5-flash sample --sample-n 300

# net-new harvest -> append to data/bengali.csv.gz (resumable batch)
python -m eroll.transliterate_rolls --state west_bengal \
    --provider gemini --model gemini/gemini-2.5-flash harvest

# re-append from a resolved checkpoint without re-calling the API (crash recovery)
python -m eroll.transliterate_rolls --state west_bengal merge
```

The older `run` → `extract` (→ parallel parquet → modal-dedup) path remains for the original
GPT-4o-built states (`assam`, `punjab`) and is regression-tested byte-for-byte.

## Parallel id-join (no LLM)

When a state ships the *same* roll in both native script and official English (e.g. Kerala:
Malayalam yearly files + an English `all_clean`), join them on the elector `id` to read pairs
straight from the official roll — free, authoritative spellings. The native side may be a glob
spanning multiple yearly files (counts pool across years for the corroboration filter).

```bash
python -m eroll.parallel_pairs \
    --native 'data/kerala_201[2-6].csv.gz' --english data/kerala_all_clean.csv.gz \
    --script malayalam --lang malayalam \
    --fields elector_name,father_or_husband_name --min-len 3 --min-count 2
```

This is noisier than the LLM (the English file is consolidated across years, so an `id` can
drift); `--min-len`/`--min-count` filter OCR fragments and uncorroborated pairs.

## Corpora

| Corpus | Built from |
|---|---|
| `bengali.csv.gz` | Assam + West Bengal + Tripura |
| `gujarati.csv.gz` | Gujarat + Daman + Dadra |
| `malayalam.csv.gz` | Kerala (parallel id-join) |
| `kannada.csv.gz` | Karnataka |
| `telugu.csv.gz` | Telugu-state roll |
| `punjabi.csv.gz` | Punjab (legacy GPT-4o) |

## Adding a state / language

Add one `StateConfig` to `eroll/states.py` (name, language, native unicode range, input
filename, corpus header; optional `columns` when the schema differs). Same `language` →
appends to the same corpus. Multi-file or archive sources are consolidated to one `.csv.gz`
first. Everything else is state-agnostic.

## Layout

| Module | Role |
|---|---|
| `eroll/states.py` | Per-state config + `STATES` registry |
| `eroll/pipeline.py` | Unique-token extraction + word-map join-back |
| `eroll/extract_pairs.py` | Aligned pair extraction (modal dedup); corpus load/merge helpers |
| `eroll/transliterate_rolls.py` | Orchestrator CLI (profile/sample/harvest/merge/run/extract) |
| `eroll/parallel_pairs.py` | Cross-file id-join → pairs (duckdb), for native+English rolls |
| `eroll/dataverse.py` | Download rolls from a Harvard Dataverse dataset |
| `eroll/pricing.py` | Batch-API cost table for the profile step |
