# eroll_transliteration

Reusable pipeline for transliterating Indian **electoral rolls** from native script to
English, building high-quality `(native, english)` training corpora and improved
romanized rolls. The LLM step runs in **batch mode** through the
[`indicate`](https://github.com/in-rolls/indicate) core library; this repo holds the
roll-specific glue (per-state config, streaming IO, join-back, corpus extraction).

**Forward direction only** (Indic → English). Rolls that are already romanized English
(e.g. Andhra) are out of scope — there is no native source side to learn from.

## Install

```bash
uv sync                # uses ../indicate as an editable path dependency (see pyproject)
```

Set the chosen provider's API key before any LLM step (batch via litellm — OpenAI by
default; Anthropic is not a batch provider in litellm, use Bedrock for Claude):

```bash
export OPENAI_API_KEY=...        # or configure another litellm batch provider
export EROLL_DATA_DIR=/path/to/rolls   # defaults to ./data
```

## Usage

```bash
# 1. Profile: unique-token counts + batch-cost estimate (no API calls)
python -m eroll.transliterate_rolls --state assam profile

# 2. Sample-review 300 random tokens before committing to a full run
python -m eroll.transliterate_rolls --state assam sample --sample-n 300

# 3. Full run: batch-transliterate all uniques -> parallel parquet (resumable)
python -m eroll.transliterate_rolls --state assam run

# 4. Extract the (native, english) corpus -> data/assamese.csv.gz
python -m eroll.transliterate_rolls --state assam extract

# Or gated end-to-end:
python -m eroll.transliterate_rolls --state assam all
```

The produced corpus (e.g. `data/assamese.csv.gz`) is handed to `indicate` for model
training; only the trained model ships in the `indicate` wheel.

## Adding a state

Add one `StateConfig` entry to `eroll/states.py` (name, language, native unicode range,
input filename, corpus header). Everything else is state-agnostic.

## Layout

| Module | Role |
|---|---|
| `eroll/states.py` | Per-state config + `STATES` registry |
| `eroll/pipeline.py` | Unique-token extraction + word-map join-back |
| `eroll/extract_pairs.py` | Aligned `(native, english)` pair extraction (modal dedup) |
| `eroll/transliterate_rolls.py` | Orchestrator CLI (profile/sample/run/extract/all) |
| `eroll/pricing.py` | Batch-API cost table for the profile step |
