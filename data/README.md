# Roll data

Roll inputs and pipeline artifacts live here (path overridable via `EROLL_DATA_DIR`).
Large files are **not** committed — host them on Dataverse or git-LFS, mirroring the
`indicate` repo's data convention.

## Inputs (place here or point `EROLL_DATA_DIR` at them)

| File | State | Notes |
|---|---|---|
| `assam_all_clean+t13n.csv.gz` | Assam | Native Assamese columns + weak libindic `_t13n` columns. The live forward target. Relocated out of the `indicate` repo. |
| `andhra.7z` | Andhra | **Excluded / not required here** — already romanized English (no native script), so out of scope for a forward Indic→English corpus. Keep wherever you store raw Andhra rolls; the pipeline never reads it. |
| `punjab_transliteration_subset.parquet` | Punjab | GPT-4o parallel parquet (`<field>` + `<field>_transliterated`). Used only by the `extract_pairs` regression test. Currently lives in `indicate/data/`. |

## Outputs (generated; gitignored)

| File | Produced by |
|---|---|
| `<state>_tokens.jsonl` (+ `.batchstate.json`) | batch checkpoint/state |
| `<state>_parallel.parquet` | `run` (join-back) |
| `<language>.csv.gz` (e.g. `assamese.csv.gz`) | `extract` — the corpus handed to `indicate` for training |
| `eval/<state>_sample.tsv` | `sample` — human-review gate |
