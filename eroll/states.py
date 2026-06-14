"""Per-state roll configuration.

All states are **forward** (native script -> English). A new state is one
:class:`StateConfig` entry; the orchestrator and pipeline are state-agnostic.

Data paths resolve under ``EROLL_DATA_DIR`` (env) or ``<repo>/data``. Roll inputs and
the produced parquet/corpus live there; only the final corpus crosses back to
``indicate`` for model training.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The 11 name/place columns present (identically ordered) across state roll schemas.
# Order is significant: it must match ``training/extract_punjabi.py`` SOURCE_FIELDS so
# the generalized extractor reproduces the committed Punjabi corpus byte-for-byte.
NAME_PLACE_COLUMNS: tuple[str, ...] = (
    "elector_name",
    "father_or_husband_name",
    "ac_name",
    "parl_constituency",
    "main_town",
    "police_station",
    "mandal",
    "revenue_division",
    "district",
    "polling_station_name",
    "polling_station_address",
)

# The Daman & Dadra union-territory rolls use a different person-name schema (separate
# name/father/husband/mother columns instead of elector_name/father_or_husband_name); the
# place columns match. Same scrape, different vintage.
GUJ_UT_COLUMNS: tuple[str, ...] = (
    "name",
    "father's name",
    "husband's name",
    "mother's name",
    "ac_name",
    "parl_constituency",
    "main_town",
    "police_station",
    "mandal",
    "revenue_division",
    "district",
    "polling_station_name",
    "polling_station_address",
)

# The Kerala (Malayalam) yearly rolls carry the standard 11 columns plus three extra
# Malayalam name fields (husband / mother_name / household_name) worth mining.
KERALA_COLUMNS: tuple[str, ...] = NAME_PLACE_COLUMNS + (
    "husband",
    "mother_name",
    "household_name",
)


def data_dir() -> Path:
    """Resolve the data directory (``EROLL_DATA_DIR`` or ``<repo>/data``)."""
    env = os.environ.get("EROLL_DATA_DIR")
    return Path(env) if env else REPO_ROOT / "data"


@dataclass(frozen=True)
class StateConfig:
    """Configuration for transliterating one state's electoral roll (forward)."""

    name: str
    language: str  # IndicLLMTransliterator key, e.g. "assamese"
    native_range: tuple[str, str]  # (lo, hi) unicode chars for the native script block
    input_glob: str  # filename under the data dir (``.csv.gz`` or archive member)
    corpus_header: tuple[str, str]  # CSV header, e.g. ("assamese", "english")
    columns: tuple[str, ...] = NAME_PLACE_COLUMNS
    archive: str | None = None  # e.g. "andhra.7z"; None for a plain csv(.gz)
    translit_suffix: str = "_t13n_llm"  # join_back output / extract romanized partner
    csv_chunksize: int = 200_000
    max_len: int = 40
    parallel_parquet_name: str | None = None
    corpus_csv_name: str | None = None

    @property
    def native_run(self) -> re.Pattern[str]:
        """Regex matching a maximal run of native-script characters."""
        lo, hi = self.native_range
        return re.compile(f"[{lo}-{hi}]+")

    @property
    def input_path(self) -> Path:
        return data_dir() / (self.archive or self.input_glob)

    @property
    def parallel_parquet(self) -> Path:
        return data_dir() / (
            self.parallel_parquet_name or f"{self.name}_parallel.parquet"
        )

    @property
    def corpus_csv(self) -> Path:
        return data_dir() / (self.corpus_csv_name or f"{self.language}.csv.gz")


STATES: dict[str, StateConfig] = {
    # Live target: Assam rolls are in Bengali script (Eastern-Nagari, U+0980-U+09FF),
    # with weak libindic `_t13n` columns to beat. LLM (GPT-4o) produces better English.
    "assam": StateConfig(
        name="assam",
        language="bengali",
        native_range=("ঀ", "৿"),
        input_glob="assam_all_clean+t13n.csv.gz",
        corpus_header=("bengali", "english"),
        translit_suffix="_t13n_llm",
    ),
    # West Bengal rolls are Bengali script too -- same corpus as assam. We only HARVEST
    # net-new tokens (absent from bengali.csv.gz) and append them, so no parallel parquet
    # is built; corpus_csv intentionally collides with assam's (shared bengali.csv.gz).
    # Unlike assam, the polling_station_* columns here carry Bengali script (mined too).
    "west_bengal": StateConfig(
        name="west_bengal",
        language="bengali",
        native_range=("ঀ", "৿"),
        input_glob="wb_all.csv.gz",
        corpus_header=("bengali", "english"),
        translit_suffix="_t13n_llm",
    ),
    # Gujarati (U+0A80-U+0AFF) corpus is harvested from three sources that all append net-new
    # tokens into the shared gujarati.csv.gz: Gujarat state (standard schema) plus the Daman
    # and Dadra UT rolls (GUJ_UT_COLUMNS schema, pre-consolidated from multi-file sources).
    "gujarat": StateConfig(
        name="gujarat",
        language="gujarati",
        native_range=("઀", "૿"),
        input_glob="guj_all_clean+t13n.csv.gz",
        corpus_header=("gujarati", "english"),
        translit_suffix="_t13n_llm",
    ),
    "daman": StateConfig(
        name="daman",
        language="gujarati",
        native_range=("઀", "૿"),
        input_glob="daman_guj_all.csv.gz",  # consolidated 2015 + 2017
        corpus_header=("gujarati", "english"),
        columns=GUJ_UT_COLUMNS,
        translit_suffix="_t13n_llm",
    ),
    "dadra": StateConfig(
        name="dadra",
        language="gujarati",
        native_range=("઀", "૿"),
        input_glob="dadra_guj_all.csv.gz",  # consolidated from the 264-file 2017 archive
        corpus_header=("gujarati", "english"),
        columns=GUJ_UT_COLUMNS,
        translit_suffix="_t13n_llm",
    ),
    # Karnataka (Kannada, U+0C80-U+0CFF): standard schema + weak libindic _t13n columns to
    # beat. LLM harvest -> kannada.csv.gz (no official English roll exists, unlike Kerala).
    "karnataka": StateConfig(
        name="karnataka",
        language="kannada",
        native_range=("ಀ", "೿"),
        input_glob="kar_all_clean+t13n.csv.gz",
        corpus_header=("kannada", "english"),
        translit_suffix="_t13n_llm",
    ),
    # Telugu (U+0C00-U+0C7F): combined Telugu-state roll (native script, not the romanized
    # Andhra dump). LLM harvest -> telugu.csv.gz.
    "telugu": StateConfig(
        name="telugu",
        language="telugu",
        native_range=("ఀ", "౿"),
        input_glob="tel_all.csv.gz",
        corpus_header=("telugu", "english"),
        translit_suffix="_t13n_llm",
    ),
    # Tripura rolls are in Bengali script -> append net-new to the shared bengali.csv.gz
    # (same pattern as west_bengal). LLM harvest.
    "tripura": StateConfig(
        name="tripura",
        language="bengali",
        native_range=("ঀ", "৿"),
        input_glob="tripura_all_clean+t13n.csv.gz",
        corpus_header=("bengali", "english"),
        translit_suffix="_t13n_llm",
    ),
    # Punjab (Gurmukhi, U+0A00-U+0A7F): forward, already built via GPT-4o. Kept so the
    # generalized extractor can be regression-tested against the committed corpus --
    # its parallel parquet uses the `_transliterated` suffix from the original notebook.
    "punjab": StateConfig(
        name="punjab",
        language="punjabi",
        native_range=("਀", "੿"),
        input_glob="punjab_all_clean+t13n.csv.gz",
        corpus_header=("punjabi", "english"),
        translit_suffix="_transliterated",
        parallel_parquet_name="punjab_transliteration_subset.parquet",
        corpus_csv_name="punjabi.csv.gz",
    ),
    # Andhra is intentionally EXCLUDED: the roll is already romanized English with no
    # native-script column, so a forward (Indic->English) corpus has no source side.
    # The 7z is parked, not processed. (Do not add a live entry.)
}
