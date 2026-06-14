"""eroll -- reusable electoral-roll transliteration pipeline.

Turns Indian electoral rolls (native script) into high-quality Indic->English
training corpora and improved romanized rolls, using the ``indicate`` core library's
batch-mode LLM transliteration. Forward direction only (Indic -> English); rolls that
are already romanized English are out of scope.
"""

__version__ = "0.1.0"

from .states import STATES, StateConfig

__all__ = ["STATES", "StateConfig"]
