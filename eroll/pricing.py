"""Batch-API cost estimation for the ``profile`` step.

Prices are **batch** rates (provider Batch APIs apply ~50% off standard) in USD per
1M tokens, as ``(input, output)``. These change frequently -- VERIFY CURRENT PRICING
before trusting an estimate. (Lives here for now; may move into ``indicate`` later.)

NOTE: GPT-4o is legacy as of 2026 (OpenAI's current line is GPT-5.x); fill in whatever
cheap OpenAI batch model you actually use.
"""

from __future__ import annotations

# model -> (input_$_per_mtok, output_$_per_mtok), already batch (50%-off) rates.
# OpenAI Batch API = 50% off standard. gpt-4o standard $2.50/$10 -> batch $1.25/$5;
# gpt-4o-mini standard $0.15/$0.60 -> batch $0.075/$0.30. VERIFY before relying on these.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Current cheap+capable OpenAI tier (2026); recommended over legacy gpt-4o.
    "gpt-5.4-nano": (0.10, 0.625),
    "gpt-5.4-mini": (0.375, 2.25),
    "gpt-4o": (1.25, 5.00),  # legacy (gpt-4o-2024-08-06)
    "gpt-4o-mini": (0.075, 0.30),
    "claude-haiku-4-5": (0.50, 2.50),
    "claude-sonnet-4-6": (1.50, 7.50),
    "claude-opus-4-8": (2.50, 12.50),
    # Gemini Batch API = 50% off standard. Keys match the litellm `gemini/...` model id.
    # gemini-2.5-flash standard $0.30/$2.50 -> batch $0.15/$1.25. VERIFY before relying.
    "gemini/gemini-2.5-flash": (0.15, 1.25),
    "gemini/gemini-2.5-flash-lite": (0.05, 0.20),
}


def estimate_cost(
    n_requests: int,
    avg_input_tokens: float,
    avg_output_tokens: float,
    model: str,
) -> float | None:
    """Rough USD estimate for ``n_requests`` batch requests, or None if unpriced."""
    price = MODEL_PRICING.get(model)
    if price is None:
        return None
    in_per_mtok, out_per_mtok = price
    in_cost = n_requests * avg_input_tokens / 1_000_000 * in_per_mtok
    out_cost = n_requests * avg_output_tokens / 1_000_000 * out_per_mtok
    return in_cost + out_cost
