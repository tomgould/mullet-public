"""Delegation safety D4 — dispatch-time context-fit assertion.

Defence-in-depth gate wired at each provider's pre-HTTP boundary
(``openai_compat._call_with_retry``, ``anthropic._call_with_retry``,
``gemini._call_with_retry``). Upstream layers (D1 measurement, D2
candidate filter, D3 structured error) are supposed to keep
over-sized prompts away from under-sized models, but if any of those
layers is bypassed — a direct ``Provider(model=...)`` instantiation,
a learned constraint mutating ``max_context`` between pick and call,
a caller patching around the router — D4 is the last hard stop
before we hand a request to an endpoint that will ``400`` on us.

Design notes:
  * Uses the SAME safety margin as D2 (``CONTEXT_SAFETY_MARGIN`` from
    ``model_registry``) — equivalent to the ``max_ctx * 0.91`` inverse
    check specified in the cycle plan, but re-expressing the rule as
    ``tokens * 1.1 > max_ctx`` avoids drifting from D2's wording.
  * SOFT-fails (returns without raising) on estimator error or unknown
    model. D4 is a safety net, not a gate — blocking on an internal
    estimator crash would be worse than passing through an oversized
    prompt to the real model, which would fail loudly anyway.
  * Raises ``ModelContextExceeded(all_candidates_exhausted=True)``
    because by the time D4 runs we are already committed to a
    specific provider + model. There is no fallback chain left.
"""
from __future__ import annotations

import logging
from typing import Any

from mullet.core.providers.constraints import get_constraint_registry
from mullet.core.providers.errors import ModelContextExceeded
from mullet.core.providers.model_registry import CONTEXT_SAFETY_MARGIN
from mullet.core.providers.tokenize import estimate_tokens

logger = logging.getLogger(__name__)


def assert_context_fits(
    provider_name: str,
    model: str,
    messages: list[dict] | Any,
) -> None:
    """Raise ``ModelContextExceeded`` if the composed payload would overflow.

    Applied at the dispatch-time boundary, after all router/picker/
    constraint-shaping logic has run and immediately before the real
    HTTP call.

    Soft-fails (returns) on estimator or constraint-registry errors —
    D4 is a safety net, not a gate. Hard-raises only when the provider's
    own table says the prompt cannot fit.

    Args:
        provider_name: short alias (``openai``, ``anthropic``, ``gemini``)
            — used only to populate the raised error.
        model: the model id the request is about to be sent to.
        messages: the final composed messages list (or any JSON-serialisable
            payload accepted by ``estimate_tokens``).

    Raises:
        ModelContextExceeded: with ``all_candidates_exhausted=True``,
        ``context_tokens`` set to the estimate, ``tried_candidates`` set
        to ``[model]``, and ``max_candidate_context`` set to the model's
        ``max_context`` — ONLY when the registry actually knows this
        model AND the prompt would overflow by more than the safety
        margin.
    """
    try:
        tokens = estimate_tokens(messages)
    except Exception as exc:
        logger.debug(
            "context_gate: estimate_tokens failed (%s); skipping check", exc
        )
        return

    try:
        max_ctx = get_constraint_registry().get(model).max_context
    except Exception as exc:
        logger.debug(
            "context_gate: constraint lookup for %s failed (%s); skipping",
            model, exc,
        )
        return

    if not max_ctx:
        return

    # Use the SAME inequality shape as the D2 filter (_filter_candidates_by_context).
    # Reject when max_ctx < tokens * CONTEXT_SAFETY_MARGIN — equivalent to the
    # ``max_ctx * 0.91 >= tokens`` formulation in the cycle plan without drift.
    if tokens * CONTEXT_SAFETY_MARGIN > max_ctx:
        logger.warning(
            "context_gate: BLOCK provider=%s model=%s tokens=%d max_ctx=%d "
            "(overflow margin=%.2f)",
            provider_name, model, tokens, max_ctx, CONTEXT_SAFETY_MARGIN,
        )
        raise ModelContextExceeded(
            provider=provider_name,
            tier=None,
            context_tokens=tokens,
            tried_candidates=[model],
            max_candidate_context=max_ctx,
            all_candidates_exhausted=True,
        )


__all__ = ["assert_context_fits"]
