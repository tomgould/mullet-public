"""Structured provider / delegation errors.

Today the model picker raises ``ValueError(...)`` with a human-readable
message when no candidate fits. That loses two distinctions callers
need to make programmatic decisions:

  1. Which provider / tier / tried candidates triggered the refusal.
  2. Whether a bigger sibling model exists (caller can escalate) or
     whether every candidate in the table is too small (caller must
     split the task or refuse).

``ModelContextExceeded`` wraps both so ``handle_mullet_run`` can surface
structured ``filtered_providers`` entries, and the swarm dispatcher can
route to the next tier on ``all_candidates_exhausted=False`` while
refusing + logging ERROR on ``True``.

It subclasses ``ValueError`` so every existing ``except ValueError:``
handler (router resolver, fallback-chain builder, CLI surface) continues
to catch it without a migration burst.
"""

from __future__ import annotations


class ModelContextExceeded(ValueError):
    """Raised when a prompt's token count exceeds a candidate model's max_context.

    Two variants via ``all_candidates_exhausted``:
      False  - escalation possible (a bigger sibling model exists in same
               provider/tier that a smarter picker could have chosen; the
               rejection came from a constraint filter, allowlist or
               tier cap — not from the static table running out).
      True   - no candidate can serve; caller should split the task or
               refuse.

    Subclasses ValueError for back-compat with ``except ValueError:`` catchers.

    Attributes:
        provider: provider alias (``kimi``, ``anthropic``, ``gemini``).
        tier: requested complexity tier (``STANDARD``, ``ADVANCED``, ...).
              ``None`` when the raise site doesn't track a tier.
        context_tokens: the estimated prompt size in tokens that couldn't fit.
        tried_candidates: model ids that were evaluated before the raise.
        max_candidate_context: largest ``max_context`` across tried candidates
              (or ``None`` if unknown — useful to report "max we could offer").
        all_candidates_exhausted: see variants above.
    """

    def __init__(
        self,
        *,
        provider: str,
        tier: str | None,
        context_tokens: int,
        tried_candidates: list[str],
        max_candidate_context: int | None = None,
        all_candidates_exhausted: bool = False,
    ):
        self.provider = provider
        self.tier = tier
        self.context_tokens = context_tokens
        self.tried_candidates = list(tried_candidates)
        self.max_candidate_context = max_candidate_context
        self.all_candidates_exhausted = all_candidates_exhausted
        detail = (
            f"provider={provider} tier={tier or 'any'} "
            f"prompt_tokens={context_tokens} "
            f"candidates tried={self.tried_candidates} "
            f"max context={max_candidate_context} "
            f"exhausted={all_candidates_exhausted}"
        )
        super().__init__(f"ModelContextExceeded - {detail}")


class ObedienceError(ValueError):
    """Raised when user-specified obedience constraints can't be satisfied.

    Contexts:
      - allowlist/constraint conflict (every provider in the allowlist
        was rejected by ConstraintRegistry).
      - no provider in the allowlist is available at runtime.
      - tier escalation forbidden by the allowlist (the requested tier
        cannot be served by any allowlisted provider).

    Subclass of ``ValueError`` for back-compat with every existing
    ``except ValueError:`` catcher (CLI surface, MCP handler, router
    resolver, fallback-chain builder). Callers wanting structured
    obedience-failure info can ``except ObedienceError`` to read
    ``allowed_providers``, ``filtered_providers`` and
    ``requested_tier`` directly instead of regex-parsing the message.

    Attributes:
        allowed_providers: list of provider aliases the user authorised
            (``[]`` when unset / unrestricted).
        filtered_providers: ``{alias: reason}`` for each allowlisted
            provider that was eliminated by the constraint registry
            (empty dict when filtering wasn't the cause).
        requested_tier: the requested complexity tier (``"STANDARD"``,
            ``"ADVANCED"``, ...) — ``None`` when the raise site doesn't
            track a tier.
    """

    def __init__(
        self,
        *,
        message: str,
        allowed_providers: list[str] | None = None,
        filtered_providers: dict[str, str] | None = None,
        requested_tier: str | None = None,
    ):
        self.allowed_providers = list(allowed_providers) if allowed_providers else []
        self.filtered_providers = dict(filtered_providers) if filtered_providers else {}
        self.requested_tier = requested_tier
        super().__init__(message)


__all__ = ["ModelContextExceeded", "ObedienceError"]
