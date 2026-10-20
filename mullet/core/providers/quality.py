"""Per-model quality scores for the value-based routing picker (routing-v3).

Domain sub-scores (0-100 each), NOT a single flattened number -- mixing
SWE-Bench pass-rates with composite indices like Artificial Analysis's
Intelligence Index on one scale would misroute non-coding tasks. See
docs/design/routing-v3-value-picker.md Phase 0 for the full rationale.

Scoring method per domain:
- coding: average of SWE-Bench Pro % and Terminal-Bench 2.x % where both
  are available from independent/vendor sources (used directly as 0-100
  since both are already percentages); single-source when only one exists.
- reasoning: dedicated reasoning benchmarks (Humanity's Last Exam, GPQA,
  AIME) where available; otherwise estimated from the coding score with
  an explicit discount, confidence="estimated".
- general: Artificial Analysis Intelligence Index score used directly
  (already a cross-model composite on a comparable ~0-100 scale) where
  available; otherwise estimated.

confidence:
- "direct": score computed from a real, cited benchmark for THIS exact model.
- "inherited": score copied from a named sibling/parent model, with an
  explicit variant penalty applied (see VARIANT_PENALTY below) and
  inherited_from set.
- "estimated": no direct or inheritable data existed; a conservative
  number was chosen and must be treated as low-confidence -- exclude from
  being the SOLE candidate clearing a CRITICAL-tier floor (enforced by the
  Phase 2 picker, not here).

Research date: 2026-07-05.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Penalty applied when a variant's score is inherited from its parent/base
# model rather than measured directly -- these variants explicitly trade
# quality for something else (speed, stability, cost) and must not silently
# inherit the parent's score at face value.
VARIANT_PENALTY: dict[str, int] = {
    "preview": -5,
    "fast": -3,
    "highspeed": -3,
    "lite": -8,
}


@dataclass(frozen=True)
class QualityScore:
    coding: int
    reasoning: int
    general: int
    confidence: Literal["direct", "inherited", "estimated"]
    source: str
    inherited_from: str | None = None
    is_reasoning_tuned: bool = False


MODEL_QUALITY: dict[str, QualityScore] = {
    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------
    "claude-opus-4-8": QualityScore(
        coding=72,
        reasoning=68,
        general=55,
        confidence="direct",
        source=(
            "SWE-Bench Pro 69.2%/Terminal-Bench 2.1 74.6%, "
            "Anthropic launch materials + TestingCatalog, 2026-05-28 release"
        ),
    ),
    "claude-opus-4-7": QualityScore(
        coding=65,
        reasoning=62,
        general=50,
        confidence="direct",
        source=(
            "SWE-Bench Pro 64.3%/Terminal-Bench 2.1 66.1%, "
            "same comparison set as Opus 4.8 launch coverage"
        ),
    ),
    "claude-opus-4-6": QualityScore(
        coding=60,
        reasoning=58,
        general=46,
        confidence="estimated",
        source=(
            "SWE-Bench Verified ~80.6% inferred via DeepSeek V4-Pro-Max comparison "
            "(codersera.com/morphllm.com coverage); no direct SWE-Bench Pro/"
            "Terminal-Bench figure found -- needs a real citation in a future refresh"
        ),
    ),
    "claude-sonnet-5": QualityScore(
        coding=71,
        reasoning=57,
        general=52,
        confidence="direct",
        source=(
            "SWE-Bench Pro 63.2%, Terminal-Bench 2.1 80.4% (same-harness comparison "
            "vs Sonnet 4.6/Opus 4.8), HLE 57.4% w/tools -- MarkTechPost 2026-06-30 "
            "Sonnet 5 vs Sonnet 4.6 vs Opus 4.8 comparison + Anthropic launch materials. "
            "NOTE: this is the cross-provider anchor model per Phase 0 design -- "
            "other providers' 'general' estimates are loosely calibrated against this."
        ),
    ),
    "claude-sonnet-4-6": QualityScore(
        coding=54,
        reasoning=45,
        general=40,
        confidence="direct",
        source=(
            "SWE-Bench Pro 58.1%, Terminal-Bench 2.1 67.0% -- "
            "MarkTechPost 2026-06-30 same-harness comparison"
        ),
    ),
    "claude-haiku-4-5-20251001": QualityScore(
        coding=35,
        reasoning=30,
        general=25,
        confidence="estimated",
        source=(
            "No direct benchmark found in 2026-07-05 research pass; positioned as "
            "the bottom of the Anthropic family by tier convention -- "
            "needs a real citation in a future refresh"
        ),
    ),
    # ------------------------------------------------------------------
    # Kimi
    # ------------------------------------------------------------------
    "kimi-k2.7-code": QualityScore(
        coding=35,
        reasoning=35,
        general=35,
        confidence="estimated",
        is_reasoning_tuned=True,
        source=(
            "CONFLICTING DATA: vals.ai (independent) reported near-0% on "
            "SWE-bench/Terminal-Bench 2.1, likely a tool-calling format "
            "incompatibility rather than true capability (Mullet dispatches "
            "Kimi successfully for real work). Moonshot's own materials "
            "claim +21.8% on their proprietary Kimi Code Bench v2 over K2.6 "
            "(62.0 vs 50.9) but never submitted to standard public benchmarks. "
            "One secondary source (Flowtivity) claims 60.4% SWE-bench Verified, "
            "unverified and conflicting with vals.ai. VentureBeat 2026-06-12 "
            "coverage flags all three data points as unreliable. Given this level "
            "of conflict, treat this score as a rough placeholder only -- do NOT "
            "let it be the sole candidate for a CRITICAL floor until real "
            "independent numbers exist."
        ),
    ),
    "kimi-k2.7-code-highspeed": QualityScore(
        coding=32,
        reasoning=32,
        general=32,
        confidence="inherited",
        inherited_from="kimi-k2.7-code",
        source=(
            "Inherited from kimi-k2.7-code (itself low-confidence, see that entry) "
            "with highspeed variant penalty -- speed comes from shorter reasoning "
            "traces per vendor description"
        ),
    ),
    "kimi-k2.6": QualityScore(
        coding=30,
        reasoning=30,
        general=30,
        confidence="estimated",
        source=(
            "Proprietary Kimi Code Bench v2 baseline score 50.9 (pre-K2.7-Code release) "
            "used as a rough directional anchor only, no independent public benchmark found"
        ),
    ),
    "kimi-k2.5": QualityScore(
        coding=25,
        reasoning=25,
        general=25,
        confidence="estimated",
        source=(
            "No direct benchmark found; positioned below k2.6 by version order only -- "
            "needs a real citation in a future refresh"
        ),
    ),
    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------
    "gemini-3.1-pro": QualityScore(
        coding=67,
        reasoning=55,
        general=50,
        confidence="direct",
        source=(
            "SWE-Bench Verified ~80.6% (tied w/ DeepSeek V4-Pro-Max), "
            "SWE-Bench Pro 54.2% (vs GPT-5.5's 58.6%), LiveCodeBench 91.7, "
            "Codeforces 3052 -- cross-referenced coverage of the GPT-5.5/"
            "Opus-4.8 launch comparisons, 2026"
        ),
    ),
    "gemini-2.5-pro": QualityScore(
        coding=40,
        reasoning=30,
        general=26,
        confidence="direct",
        source=(
            "Artificial Analysis Intelligence Index 26 "
            "(artificialanalysis.ai/models/gemini-2-5-pro); no SWE-Bench/"
            "Terminal-Bench figure found directly for coding/reasoning, estimated "
            "from general score"
        ),
    ),
    "gemini-3.5-flash": QualityScore(
        coding=45,
        reasoning=45,
        general=50,
        confidence="direct",
        source=(
            "Artificial Analysis Intelligence Index 50 (high effort) -- "
            "artificialanalysis.ai/models/gemini-3-5-flash, 2026-05-19 release"
        ),
    ),
    "gemini-2.5-flash": QualityScore(
        coding=20,
        reasoning=18,
        general=14,
        confidence="direct",
        source=(
            "Artificial Analysis Intelligence Index 14 (non-reasoning) -- "
            "artificialanalysis.ai/models/gemini-2-5-flash"
        ),
    ),
    "gemini-2.5-flash-lite": QualityScore(
        coding=12,
        reasoning=10,
        general=7,
        confidence="inherited",
        inherited_from="gemini-2.5-flash",
        source=(
            "Artificial Analysis Intelligence Index 7 (non-reasoning) direct for general; "
            "coding/reasoning inherited from gemini-2.5-flash with lite penalty -- "
            "artificialanalysis.ai/models/gemini-2-5-flash-lite"
        ),
    ),
    # ------------------------------------------------------------------
    # OpenRouter
    # ------------------------------------------------------------------
    "deepseek/deepseek-v4-pro": QualityScore(
        coding=62,
        reasoning=55,
        general=52,
        confidence="direct",
        is_reasoning_tuned=True,
        source=(
            "Terminal-Bench 2.0 67.9%, SWE-Bench Pro 55.4%, SWE-Bench Verified 80.6% "
            "(Pro-Max variant), AA Index 52, LiveCodeBench 93.5, Codeforces 3206 -- "
            "DeepSeek's own April 2026 V4 launch materials, cross-referenced via "
            "codersera/benchlm/morphllm/artificialanalysis.ai coverage. CAVEAT: one "
            "analyst (Macaron survey) flagged all V4 numbers as "
            "vendor-internal-claim-only pending third-party reproduction -- "
            "treat as direct-but-unverified. Replaced deepseek/deepseek-r1 as the "
            "CRITICAL-tier reasoning-tuned default (R1 removed 2026-07-12, not fit "
            "for purpose; v4-pro has higher direct-confidence coding/reasoning scores)."
        ),
    ),
    "deepseek/deepseek-v4-flash": QualityScore(
        coding=59,
        reasoning=50,
        general=47,
        confidence="direct",
        source=(
            "SWE-Bench Verified 79.0%, AA Index 47 (artificialanalysis.ai lists 47 "
            "as rank #10/85; a separate page for the 'Reasoning, Max Effort' variant "
            "shows 40 -- likely different Index snapshot dates, using 47 as primary) -- "
            "same DeepSeek V4 April 2026 launch materials as v4-pro"
        ),
    ),
    "qwen/qwen3.7-max": QualityScore(
        coding=57,
        reasoning=55,
        general=56,
        confidence="direct",
        source=(
            "Artificial Analysis Intelligence Index 56.6 -- MarkTechPost coverage of "
            "Alibaba's 2026-05-20/21 Qwen3.7 launch at the Alibaba Cloud Summit; "
            "vendor also claims Terminal-Bench 2.0/SWE-Bench Pro/MCP-Atlas wins over "
            "Claude Opus 4.6 Max but no specific percentage was found in this research pass"
        ),
    ),
    "qwen/qwen3-coder": QualityScore(
        coding=70,
        reasoning=45,
        general=45,
        confidence="direct",
        source=(
            "SWE-Bench Verified 69.6% at release, widely cited as beating Claude/GPT-4 "
            "on this benchmark at the time -- this is the original 480B Qwen3-Coder, "
            "not Qwen3-Coder-Next"
        ),
    ),
    "qwen/qwen3-coder:free": QualityScore(
        coding=70,
        reasoning=45,
        general=45,
        confidence="direct",
        source=(
            "Same underlying model/weights as qwen/qwen3-coder (SWE-Bench Verified "
            "69.6% at release), reached via OpenRouter's free-tier `:free` slot instead "
            "-- identical benchmarks, see that entry for citations"
        ),
    ),
    "tencent/hy3:free": QualityScore(
        coding=63,
        reasoning=50,
        general=52,
        confidence="estimated",
        source=(
            "Tencent's own July 2026 Hy3 (295B MoE, 21B active) release materials: "
            "SWE-Bench Verified 78.0, SWE-Bench Pro 57.9, Terminal-Bench 2.1 71.7 -- "
            "described as trailing Claude Opus 4.8/GPT-5.5 but competitive with/ahead "
            "of GLM-5.2, DeepSeek-V4-Pro, Qwen3.7-Max on several benchmarks. Vendor-only "
            "numbers, independent third-party reproduction still pending as of this "
            "research pass (2026-07-10) -- same caveat class as kimi-k2.7-code, do NOT "
            "let this be the sole candidate for a tier's quality floor until "
            "independently verified. Not flagged is_reasoning_tuned despite the model's "
            "configurable chain-of-thought modes -- no independent confirmation it "
            "belongs in the same verified-reasoning class as deepseek/deepseek-v4-pro."
        ),
    ),
    "anthropic/claude-opus-4.8": QualityScore(
        coding=72,
        reasoning=68,
        general=55,
        confidence="direct",
        source=(
            "Same underlying model as claude-opus-4-8 (direct Anthropic API), "
            "reached via OpenRouter's anthropic/ proxy naming instead -- identical "
            "weights/benchmarks, see that entry for citations"
        ),
    ),
    "anthropic/claude-sonnet-5": QualityScore(
        coding=71,
        reasoning=57,
        general=52,
        confidence="direct",
        source=(
            "Same underlying model as claude-sonnet-5 (direct Anthropic API), "
            "reached via OpenRouter's anthropic/ proxy naming instead -- identical "
            "weights/benchmarks, see that entry for citations"
        ),
    ),
    "openai/gpt-5": QualityScore(
        coding=45,
        reasoning=45,
        general=40,
        confidence="estimated",
        source=(
            "No direct benchmark found for the specific 'gpt-5' (non-.5) release in "
            "this research pass -- OpenAI's coverage focuses on the newer GPT-5.5/5.6 "
            "releases. This entry needs a real citation in a future refresh, or "
            "routing_matrix.py's premium reasoning branch (which still references plain "
            "'openai/gpt-5') should be reconsidered for a version bump in a separate "
            "cycle -- out of scope for this one."
        ),
    ),
}
