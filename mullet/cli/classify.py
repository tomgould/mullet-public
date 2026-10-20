"""mullet classify — print the likely intent class for a given prompt.

Debugging helper for the UserPromptSubmit hook chain: given the same
prompt a hook would see, print the classification the intent-gate layer
would assign plus a suggested next action. Output is stable and greppable
so the HUD can call this directly later.

Usage:
    python -m mullet.cli.classify "do a deep review of the router changes"
    python -m mullet.cli.classify --json "commit and push"
    echo "prompt" | python -m mullet.cli.classify -

Design note: trigger table is a superset of `skill_activator.py`'s keyword
map plus six intent classes (Trivial / Explicit / Exploratory / OpenEnded /
GitHubWork / Ambiguous).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class Classification:
    """Intent classification result for a prompt."""
    intent: str
    confidence: float
    matched_triggers: List[str]
    suggested_action: str
    skill_hints: List[str]


# Phrase → intent class. Order matters: first match wins on ties.
INTENT_TRIGGERS: list[tuple[str, str, float]] = [
    # GitHubWork — strongest signals first
    (r"github\.com/[\w\-./]+/(pull|issues)/\d+", "GitHubWork", 0.95),
    (r"\bPR\s*#?\d+\b", "GitHubWork", 0.90),
    (r"\b(look into|investigate).+(create|open)\s+(a\s+)?PR\b", "GitHubWork", 0.95),
    (r"\bcreate\s+(a\s+)?pull request\b", "GitHubWork", 0.90),
    # Explicit — concrete command
    (r"\b(commit|push|rebase|merge|cherry-pick)\b", "Explicit", 0.85),
    (r"\b(run|execute)\s+tests?\b", "Explicit", 0.85),
    (r"\breindex\b", "Explicit", 0.85),
    # OpenEnded — planning / review verbs
    (r"\b(plan|roadmap|design|architect)\b", "OpenEnded", 0.80),
    (r"\b(refactor|restructure|rewrite|clean up)\b", "OpenEnded", 0.80),
    (r"\b(improve|enhance|extend)\b", "OpenEnded", 0.75),
    (r"\b(analy[sz]e|evaluate|assess|compare|recommend)\b", "OpenEnded", 0.80),
    (r"\breview\b", "OpenEnded", 0.75),
    # Exploratory — find / how-does
    (r"\bhow\s+does\b", "Exploratory", 0.80),
    (r"\b(find|locate|search|where\s+is)\b", "Exploratory", 0.75),
    (r"\bwhat\s+(is|does|are)\b", "Exploratory", 0.70),
    # Ambiguous — vague single-word asks
    (r"^(help|fix it|do it|continue)\W*$", "Ambiguous", 0.85),
]

# Mullet-specific phrase → skill hint (superset of skill_activator.py).
SKILL_TRIGGERS: list[tuple[str, str]] = [
    (r"\bcommit(\s+and\s+push)?\b", "commit-review"),
    (r"\boverniht|overnight\s+loop\b", "loop-agreement"),
    (r"\b(benchmark|perf|profile)\b", "benchmark"),
    (r"\b(provider|routing|router)\b", "provider-routing"),
    (r"\b(hook|pretooluse|userpromptsubmit)\b", "hooks"),
    (r"\b(skill|SKILL\.md)\b", "skills"),
    (r"\btrawl|knowledge pool\b", "trawl"),
    (r"\b(delegate|kimi|gemini|haiku)\b", "delegation"),
]

ACTION_FOR_INTENT = {
    "GitHubWork": "Full cycle expected: investigate -> implement -> verify -> create PR. Delegate implementation via mullet_run.",
    "Explicit": "Execute directly (Opus, trivial op). Skip delegation.",
    "OpenEnded": "Delegate analysis/plan to Kimi via mullet_run. Opus synthesises and commits.",
    "Exploratory": "Use trawl_search_code first. Fire explore-style delegation if 2+ modules involved.",
    "Ambiguous": "Ask ONE clarifying question before acting.",
    "Trivial": "Direct tools only — no delegation needed.",
}


def classify(prompt: str) -> Classification:
    """Classify a prompt into an intent and suggest an action."""
    text = prompt.strip()
    if not text:
        return Classification(
            intent="Ambiguous",
            confidence=0.5,
            matched_triggers=[],
            suggested_action=ACTION_FOR_INTENT["Ambiguous"],
            skill_hints=[],
        )

    lower = text.lower()
    best_intent = "Trivial"
    best_confidence = 0.5
    matched: list[str] = []

    for pattern, intent, conf in INTENT_TRIGGERS:
        if re.search(pattern, lower, re.IGNORECASE):
            matched.append(pattern)
            if conf > best_confidence:
                best_intent = intent
                best_confidence = conf

    # Length heuristic: very long prompts default to OpenEnded if unmatched
    if best_intent == "Trivial" and len(text) > 400:
        best_intent = "OpenEnded"
        best_confidence = 0.65
        matched.append("length>400")

    skill_hints = [
        hint for pattern, hint in SKILL_TRIGGERS
        if re.search(pattern, lower, re.IGNORECASE)
    ]

    return Classification(
        intent=best_intent,
        confidence=round(best_confidence, 2),
        matched_triggers=matched,
        suggested_action=ACTION_FOR_INTENT[best_intent],
        skill_hints=skill_hints,
    )


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt == "-" or args.prompt is None:
        return sys.stdin.read()
    return args.prompt


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for mullet classify."""
    p = argparse.ArgumentParser(
        prog="mullet classify",
        description="Classify a prompt's intent (debug helper for hook chain).",
    )
    p.add_argument("prompt", nargs="?", help="Prompt text (or '-' for stdin)")
    p.add_argument("--json", action="store_true", dest="json_output",
                   help="Emit JSON instead of human-readable output")
    args = p.parse_args(argv)

    prompt = _read_prompt(args)
    result = classify(prompt)

    if args.json_output:
        print(json.dumps(asdict(result), indent=2))
        return 0

    print(f"intent     : {result.intent}")
    print(f"confidence : {result.confidence}")
    print(f"triggers   : {', '.join(result.matched_triggers) or '(none)'}")
    print(f"skill_hints: {', '.join(result.skill_hints) or '(none)'}")
    print(f"action     : {result.suggested_action}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
