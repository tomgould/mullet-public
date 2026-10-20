"""Task complexity analysis — ported from OMC's routeTask().

Extracts lexical and structural signals from a task prompt,
scores them, and determines a complexity tier (LOW/STANDARD/ADVANCED/CRITICAL).
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    """Complexity tier enum for provider routing."""
    LOW = "LOW"
    STANDARD = "STANDARD"
    ADVANCED = "ADVANCED"
    CRITICAL = "CRITICAL"
    # Backward compat aliases
    MEDIUM = "STANDARD"
    HIGH = "ADVANCED"


PROVIDER_TIER_CAPABILITIES: dict[str, set[str]] = {
    "local":          {"LOW", "STANDARD", "ADVANCED", "CRITICAL"},  # Phase E: free fallback at every tier
    "kimi":           {"LOW", "STANDARD", "ADVANCED", "CRITICAL"},
    "gemini":         {"LOW", "STANDARD", "ADVANCED"},
    "cheap_cloud":    {"LOW", "STANDARD"},
    "haiku":          {"LOW", "STANDARD"},
    "sonnet":         {"ADVANCED", "CRITICAL"},
    "opus":           {"CRITICAL"},
    "opus-4-7":       {"CRITICAL"},
    "opus-4-8":       {"CRITICAL"},   # Phase B M1: claude-only allows opus-4-8
    "premium_cloud":  {"ADVANCED", "CRITICAL"},
    "openrouter":     {"LOW", "STANDARD", "ADVANCED"},  # Phase B M1: cheap pay-per-token cloud
}


def tiers_for_providers(allowed: set[str] | frozenset[str]) -> list[str]:
    """Return union of tiers the given providers can serve, ordered LOW->CRITICAL."""
    order = ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]
    covered: set[str] = set()
    for name in allowed:
        covered |= PROVIDER_TIER_CAPABILITIES.get(name, set())
    return [t for t in order if t in covered]


@dataclass
class LexicalSignals:
    """Lexical signals extracted from a task prompt."""
    word_count: int = 0
    file_path_count: int = 0
    code_block_count: int = 0
    has_architecture_keywords: bool = False
    has_debugging_keywords: bool = False
    has_simple_keywords: bool = False
    has_risk_keywords: bool = False
    question_depth: str = "none"  # "what" | "how" | "why" | "none"
    has_implicit_requirements: bool = False
    has_reasoning_keywords: bool = False  # Phase A: detect analysis/critique/review prompts


@dataclass
class StructuralSignals:
    """Structural signals extracted from a task prompt."""
    estimated_subtasks: int = 0
    cross_file_dependencies: bool = False
    has_test_requirements: bool = False
    domain_specificity: str = "general"  # "security" | "infrastructure" | "general"
    requires_external_knowledge: bool = False
    reversibility: str = "easy"  # "easy" | "moderate" | "difficult"
    impact_scope: str = "line"  # "line" | "function" | "module" | "system-wide"


@dataclass
class ContextSignals:
    """Runtime context signals that affect routing decisions."""
    previous_failures: int = 0
    conversation_turns: int = 0
    plan_complexity: int = 0  # Number of planned subtasks
    remaining_tasks: int = 0
    agent_chain_depth: int = 0  # Delegation depth


@dataclass
class ComplexitySignals:
    """Aggregated complexity signals for a task."""
    lexical: LexicalSignals = field(default_factory=LexicalSignals)
    structural: StructuralSignals = field(default_factory=StructuralSignals)
    context: ContextSignals = field(default_factory=ContextSignals)


# Keyword sets
_ARCHITECTURE_KEYWORDS = {
    "refactor", "redesign", "migrate", "architecture", "scalable", "system",
    "infrastructure", "pipeline", "microservice", "monolith", "decouple",
    "abstraction", "interface", "pattern", "framework",
}
_DEBUGGING_KEYWORDS = {
    "root cause", "investigate", "race condition", "intermittent", "deadlock",
    "memory leak", "crash", "corruption", "undefined behavior", "flaky",
    "regression", "stack trace", "debug", "diagnose",
}
_SIMPLE_KEYWORDS = {
    "find", "list", "show", "what is", "where is", "how many", "count",
    "print", "display", "check", "status", "version",
}
_RISK_KEYWORDS = {
    "production", "critical", "security", "migration", "breaking",
    "deployment", "rollback", "database", "schema", "credentials",
    "authentication", "authorization", "encryption",
}
_COMPLEXITY_BLOCKLIST = {
    "debug", "implement", "refactor", "analyze", "analyse", "optimize", "optimise", "redesign",
    "architect", "investigate", "migrate", "integrate", "configure",
    "deploy", "benchmark", "profile", "review", "audit", "fix",
    "resolve", "troubleshoot", "diagnose", "design", "plan",
    "evaluate", "compare", "assess", "test", "validate", "validation",
    "add", "create", "build", "update", "modify", "change", "remove",
    "delete", "write", "generate", "convert", "transform", "handle",
    "trace", "reason", "why does", "explain why", "trade-off", "tradeoff",
    "critique",
}

# Patterns
_FILE_PATH_RE = re.compile(r'(?:[\w./\\-]+\.\w{1,10})|(?:src/|lib/|core/|tests?/)')
_CODE_BLOCK_RE = re.compile(r'```')
_QUESTION_RE = re.compile(r'\b(why|how|what)\b', re.IGNORECASE)


def extract_signals(
    prompt: str,
    agent_type: str | None = None,
    context: ContextSignals | None = None,
) -> ComplexitySignals:
    """Extract lexical, structural, and context signals from a task prompt.

    Args:
        prompt: The task text to analyze.
        agent_type: Optional agent type for specialized routing.
        context: Runtime context (failures, turns, depth) for adaptive routing.
    """
    words = prompt.split()
    lower = prompt.lower()

    lexical = LexicalSignals(
        word_count=len(words),
        file_path_count=len(_FILE_PATH_RE.findall(prompt)),
        code_block_count=len(_CODE_BLOCK_RE.findall(prompt)) // 2,
        has_architecture_keywords=any(kw in lower for kw in _ARCHITECTURE_KEYWORDS),
        has_debugging_keywords=any(kw in lower for kw in _DEBUGGING_KEYWORDS),
        has_simple_keywords=any(kw in lower for kw in _SIMPLE_KEYWORDS),
        has_risk_keywords=any(kw in lower for kw in _RISK_KEYWORDS),
        question_depth=_detect_question_depth(lower),
        has_implicit_requirements="make it work" in lower or "just fix" in lower,
        has_reasoning_keywords=any(kw in lower for kw in _REASONING_KEYWORDS),
    )

    structural = StructuralSignals(
        estimated_subtasks=_count_subtasks(prompt),
        cross_file_dependencies=lexical.file_path_count >= 2,
        has_test_requirements="test" in lower and ("add" in lower or "write" in lower or "create" in lower),
        domain_specificity=_detect_domain(lower),
        requires_external_knowledge="api" in lower or "documentation" in lower or "spec" in lower,
        reversibility=_assess_reversibility(lower),
        impact_scope=_assess_impact(lower, lexical.file_path_count),
    )

    return ComplexitySignals(
        lexical=lexical,
        structural=structural,
        context=context or ContextSignals(),
    )


def score_complexity(signals: ComplexitySignals) -> float:
    """Score task complexity from signals. Higher = more complex.

    Rescaled from OMC's integer weights to 0-100 range.
    Thresholds: <30 = LOW, <76 = STANDARD, <90 = ADVANCED, >=90 = CRITICAL.
    """
    score = 0.0
    lex = signals.lexical
    struct = signals.structural
    ctx = signals.context

    # Lexical signals
    if lex.word_count > 200:
        score += 10
    if lex.word_count > 500:
        score += 10
    if lex.file_path_count >= 2:
        score += 15
    if lex.code_block_count > 0:
        score += 5
    if lex.has_architecture_keywords:
        score += 20
    if lex.has_debugging_keywords:
        score += 15
    if lex.has_simple_keywords:
        score -= 15
    if lex.has_risk_keywords:
        score += 20
    if lex.question_depth == "why":
        score += 10
    elif lex.question_depth == "how":
        score += 5
    if lex.has_implicit_requirements:
        score += 10
    if lex.has_reasoning_keywords:
        score += 15  # Phase A: bump prose/reasoning tasks that scored ~0 before
    if 120 < lex.word_count <= 200 and lex.code_block_count == 0 and lex.file_path_count == 0:
        score += 10  # Phase A: prose-sophistication bump (120–200 word band, de-overlapped)

    # Structural signals
    if struct.estimated_subtasks > 3:
        score += 20
    elif struct.estimated_subtasks > 1:
        score += 10
    if struct.cross_file_dependencies:
        score += 15
    if struct.has_test_requirements:
        score += 5
    if struct.domain_specificity == "security":
        score += 20
    elif struct.domain_specificity == "infrastructure":
        score += 15
    if struct.requires_external_knowledge:
        score += 10
    if struct.reversibility == "difficult":
        score += 20
    elif struct.reversibility == "moderate":
        score += 10
    if struct.impact_scope == "system-wide":
        score += 20
    elif struct.impact_scope == "module":
        score += 10

    # Context signals — escalate on failures, deep chains, complex plans
    if ctx.previous_failures >= 2:
        score += 20  # Two failures → escalate to stronger model
    elif ctx.previous_failures >= 1:
        score += 10
    if ctx.agent_chain_depth >= 2:
        score += 10  # Deep delegation chains need better models
    if ctx.plan_complexity >= 5:
        score += 15  # Complex plans with many steps
    elif ctx.plan_complexity >= 3:
        score += 5

    return max(0, score)


def confidence_breakdown(signals: ComplexitySignals) -> dict:
    """Return a dict explaining which signals contributed to the routing score."""
    breakdown = {
        "lexical_signals": {},
        "structural_signals": {},
        "context_signals": {},
        "total_score": 0.0,
    }
    score_contributions = {}

    lex = signals.lexical
    struct = signals.structural
    ctx = signals.context

    # Lexical signals
    if lex.word_count > 200:
        score_contributions["lexical_word_count_200+"] = 10
    if lex.word_count > 500:
        score_contributions["lexical_word_count_500+"] = 10
    if lex.file_path_count >= 2:
        score_contributions["lexical_file_path_count_2+"] = 15
    if lex.code_block_count > 0:
        score_contributions["lexical_code_block_count_1+"] = 5
    if lex.has_architecture_keywords:
        score_contributions["lexical_architecture_keywords"] = 20
    if lex.has_debugging_keywords:
        score_contributions["lexical_debugging_keywords"] = 15
    if lex.has_simple_keywords:
        score_contributions["lexical_simple_keywords"] = -15
    if lex.has_risk_keywords:
        score_contributions["lexical_risk_keywords"] = 20
    if lex.question_depth == "why":
        score_contributions["lexical_question_depth_why"] = 10
    elif lex.question_depth == "how":
        score_contributions["lexical_question_depth_how"] = 5
    if lex.has_implicit_requirements:
        score_contributions["lexical_implicit_requirements"] = 10
    if lex.has_reasoning_keywords:
        score_contributions["lexical_reasoning_keywords"] = 15  # Phase A
    if 120 < lex.word_count <= 200 and lex.code_block_count == 0 and lex.file_path_count == 0:
        score_contributions["lexical_prose_sophistication"] = 10  # Phase A

    # Structural signals
    if struct.estimated_subtasks > 3:
        score_contributions["structural_estimated_subtasks_3+"] = 20
    elif struct.estimated_subtasks > 1:
        score_contributions["structural_estimated_subtasks_1+"] = 10
    if struct.cross_file_dependencies:
        score_contributions["structural_cross_file_dependencies"] = 15
    if struct.has_test_requirements:
        score_contributions["structural_test_requirements"] = 5
    if struct.domain_specificity == "security":
        score_contributions["structural_domain_security"] = 20
    elif struct.domain_specificity == "infrastructure":
        score_contributions["structural_domain_infrastructure"] = 15
    if struct.requires_external_knowledge:
        score_contributions["structural_external_knowledge"] = 10
    if struct.reversibility == "difficult":
        score_contributions["structural_reversibility_difficult"] = 20
    elif struct.reversibility == "moderate":
        score_contributions["structural_reversibility_moderate"] = 10
    if struct.impact_scope == "system-wide":
        score_contributions["structural_impact_system_wide"] = 20
    elif struct.impact_scope == "module":
        score_contributions["structural_impact_module"] = 10

    # Context signals
    if ctx.previous_failures >= 2:
        score_contributions["context_previous_failures_2+"] = 20
    elif ctx.previous_failures >= 1:
        score_contributions["context_previous_failures_1+"] = 10
    if ctx.agent_chain_depth >= 2:
        score_contributions["context_agent_chain_depth_2+"] = 10
    if ctx.plan_complexity >= 5:
        score_contributions["context_plan_complexity_5+"] = 15
    elif ctx.plan_complexity >= 3:
        score_contributions["context_plan_complexity_3+"] = 5
    
    total_score = 0.0
    for key, value in score_contributions.items():
        total_score += value
        if key.startswith("lexical_"):
            breakdown["lexical_signals"][key.replace("lexical_", "")] = value
        elif key.startswith("structural_"):
            breakdown["structural_signals"][key.replace("structural_", "")] = value
        elif key.startswith("context_"):
            breakdown["context_signals"][key.replace("context_", "")] = value

    breakdown["total_score"] = max(0, total_score)
    return breakdown


def score_to_tier(
    score: float,
    thresholds: dict | None = None,
    *,
    allowed_tiers: list[str] | None = None,
) -> Tier:
    """Convert complexity score to a tier, capped to allowed_tiers if provided."""
    t = thresholds or {"local_max": 30, "standard_max": 76, "advanced_max": 90}
    local_max = t.get("local_max", t.get("low_max", 30))
    standard_max = t.get("standard_max", t.get("medium_max", 76))
    advanced_max = t.get("advanced_max", 90)
    if score < local_max:
        recommended = Tier.LOW
    elif score < standard_max:
        recommended = Tier.STANDARD
    elif score < advanced_max:
        recommended = Tier.ADVANCED
    else:
        recommended = Tier.CRITICAL

    if allowed_tiers is None:
        return recommended

    order = ["LOW", "STANDARD", "ADVANCED", "CRITICAL"]
    idx = {name: i for i, name in enumerate(order)}
    rec_idx = idx[recommended.value]
    allowed_idx = sorted(idx[t] for t in allowed_tiers if t in idx)
    if not allowed_idx:
        return recommended
    if rec_idx in allowed_idx:
        return recommended
    below = [i for i in allowed_idx if i <= rec_idx]
    capped_idx = max(below) if below else min(allowed_idx)
    import logging as _lg
    _lg.getLogger(__name__).info(
        "Complexity tier %s capped to %s by allowlist",
        recommended.value, order[capped_idx],
    )
    return Tier(order[capped_idx])


# Escape hatch prefixes — user can force lightweight routing (from OMC task-size-detector)
_ESCAPE_PREFIXES = ("quick:", "simple:", "just:", "minor:", "small:")


def strip_escape_prefix(text: str) -> tuple[str, bool]:
    """Strip escape hatch prefix if present. Returns (clean_text, had_prefix)."""
    lower = text.lower().lstrip()
    for prefix in _ESCAPE_PREFIXES:
        if lower.startswith(prefix):
            return text[len(prefix):].lstrip(), True
    return text, False


def is_simple_message(text: str) -> bool:
    """Fast pre-check: is this message simple enough to skip full analysis?

    From Hermes smart_model_routing.py — char/word count plus blocklist.
    Escape hatch prefixes (quick:, simple:, just:) force simple routing.
    """
    # Escape hatch: explicit user override
    _, had_prefix = strip_escape_prefix(text)
    if had_prefix:
        return True

    if len(text) > 160:
        return False
    if len(text.split()) > 28:
        return False
    if text.count("\n") > 1:
        return False
    if "`" in text:
        return False
    if re.search(r'https?://', text):
        return False
    lower = text.lower()
    return not any(kw in lower for kw in _COMPLEXITY_BLOCKLIST)


_REASONING_KEYWORDS = {
    "analyse", "analyze", "review", "trace", "reason",
    "why does", "explain why", "trade-off", "tradeoff",
    "evaluate", "compare", "assess", "critique", "audit",
}


def is_reasoning_task(prompt: str) -> bool:
    """Fast pre-check: does this prompt call for deep reasoning/analysis?

    Returns False for simple messages (they're not worth a thinking model).
    """
    if is_simple_message(prompt):
        return False
    lower = prompt.lower()
    return any(kw in lower for kw in _REASONING_KEYWORDS)


# --- Internal helpers ---

def _detect_question_depth(lower: str) -> str:
    if "why" in lower:
        return "why"
    if "how" in lower:
        return "how"
    if "what" in lower:
        return "what"
    return "none"


def _count_subtasks(prompt: str) -> int:
    """Estimate number of distinct subtasks in the prompt."""
    lines = prompt.strip().split("\n")
    numbered = sum(1 for line in lines if re.match(r'^\s*\d+[\.\)]\s', line))
    if numbered >= 2:
        return numbered
    # Count distinct action verbs
    actions = re.findall(r'\b(add|creat|fix|updat|remov|implement|writ|build|test|deploy|handl|validat|refactor|configur|integrat|convert|generat|modif)\w*\b',
                         prompt.lower())
    return max(1, len(set(actions)))


def _detect_domain(lower: str) -> str:
    security_terms = {"auth", "login", "password", "token", "permission", "rbac", "oauth", "jwt", "encrypt"}
    infra_terms = {"docker", "kubernetes", "deploy", "ci/cd", "terraform", "ansible", "nginx", "aws", "gcp"}

    if any(t in lower for t in security_terms):
        return "security"
    if any(t in lower for t in infra_terms):
        return "infrastructure"
    return "general"


def _assess_reversibility(lower: str) -> str:
    hard = {"migration", "delete", "drop", "remove table", "schema change", "breaking change"}
    moderate = {"refactor", "rename", "move", "restructure"}

    if any(t in lower for t in hard):
        return "difficult"
    if any(t in lower for t in moderate):
        return "moderate"
    return "easy"


def _assess_impact(lower: str, file_count: int) -> str:
    if any(t in lower for t in {"system-wide", "entire codebase", "all modules", "everything"}):
        return "system-wide"
    if any(t in lower for t in {"module", "package", "service"}) or file_count >= 5:
        return "module"
    if any(t in lower for t in {"function", "method", "class"}) or file_count >= 2:
        return "function"
    return "line"


def apply_speed_mode(tier: Tier, speed_mode: str) -> Tier:
    """Override routing tier based on speed mode preference.

    economy  → keep recommended tier as-is (local LLM first)
    balanced → promote LOW → STANDARD (skip local for simple tasks too)
    fast     → promote LOW/STANDARD → STANDARD (always use cloud cheap)
    max      → promote everything to ADVANCED (Sonnet for all tasks)

    Args:
        tier: The tier recommended by complexity analysis.
        speed_mode: One of "economy", "balanced", "fast", "max".
    """
    if speed_mode == "economy":
        return tier
    if speed_mode == "balanced":
        # Skip local for tasks that are LOW — bump to STANDARD (Kimi/Gemini)
        if tier == Tier.LOW:
            return Tier.STANDARD
        return tier
    if speed_mode == "fast":
        # Always at least STANDARD — skip local entirely
        if tier == Tier.LOW:
            return Tier.STANDARD
        return tier
    if speed_mode == "max":
        # All tasks at least ADVANCED (Sonnet)
        tiers = [Tier.LOW, Tier.STANDARD, Tier.ADVANCED, Tier.CRITICAL]
        return tiers[max(tiers.index(tier), tiers.index(Tier.ADVANCED))]
    return tier  # Unknown mode — no change


def speed_mode_badge(speed_mode: str) -> str:
    """Return a short HUD badge string for the current speed mode."""
    badges = {
        "economy": "[eco]",
        "balanced": "[bal]",
        "fast": "[fast]",
        "max": "[max]",
    }
    return badges.get(speed_mode, f"[{speed_mode}]")


def recommend_tier(task: str, file_count: int = 0, context: "ContextSignals | None" = None) -> Tier:
    """Recommend a provider tier for a task based on complexity analysis.

    Combines lexical/structural signals with optional file count and context.
    Use this as the single entry point for complexity-aware dispatch.

    Tiers map to providers:
      LOW      → local LLM (free, fast for simple tasks)
      STANDARD → Kimi K2 / Gemini Flash (cheap, good for multi-file tasks)
      ADVANCED → Sonnet (complex reasoning, architecture, security)
      CRITICAL → Opus (final synthesis, decisions, multi-system)

    Args:
        task: Task description string.
        file_count: Number of files involved (if known). Bumps score.
        context: Runtime context signals (failures, depth, plan complexity).
    """
    signals = extract_signals(task, context=context)
    # If caller knows the file count, override the estimated value
    if file_count > 0:
        signals.lexical.file_path_count = max(signals.lexical.file_path_count, file_count)
        signals.structural.cross_file_dependencies = file_count >= 2
    score = score_complexity(signals)
    return score_to_tier(score)


# --- Task type classification ---

TASK_TYPE_PROVIDERS = {
    "code": "ADVANCED",
    "search": "LOW",
    "generate": "STANDARD",
    "lookup": "LOW",
}

_CODE_KEYWORDS = {"write", "implement", "fix", "refactor", "create", "add", "build", "port", "migrate", "debug"}
_SEARCH_KEYWORDS = {"find", "search", "read", "look", "check", "where", "grep", "locate", "show me"}
_GENERATE_KEYWORDS = {"explain", "summarize", "document", "describe", "report", "list all", "compare"}
_LOOKUP_KEYWORDS = {"what is", "how many", "version", "status", "which", "does it", "is there"}


def classify_task_type(task: str) -> str:
    """Classify a task by type for provider routing."""
    lower = task.lower()
    scores = {"code": 0, "search": 0, "generate": 0, "lookup": 0}
    for kw in _CODE_KEYWORDS:
        if kw in lower:
            scores["code"] += 1
    for kw in _SEARCH_KEYWORDS:
        if kw in lower:
            scores["search"] += 1
    for kw in _GENERATE_KEYWORDS:
        if kw in lower:
            scores["generate"] += 1
    for kw in _LOOKUP_KEYWORDS:
        if kw in lower:
            scores["lookup"] += 1
    if max(scores.values()) == 0:
        return "code"  # Default
    return max(scores, key=scores.get)
