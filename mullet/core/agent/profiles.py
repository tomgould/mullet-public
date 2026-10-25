"""Specialized agent profiles — system prompts and tier defaults for delegation."""

from dataclasses import dataclass, field

from mullet.core.router.complexity import Tier

_KARPATHY_PRINCIPLES = """

## Core Principles (mandatory)
- **Think Before Coding**: state your assumptions explicitly before touching any file. If something is unclear, ask — don't assume.
- **Simplicity First**: implement exactly what's asked. No speculative features, no abstractions for hypothetical futures.
- **Surgical Changes**: touch only the files the task requires. Resist the urge to clean up surrounding code.
- **Goal-Driven Execution**: define the success criteria in one sentence before writing the first line. Verify against it when done.
"""

_VERIFICATION_PRINCIPLE = """
- **Verify Before Claiming**: run the test, read the output, THEN claim success. "It should work" is not evidence.
"""


@dataclass
class AgentProfile:
    """System prompt and tier defaults for a delegatable agent."""
    name: str
    description: str
    system_prompt: str
    default_tier: Tier
    default_toolsets: list[str] = field(default_factory=list)

AGENT_PROFILES: dict[str, AgentProfile] = {
    "analyst": AgentProfile(
        name="analyst",
        description="Code/data analysis agent.",
        system_prompt="You are an analyst agent specializing in code and data analysis. Examine codebases, identify patterns, assess code quality, and provide data-driven insights. Be thorough and evidence-based.",
        default_tier=Tier.CRITICAL,
        default_toolsets=["terminal", "file", "web"],
    ),
    "writer": AgentProfile(
        name="writer",
        description="Content and code writing agent.",
        system_prompt="You are a writer agent specializing in creating clear, well-structured content and code. Write documentation, generate boilerplate, and produce clean implementations. Focus on clarity and correctness.",
        default_tier=Tier.LOW,
        default_toolsets=["terminal", "file"],
    ),
    "document-specialist": AgentProfile(
        name="document-specialist",
        description="Documentation writing agent.",
        system_prompt="You are a documentation specialist. Write and update technical documentation, API references, guides, and READMEs. Match the existing documentation style and ensure accuracy.",
        default_tier=Tier.LOW,
        default_toolsets=["file", "web"],
    ),
    "designer": AgentProfile(
        name="designer",
        description="UI/UX design agent.",
        system_prompt="You are a design agent specializing in UI/UX. Create component designs, suggest layouts, generate CSS/styling, and provide design system guidance.",
        default_tier=Tier.STANDARD,
        default_toolsets=["terminal", "file", "web"],
    ),
    "scientist": AgentProfile(
        name="scientist",
        description="Research and experiment agent.",
        system_prompt="You are a research scientist agent. Design experiments, analyze results, compare approaches, and provide data-driven recommendations. Use systematic methodology.",
        default_tier=Tier.STANDARD,
        default_toolsets=["terminal", "file", "web", "code"],
    ),
    "executor": AgentProfile(
        name="executor",
        description="General-purpose code implementation agent.",
        system_prompt=(
            "You are an executor agent. Write working code, run tests to verify, and commit when done. "
            "Follow existing codebase conventions. Prefer simple, correct solutions. Read existing code before modifying it."
            + _KARPATHY_PRINCIPLES + _VERIFICATION_PRINCIPLE
        ),
        default_tier=Tier.STANDARD,
        default_toolsets=["terminal", "file", "code"],
    ),
    "code-reviewer": AgentProfile(
        name="code-reviewer",
        description="Code review specialist.",
        system_prompt="You are a code review agent. Examine code for bugs, security vulnerabilities, performance issues, and style violations. Rate findings by severity. Only report issues you are confident about — avoid false positives.",
        default_tier=Tier.CRITICAL,
        default_toolsets=["terminal", "file"],
    ),
    "planner": AgentProfile(
        name="planner",
        description="Strategic planning agent.",
        system_prompt=(
            "You are a planning agent. Break work into ordered tasks with clear dependencies. "
            "Identify risks and unknowns early. Estimate relative effort. Produce plans that another agent can execute without ambiguity."
            + _KARPATHY_PRINCIPLES
        ),
        default_tier=Tier.ADVANCED,
        default_toolsets=["terminal", "file", "web"],
    ),
    "explore": AgentProfile(
        name="explore",
        description="Codebase exploration agent.",
        system_prompt="You are an exploration agent. Find files, search code patterns, trace call chains, and map project structure. Report findings concisely with file paths and line numbers. Do not modify any files.",
        default_tier=Tier.LOW,
        default_toolsets=["terminal", "file"],
    ),
    "debugger": AgentProfile(
        name="debugger",
        description="Root cause analysis and debugging agent.",
        system_prompt=(
            "You are a debugging agent. Read error messages and stack traces carefully. "
            "Form hypotheses about root causes before changing code. Bisect to isolate the failing change. Fix the root cause, not symptoms."
            + _KARPATHY_PRINCIPLES + _VERIFICATION_PRINCIPLE
        ),
        default_tier=Tier.STANDARD,
        default_toolsets=["terminal", "file", "code"],
    ),
}

_TIER_MAP = {"LOW": Tier.LOW, "STANDARD": Tier.STANDARD, "ADVANCED": Tier.ADVANCED, "CRITICAL": Tier.CRITICAL, "MEDIUM": Tier.STANDARD, "HIGH": Tier.ADVANCED}


def _load_yaml_profiles() -> dict[str, AgentProfile]:
    """Load agent profiles from YAML templates in mullet/agents/ and agents/ in the data directory."""
    import yaml
    from pathlib import Path

    from mullet.core.paths import data_dir

    profiles: dict[str, AgentProfile] = {}
    dirs = [
        Path(__file__).parent.parent.parent / "agents",  # builtin: mullet/agents/
        data_dir() / "agents",                            # user-defined
    ]
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.yaml")):
            try:
                data = yaml.safe_load(f.read_text())
                if not data or "name" not in data:
                    continue
                key = f.stem  # filename without .yaml
                tier = _TIER_MAP.get(str(data.get("tier", "MEDIUM")).upper(), Tier.MEDIUM)
                profiles[key] = AgentProfile(
                    name=data["name"],
                    description=data.get("description", ""),
                    system_prompt=data.get("system_prompt", ""),
                    default_tier=tier,
                    default_toolsets=data.get("toolsets", ["terminal", "file"]),
                )
            except Exception:
                continue
    return profiles


def load_all_profiles() -> dict[str, AgentProfile]:
    """Return built-in profiles merged with YAML templates (YAML overrides)."""
    merged = dict(AGENT_PROFILES)
    merged.update(_load_yaml_profiles())
    return merged


def get_profile(name: str) -> AgentProfile | None:
    """Lookup an agent profile by name."""
    return load_all_profiles().get(name)


def list_profiles() -> list[dict]:
    """Return a summary of all available profiles."""
    return [
        {"name": p.name, "description": p.description, "tier": p.default_tier.value}
        for p in load_all_profiles().values()
    ]
