"""Config validation for mullet.yaml.

Catches common mistakes early with clear, actionable error messages.
"""

from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    """A single validation issue."""
    path: str
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class ValidationResult:
    """Result of config validation."""
    valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(self, path: str, message: str):
        """Append an `error`-severity issue at `path` and flag the result invalid."""
        self.issues.append(ValidationIssue(path=path, message=message, severity="error"))
        self.valid = False

    def add_warning(self, path: str, message: str):
        """Append a `warning`-severity issue at `path` (does not flip ``valid``)."""
        self.issues.append(ValidationIssue(path=path, message=message, severity="warning"))

    def format(self) -> str:
        """Return a one-line-per-issue human-readable summary."""
        if not self.issues:
            return "Config valid."
        lines = []
        for issue in self.issues:
            prefix = "ERROR" if issue.severity == "error" else "WARN"
            lines.append(f"  [{prefix}] {issue.path}: {issue.message}")
        return "\n".join(lines)


# Valid provider types
# Provider types that have a real Provider class in mullet.core.providers.
# - openai_compatible: any provider with an OpenAI-compatible REST surface
#   (LM Studio, Kimi, OpenRouter, OpenAI itself, DeepSeek direct, etc.)
# - anthropic: Anthropic SDK or `claude` CLI fallback
# - bedrock: AWS Bedrock
# - gemini: Google Gemini native API (mullet/core/providers/gemini.py)
_VALID_PROVIDER_TYPES = {"openai_compatible", "anthropic", "bedrock", "gemini"}

# Provider price tiers. "standard" sits between cheap and premium for things
# like Claude Sonnet — cheaper than Opus but not in the "ultra-cheap"
# OpenRouter/Kimi class.
_VALID_TIERS = {"free", "cheap", "standard", "premium"}
_VALID_SELECTION_STRATEGIES = {"round_robin", "least_used", "random"}


def validate_config(data: dict) -> ValidationResult:
    """Validate a mullet.yaml config dict.

    Checks structure, types, required fields, and value ranges.
    Returns ValidationResult with all issues found.
    """
    result = ValidationResult()

    if not isinstance(data, dict):
        result.add_error("root", "Config must be a YAML mapping (dict)")
        return result

    # Validate providers section
    providers = data.get("providers")
    if providers is not None:
        if not isinstance(providers, dict):
            result.add_error("providers", "Must be a mapping of provider names")
        else:
            for name, pconfig in providers.items():
                _validate_provider(name, pconfig, result)

    # Validate routing section
    routing = data.get("routing")
    if routing is not None:
        if not isinstance(routing, dict):
            result.add_error("routing", "Must be a mapping")
        else:
            if "default_tier" in routing:
                tier = routing["default_tier"]
                valid_tiers = ("LOW", "STANDARD", "ADVANCED", "CRITICAL", "MEDIUM", "HIGH")
                if tier not in valid_tiers:
                    result.add_error("routing.default_tier", f"Must be one of {', '.join(valid_tiers[:4])} (got '{tier}')")

    # Validate budget section
    budget = data.get("budget")
    if budget is not None:
        if not isinstance(budget, dict):
            result.add_error("budget", "Must be a mapping")
        else:
            for key in ("per_task", "per_session", "daily"):
                if key in budget:
                    val = budget[key]
                    if not isinstance(val, (int, float)) or val <= 0:
                        result.add_error(f"budget.{key}", f"Must be a positive number (got {val!r})")
            from mullet.jobs.budget import parse_budget

            for problem in parse_budget(budget, "budget").problems:
                result.add_error("budget", problem)

    delegation = data.get("delegation")
    if delegation is not None:
        if not isinstance(delegation, dict):
            result.add_error("delegation", "Must be a mapping")
        else:
            from mullet.core.toolset_policy import parse_policy

            for problem in parse_policy(delegation.get("toolsets"), "delegation.toolsets").problems:
                result.add_error("delegation.toolsets", problem)

    session = data.get("session")
    if session is not None:
        if not isinstance(session, dict):
            result.add_error("session", "Must be a mapping")
        else:
            from mullet.core.strategy import parse_settings

            for problem in parse_settings(session, "session").problems:
                if "unknown setting" in problem:
                    result.add_warning(problem.split(":", 1)[0], "Unknown session setting (ignored)")
                else:
                    result.add_error("session", problem)

    jobs = data.get("jobs")
    if jobs is not None:
        if not isinstance(jobs, dict):
            result.add_error("jobs", "Must be a mapping")
        else:
            from mullet.jobs.settings import parse_settings as parse_jobs

            for problem in parse_jobs(jobs, "jobs").problems:
                if "unknown setting" in problem:
                    result.add_warning(problem.split(":", 1)[0], "Unknown jobs setting (ignored)")
                else:
                    result.add_error("jobs", problem)

    sandbox = data.get("sandbox")
    if sandbox is not None:
        from mullet.jobs.sandbox_backends.settings import parse_settings as parse_sandbox

        for problem in parse_sandbox(sandbox, "sandbox").problems:
            if "unknown setting" in problem:
                result.add_warning(problem.split(":", 1)[0], "Unknown sandbox setting (ignored)")
            else:
                result.add_error(problem.split(":", 1)[0], problem)

    # Validate max_iterations
    max_iter = data.get("max_iterations")
    if max_iter is not None:
        if not isinstance(max_iter, int) or max_iter < 1 or max_iter > 500:
            result.add_error("max_iterations", f"Must be 1-500 (got {max_iter!r})")

    # Validate server section
    server = data.get("server")
    if server is not None:
        if not isinstance(server, dict):
            result.add_error("server", "Must be a mapping")
        else:
            port = server.get("port")
            if port is not None and (not isinstance(port, int) or port < 1 or port > 65535):
                result.add_error("server.port", f"Must be 1-65535 (got {port!r})")

    # Validate toolsets
    toolsets = data.get("toolsets")
    if toolsets is not None:
        if not isinstance(toolsets, list):
            result.add_error("toolsets", "Must be a list of toolset names")

    # Warn about unknown top-level keys
    known_keys = {
        "providers", "routing", "budget", "max_iterations", "server",
        "knowledge", "toolsets", "plugins", "skills", "default_toolsets", "ladder", "delegation", "session",
        "sandbox", "jobs",
    }
    for key in data:
        if key not in known_keys and not key.startswith("#"):
            result.add_warning(key, "Unknown config key (ignored)")

    return result


def _validate_provider(name: str, config: dict, result: ValidationResult):
    """Validate a single provider config."""
    prefix = f"providers.{name}"

    if not isinstance(config, dict):
        result.add_error(prefix, "Must be a mapping")
        return

    # Type is required
    ptype = config.get("type")
    if not ptype:
        result.add_error(f"{prefix}.type", "Required. Must be one of: openai_compatible, anthropic, bedrock")
    elif ptype not in _VALID_PROVIDER_TYPES:
        result.add_error(f"{prefix}.type", f"Unknown type '{ptype}'. Must be one of: {', '.join(_VALID_PROVIDER_TYPES)}")

    # Tier is recommended
    tier = config.get("tier")
    if tier and tier not in _VALID_TIERS:
        result.add_error(f"{prefix}.tier", f"Must be one of: {', '.join(_VALID_TIERS)} (got '{tier}')")

    # OpenAI-compatible needs base_url
    if ptype == "openai_compatible" and not config.get("base_url"):
        result.add_warning(f"{prefix}.base_url", "Recommended for openai_compatible providers")

    # API key env var should reference a real env var pattern
    api_key_env = config.get("api_key_env")
    if api_key_env and not isinstance(api_key_env, str):
        result.add_error(f"{prefix}.api_key_env", "Must be a string (environment variable name)")

    # Models should be a dict
    models = config.get("models")
    if models is not None and not isinstance(models, dict):
        result.add_error(f"{prefix}.models", "Must be a mapping of role → model name")
