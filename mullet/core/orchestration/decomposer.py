"""Task decomposer — split tasks into parallelizable components.

Analyses task text to detect type, complexity, and areas of work.
Generates subtasks with non-overlapping file ownership and
dependency-based execution ordering.

Ported from OMC features/task-decomposer/.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# --- Task Types ---

TASK_TYPES = {
    "fullstack": ["fullstack", "full stack", "frontend and backend", "front-end and back-end"],
    "refactor": ["refactor", "restructure", "reorganize", "clean up", "modernize"],
    "bugfix": ["fix", "bug", "issue", "broken", "error", "crash", "failing"],
    "feature": ["add", "implement", "create", "build", "develop", "new feature"],
    "testing": ["test", "coverage", "spec", "e2e", "integration test", "unit test"],
    "migration": ["migrate", "upgrade", "convert", "port", "transition"],
    "docs": ["document", "documentation", "readme", "api docs"],
    "devops": ["deploy", "ci/cd", "docker", "kubernetes", "pipeline", "infrastructure"],
}

# Areas that indicate separate components
AREA_KEYWORDS = {
    "frontend": ["frontend", "front-end", "ui", "react", "vue", "angular", "css", "html", "component"],
    "backend": ["backend", "back-end", "api", "server", "endpoint", "route", "controller"],
    "database": ["database", "schema", "migration", "model", "query", "sql", "orm"],
    "auth": ["auth", "login", "signup", "password", "session", "jwt", "oauth", "rbac"],
    "testing": ["test", "spec", "coverage", "fixture", "mock", "pytest", "jest"],
    "config": ["config", "environment", "settings", "env", "yaml", "toml"],
    "cli": ["cli", "command", "argument", "parser", "terminal"],
    "docs": ["docs", "documentation", "readme", "changelog", "api docs"],
}

# Files that are typically shared across components
SHARED_FILES = [
    "package.json", "pyproject.toml", "setup.py", "setup.cfg",
    "tsconfig.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "requirements.txt", "Pipfile", "poetry.lock",
    "README.md", ".gitignore", ".env", ".env.example",
    "docker-compose.yml", "Dockerfile",
    "Makefile", "Justfile",
]


@dataclass
class TaskAnalysis:
    """Analysis of a task's structure and requirements."""
    task: str
    task_type: str = "feature"
    complexity: float = 0.5
    is_parallelizable: bool = False
    estimated_components: int = 1
    areas: list[str] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=list)
    dependencies: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Subtask:
    """A decomposed subtask with ownership and dependencies."""
    id: str
    name: str
    prompt: str
    area: str = ""
    file_patterns: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    model_tier: str = "MEDIUM"
    acceptance_criteria: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


@dataclass
class DecompositionResult:
    """Result of task decomposition."""
    analysis: TaskAnalysis
    subtasks: list[Subtask]
    shared_files: list[str]
    execution_order: list[list[str]]  # Batches of subtask IDs that can run in parallel
    warnings: list[str] = field(default_factory=list)


def decompose_task(task: str, project_root: str = ".") -> DecompositionResult:
    """Main entry point: decompose a task into parallelizable subtasks.

    Steps:
    1. Analyze task type, complexity, areas
    2. Generate subtasks with file ownership
    3. Detect shared files and conflicts
    4. Calculate execution order (dependency batching)
    5. Validate decomposition
    """
    analysis = analyze_task(task)

    if not analysis.is_parallelizable:
        single = Subtask(
            id="main",
            name="Main Task",
            prompt=task,
            model_tier="MEDIUM" if analysis.complexity < 0.7 else "HIGH",
        )
        return DecompositionResult(
            analysis=analysis,
            subtasks=[single],
            shared_files=[],
            execution_order=[["main"]],
        )

    subtasks = generate_subtasks(analysis, task)
    shared = detect_shared_files(subtasks)
    detect_conflicts(subtasks, shared)
    execution_order = calculate_execution_order(subtasks)
    warnings = validate_decomposition(subtasks, shared)

    logger.info("Decomposed into %d subtasks (%d batches): %s",
                len(subtasks), len(execution_order),
                [s.id for s in subtasks])

    return DecompositionResult(
        analysis=analysis,
        subtasks=subtasks,
        shared_files=shared,
        execution_order=execution_order,
        warnings=warnings,
    )


def analyze_task(task: str) -> TaskAnalysis:
    """Analyze a task to determine type, complexity, and areas."""
    lower = task.lower()

    # Detect task type
    task_type = "feature"
    for ttype, keywords in TASK_TYPES.items():
        if any(kw in lower for kw in keywords):
            task_type = ttype
            break

    # Detect areas
    areas = []
    for area, keywords in AREA_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            areas.append(area)

    # Estimate complexity (0-1)
    complexity = _estimate_complexity(lower, areas)

    # Extract file patterns from task text
    file_patterns = re.findall(r'(?:[\w./\\-]+\.\w{1,10})', task)

    # Detect dependencies between areas
    dependencies = _detect_dependencies(areas, task_type)

    # Parallelizable if complex enough and multiple areas
    is_parallelizable = complexity > 0.3 and len(areas) >= 2

    return TaskAnalysis(
        task=task,
        task_type=task_type,
        complexity=complexity,
        is_parallelizable=is_parallelizable,
        estimated_components=max(1, min(len(areas), 6)) if is_parallelizable else 1,
        areas=areas,
        file_patterns=file_patterns,
        dependencies=dependencies,
    )


def generate_subtasks(analysis: TaskAnalysis, original_task: str) -> list[Subtask]:
    """Generate subtasks from analysis areas."""
    subtasks = []

    for area in analysis.areas:
        # Determine file patterns for this area
        patterns = _area_file_patterns(area)

        # Determine dependencies
        blocked_by = []
        for dep_from, dep_to in analysis.dependencies:
            if dep_to == area:
                blocked_by.append(dep_from)

        # Determine tier
        tier = "HIGH" if area in ("auth", "database") else "MEDIUM"
        if analysis.task_type == "testing":
            tier = "MEDIUM"

        # Generate acceptance criteria
        criteria = _generate_criteria(area, analysis.task_type)

        subtask = Subtask(
            id=area,
            name=f"{area.title()} — {analysis.task_type}",
            prompt=f"Focus on the {area} aspect of: {original_task}\n\n"
                   f"You own these file patterns: {', '.join(patterns)}\n"
                   f"Do NOT modify files outside your ownership.",
            area=area,
            file_patterns=patterns,
            blocked_by=blocked_by,
            model_tier=tier,
            acceptance_criteria=criteria,
        )
        subtasks.append(subtask)

    return subtasks


def detect_shared_files(subtasks: list[Subtask]) -> list[str]:
    """Identify files that are shared across multiple subtasks."""
    shared = []
    for f in SHARED_FILES:
        owners = [s.id for s in subtasks if any(f in p for p in s.file_patterns)]
        if len(owners) > 1 or len(subtasks) > 1:
            shared.append(f)
    return shared


def detect_conflicts(subtasks: list[Subtask], shared_files: list[str]):
    """Detect file ownership conflicts between subtasks."""
    all_patterns: dict[str, list[str]] = {}  # pattern -> [subtask_ids]

    for st in subtasks:
        for pattern in st.file_patterns:
            all_patterns.setdefault(pattern, []).append(st.id)

    for st in subtasks:
        conflicts = []
        for pattern in st.file_patterns:
            owners = all_patterns.get(pattern, [])
            if len(owners) > 1 and pattern not in shared_files:
                conflicts.append(pattern)
        st.conflicts = conflicts


def calculate_execution_order(subtasks: list[Subtask]) -> list[list[str]]:
    """Calculate dependency-based execution batches.

    Returns batches where all subtasks in a batch can run in parallel.
    """
    remaining = {s.id: s for s in subtasks}
    completed: set[str] = set()
    batches: list[list[str]] = []

    max_iterations = len(subtasks) + 1  # Prevent infinite loop on cycles
    for _ in range(max_iterations):
        if not remaining:
            break

        # Find subtasks whose dependencies are all completed
        ready = []
        for sid, st in remaining.items():
            if all(dep in completed for dep in st.blocked_by):
                ready.append(sid)

        if not ready:
            # Circular dependency — force remaining into one batch
            ready = list(remaining.keys())
            logger.warning("Circular dependency detected, forcing batch: %s", ready)

        batches.append(ready)
        for sid in ready:
            completed.add(sid)
            del remaining[sid]

    return batches


def validate_decomposition(subtasks: list[Subtask], shared_files: list[str]) -> list[str]:
    """Validate decomposition and return warnings."""
    warnings = []

    # Check for conflicts
    for st in subtasks:
        if st.conflicts:
            warnings.append(f"Subtask '{st.id}' has file conflicts: {st.conflicts}")

    # Check for missing ownership
    if not any(st.file_patterns for st in subtasks):
        warnings.append("No file patterns assigned — subtasks may overlap")

    # Check for too many subtasks
    if len(subtasks) > 6:
        warnings.append(f"High parallelism ({len(subtasks)} subtasks) — consider consolidating")

    return warnings


# --- Internal helpers ---

def _estimate_complexity(lower: str, areas: list[str]) -> float:
    """Estimate task complexity 0-1."""
    score = 0.0

    # Word count contributes
    words = len(lower.split())
    if words > 50:
        score += 0.2
    elif words > 20:
        score += 0.1

    # Multiple areas = more complex
    score += min(0.4, len(areas) * 0.1)

    # Task type modifiers
    if any(w in lower for w in ["refactor", "migrate", "redesign", "architect", "fullstack", "full stack"]):
        score += 0.2
    if any(w in lower for w in ["fix", "bug", "typo"]) and len(areas) <= 1:
        score -= 0.1

    # File references
    file_count = len(re.findall(r'[\w/]+\.\w{1,10}', lower))
    score += min(0.2, file_count * 0.05)

    return max(0.0, min(1.0, score))


def _detect_dependencies(areas: list[str], task_type: str) -> list[tuple[str, str]]:
    """Detect dependencies between areas. Returns [(from, to)] tuples."""
    deps = []

    # Database changes should happen before backend
    if "database" in areas and "backend" in areas:
        deps.append(("database", "backend"))

    # Backend before frontend (API contracts)
    if "backend" in areas and "frontend" in areas:
        deps.append(("backend", "frontend"))

    # Auth before other backend work
    if "auth" in areas and "backend" in areas:
        deps.append(("auth", "backend"))

    # Config before everything
    if "config" in areas:
        for area in areas:
            if area != "config":
                deps.append(("config", area))

    # Testing after implementation
    if "testing" in areas:
        for area in areas:
            if area not in ("testing", "docs"):
                deps.append((area, "testing"))

    return deps


def _area_file_patterns(area: str) -> list[str]:
    """Return typical file patterns for an area."""
    patterns = {
        "frontend": ["src/components/**", "src/pages/**", "src/hooks/**", "*.tsx", "*.jsx", "*.css"],
        "backend": ["src/api/**", "src/routes/**", "src/controllers/**", "src/services/**"],
        "database": ["src/models/**", "migrations/**", "src/db/**", "*.sql"],
        "auth": ["src/auth/**", "src/middleware/auth*"],
        "testing": ["tests/**", "test/**", "__tests__/**", "*.test.*", "*.spec.*"],
        "config": ["*.yaml", "*.toml", "*.json", ".env*"],
        "cli": ["src/cli/**", "src/commands/**"],
        "docs": ["docs/**", "*.md"],
    }
    return patterns.get(area, [f"src/{area}/**"])


def _generate_criteria(area: str, task_type: str) -> list[str]:
    """Generate acceptance criteria for a subtask."""
    criteria = []

    if task_type == "testing":
        criteria.append("All new tests pass")
        criteria.append("No existing tests broken")
    elif task_type == "bugfix":
        criteria.append("Bug is fixed and no regression introduced")
        criteria.append("Root cause is addressed, not just symptoms")
    elif task_type == "refactor":
        criteria.append("Existing behavior is preserved")
        criteria.append("Code is cleaner/simpler")
    else:
        criteria.append("Feature works as described")

    if area == "testing":
        criteria.append("Coverage for new/changed code")
    elif area == "auth":
        criteria.append("No security vulnerabilities introduced")
    elif area == "database":
        criteria.append("Migration is reversible")

    return criteria
