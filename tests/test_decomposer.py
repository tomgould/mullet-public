"""Tests for the task decomposer engine."""

from mullet.core.orchestration.decomposer import (
    analyze_task,
    decompose_task,
    generate_subtasks,
    calculate_execution_order,
    detect_shared_files,
    detect_conflicts,
    validate_decomposition,
    Subtask,
    TaskAnalysis,
)


# --- analyze_task ---

def test_analyze_simple_task():
    a = analyze_task("fix the typo in README")
    assert a.task_type == "bugfix"
    assert a.is_parallelizable is False
    assert a.estimated_components == 1


def test_analyze_fullstack_task():
    a = analyze_task("build a fullstack user dashboard with frontend components and backend API")
    assert a.task_type == "fullstack"
    assert a.is_parallelizable is True
    assert "frontend" in a.areas
    assert "backend" in a.areas
    assert a.estimated_components >= 2


def test_analyze_refactor_task():
    a = analyze_task("refactor the auth module and update the database schema")
    assert a.task_type == "refactor"
    assert "auth" in a.areas
    assert "database" in a.areas


def test_analyze_testing_task():
    a = analyze_task("write unit tests for the API endpoints and frontend components")
    assert a.task_type == "testing"
    assert "testing" in a.areas


def test_analyze_file_patterns():
    a = analyze_task("fix src/auth.py and update tests/test_auth.py")
    assert len(a.file_patterns) >= 2


def test_analyze_complexity_simple():
    a = analyze_task("fix typo")
    assert a.complexity < 0.3


def test_analyze_complexity_complex():
    a = analyze_task(
        "refactor the entire authentication system to use OAuth2, "
        "update the database schema with new migration, "
        "add frontend login components, and write comprehensive tests"
    )
    assert a.complexity > 0.5
    assert a.is_parallelizable is True


# --- decompose_task ---

def test_decompose_simple():
    result = decompose_task("fix a bug in utils.py")
    assert len(result.subtasks) == 1
    assert result.subtasks[0].id == "main"
    assert result.execution_order == [["main"]]


def test_decompose_multi_area():
    result = decompose_task("add user auth to the backend API and frontend login page")
    assert len(result.subtasks) >= 2
    assert len(result.execution_order) >= 1


def test_decompose_returns_warnings():
    result = decompose_task("fix everything")
    assert isinstance(result.warnings, list)


# --- generate_subtasks ---

def test_generate_subtasks_from_analysis():
    analysis = TaskAnalysis(
        task="build frontend and backend",
        areas=["frontend", "backend"],
        dependencies=[("backend", "frontend")],
    )
    subtasks = generate_subtasks(analysis, "build frontend and backend")
    assert len(subtasks) == 2

    frontend = next(s for s in subtasks if s.area == "frontend")
    assert "backend" in frontend.blocked_by


def test_generate_subtasks_file_patterns():
    analysis = TaskAnalysis(
        task="test everything",
        areas=["testing"],
    )
    subtasks = generate_subtasks(analysis, "test everything")
    assert len(subtasks) == 1
    assert any("test" in p for p in subtasks[0].file_patterns)


# --- calculate_execution_order ---

def test_execution_order_no_deps():
    subtasks = [
        Subtask(id="a", name="A", prompt=""),
        Subtask(id="b", name="B", prompt=""),
        Subtask(id="c", name="C", prompt=""),
    ]
    order = calculate_execution_order(subtasks)
    assert len(order) == 1  # All in one batch
    assert set(order[0]) == {"a", "b", "c"}


def test_execution_order_with_deps():
    subtasks = [
        Subtask(id="database", name="DB", prompt=""),
        Subtask(id="backend", name="API", prompt="", blocked_by=["database"]),
        Subtask(id="frontend", name="UI", prompt="", blocked_by=["backend"]),
    ]
    order = calculate_execution_order(subtasks)
    assert len(order) == 3  # Sequential: db -> backend -> frontend
    assert order[0] == ["database"]
    assert order[1] == ["backend"]
    assert order[2] == ["frontend"]


def test_execution_order_mixed():
    subtasks = [
        Subtask(id="config", name="Config", prompt=""),
        Subtask(id="backend", name="API", prompt="", blocked_by=["config"]),
        Subtask(id="frontend", name="UI", prompt="", blocked_by=["config"]),
        Subtask(id="testing", name="Tests", prompt="", blocked_by=["backend", "frontend"]),
    ]
    order = calculate_execution_order(subtasks)
    assert order[0] == ["config"]
    assert set(order[1]) == {"backend", "frontend"}  # Parallel
    assert order[2] == ["testing"]


def test_execution_order_circular():
    subtasks = [
        Subtask(id="a", name="A", prompt="", blocked_by=["b"]),
        Subtask(id="b", name="B", prompt="", blocked_by=["a"]),
    ]
    order = calculate_execution_order(subtasks)
    # Should force into one batch despite cycle
    assert len(order) == 1
    assert set(order[0]) == {"a", "b"}


# --- detect_shared_files ---

def test_detect_shared_files():
    subtasks = [
        Subtask(id="a", name="A", prompt="", file_patterns=["package.json", "src/a/**"]),
        Subtask(id="b", name="B", prompt="", file_patterns=["src/b/**"]),
    ]
    shared = detect_shared_files(subtasks)
    assert "package.json" in shared


# --- detect_conflicts ---

def test_detect_conflicts_none():
    subtasks = [
        Subtask(id="a", name="A", prompt="", file_patterns=["src/a/**"]),
        Subtask(id="b", name="B", prompt="", file_patterns=["src/b/**"]),
    ]
    detect_conflicts(subtasks, [])
    assert subtasks[0].conflicts == []
    assert subtasks[1].conflicts == []


def test_detect_conflicts_overlap():
    subtasks = [
        Subtask(id="a", name="A", prompt="", file_patterns=["src/shared.py"]),
        Subtask(id="b", name="B", prompt="", file_patterns=["src/shared.py"]),
    ]
    detect_conflicts(subtasks, [])
    assert "src/shared.py" in subtasks[0].conflicts


def test_detect_conflicts_shared_excluded():
    subtasks = [
        Subtask(id="a", name="A", prompt="", file_patterns=["package.json"]),
        Subtask(id="b", name="B", prompt="", file_patterns=["package.json"]),
    ]
    detect_conflicts(subtasks, ["package.json"])
    assert subtasks[0].conflicts == []  # Shared files don't count as conflicts


# --- validate_decomposition ---

def test_validate_clean():
    subtasks = [
        Subtask(id="a", name="A", prompt="", file_patterns=["src/a/**"]),
    ]
    warnings = validate_decomposition(subtasks, [])
    assert warnings == []


def test_validate_conflicts_warning():
    subtasks = [
        Subtask(id="a", name="A", prompt="", file_patterns=["src/x.py"], conflicts=["src/x.py"]),
    ]
    warnings = validate_decomposition(subtasks, [])
    assert any("conflict" in w for w in warnings)


def test_validate_too_many_subtasks():
    subtasks = [Subtask(id=str(i), name=f"T{i}", prompt="") for i in range(8)]
    warnings = validate_decomposition(subtasks, [])
    assert any("parallelism" in w.lower() or "consolidat" in w.lower() for w in warnings)
