"""LSP integration tool — language server diagnostics and navigation.

Provides agents with access to language server features:
- diagnostics: errors, warnings, hints from the project
- definition: go-to-definition for a symbol at a location
- references: find all references to a symbol
- hover: get type/doc info for a symbol

Uses subprocess to run language-specific analyzers rather than
maintaining persistent LSP connections (simpler, no state).
"""

import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

LSP_SCHEMA = {
    "description": (
        "Get code intelligence from language tools. Provides diagnostics "
        "(errors/warnings), go-to-definition, find references, and hover info. "
        "Works with Python (pyright/mypy), TypeScript (tsc), and Go (gopls)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["diagnostics", "definition", "references", "hover", "symbols"],
                "description": "LSP action: diagnostics (errors/warnings), definition (go-to-def), "
                               "references (find usages), hover (type info), symbols (file symbols).",
            },
            "path": {
                "type": "string",
                "description": "File path to analyze. Required for all actions.",
            },
            "line": {
                "type": "integer",
                "description": "Line number (1-based). Required for definition/references/hover.",
            },
            "column": {
                "type": "integer",
                "description": "Column number (1-based). Required for definition/references/hover.",
            },
            "symbol": {
                "type": "string",
                "description": "Symbol name to search for. Alternative to line/column for references.",
            },
        },
        "required": ["action", "path"],
    },
}

MAX_DIAGNOSTICS = 50


def handle_lsp(action: str, path: str, line: int = 0, column: int = 0,
               symbol: str = "", **kwargs) -> str:
    """Dispatch LSP actions."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})

    handlers = {
        "diagnostics": _diagnostics,
        "definition": _definition,
        "references": _references,
        "hover": _hover,
        "symbols": _symbols,
    }
    handler = handlers.get(action)
    if not handler:
        return json.dumps({"error": f"Unknown LSP action: {action}. Use: {', '.join(handlers)}"})

    try:
        return handler(path, line, column, symbol)
    except Exception as e:
        logger.warning("LSP action %s failed: %s", action, e)
        return json.dumps({"error": f"LSP {action} failed: {type(e).__name__}: {e}"})


def _diagnostics(path: str, line: int, column: int, symbol: str) -> str:
    """Get diagnostics (errors, warnings) for a file or project."""
    ext = Path(path).suffix.lower()

    if ext == ".py":
        return _python_diagnostics(path)
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        return _typescript_diagnostics(path)
    elif ext == ".go":
        return _go_diagnostics(path)
    else:
        # Try python first (most common), then fall back
        return _python_diagnostics(path)


def _python_diagnostics(path: str) -> str:
    """Run pyright or mypy for Python diagnostics."""
    cwd = _find_project_root(path)

    # Try pyright first (faster, better for type checking)
    result = _try_tool(
        ["pyright", "--outputjson", path],
        cwd=cwd,
    )
    if result is not None:
        return _parse_pyright_output(result, path)

    # Fall back to mypy
    result = _try_tool(
        ["python", "-m", "mypy", "--no-color-output", "--show-column-numbers", path],
        cwd=cwd,
    )
    if result is not None:
        return _parse_mypy_output(result, path)

    # Fall back to py_compile for syntax errors
    result = _try_tool(
        ["python", "-m", "py_compile", path],
        cwd=cwd,
    )
    if result is not None:
        if result.returncode == 0:
            return json.dumps({"diagnostics": [], "tool": "py_compile", "message": "No syntax errors"})
        return json.dumps({
            "diagnostics": [{"severity": "error", "message": result.stderr.strip(), "file": path}],
            "tool": "py_compile",
        })

    return json.dumps({"error": "No Python diagnostic tool available (install pyright or mypy)"})


def _typescript_diagnostics(path: str) -> str:
    """Run tsc for TypeScript diagnostics."""
    cwd = _find_project_root(path)

    result = _try_tool(
        ["npx", "tsc", "--noEmit", "--pretty", "false", path],
        cwd=cwd,
    )
    if result is not None:
        return _parse_tsc_output(result, path)

    return json.dumps({"error": "TypeScript compiler not available (npx tsc)"})


def _go_diagnostics(path: str) -> str:
    """Run go vet for Go diagnostics."""
    cwd = _find_project_root(path)

    result = _try_tool(["go", "vet", path], cwd=cwd)
    if result is not None:
        diagnostics = []
        for line in (result.stderr + result.stdout).split("\n"):
            line = line.strip()
            if line and ":" in line:
                diagnostics.append({"severity": "warning", "message": line, "file": path})
        return json.dumps({"diagnostics": diagnostics[:MAX_DIAGNOSTICS], "tool": "go vet"})

    return json.dumps({"error": "Go toolchain not available"})


def _definition(path: str, line: int, column: int, symbol: str) -> str:
    """Find definition of symbol at location."""
    if not line:
        return json.dumps({"error": "line parameter required for definition"})

    ext = Path(path).suffix.lower()
    cwd = _find_project_root(path)

    if ext == ".py":
        # Use jedi for Python go-to-definition
        try:
            import jedi
            script = jedi.Script(path=path, project=jedi.Project(path=cwd))
            defs = script.goto(line=line, column=max(0, column - 1))
            results = []
            for d in defs:
                results.append({
                    "name": d.name,
                    "module": d.module_name,
                    "path": str(d.module_path) if d.module_path else None,
                    "line": d.line,
                    "column": d.column + 1,
                    "type": d.type,
                    "description": d.description,
                })
            return json.dumps({"definitions": results, "tool": "jedi"})
        except ImportError:
            return json.dumps({"error": "jedi not installed (pip install jedi)"})
        except Exception as e:
            return json.dumps({"error": f"Definition lookup failed: {e}"})

    # For other languages, use grep-based fallback
    if symbol:
        return _grep_definition(symbol, cwd, ext)

    return json.dumps({"error": f"Definition lookup not supported for {ext} without jedi"})


def _references(path: str, line: int, column: int, symbol: str) -> str:
    """Find all references to a symbol."""
    if not symbol:
        # Try to extract symbol from file at line/column
        if line:
            symbol = _extract_symbol(path, line, column)
        if not symbol:
            return json.dumps({"error": "symbol parameter required for references"})

    cwd = _find_project_root(path)
    ext = Path(path).suffix.lower()

    # Use grep to find references
    patterns = {
        ".py": f"\\b{symbol}\\b",
        ".ts": f"\\b{symbol}\\b",
        ".tsx": f"\\b{symbol}\\b",
        ".js": f"\\b{symbol}\\b",
        ".go": f"\\b{symbol}\\b",
    }
    pattern = patterns.get(ext, f"\\b{symbol}\\b")

    file_glob = f"*{ext}"
    result = _try_tool(
        ["grep", "-rn", "--include", file_glob, "-E", pattern, "."],
        cwd=cwd,
    )
    if result is None:
        return json.dumps({"error": "grep not available"})

    refs = []
    for line_text in result.stdout.split("\n"):
        line_text = line_text.strip()
        if not line_text or ":" not in line_text:
            continue
        parts = line_text.split(":", 2)
        if len(parts) >= 3:
            refs.append({
                "file": parts[0],
                "line": int(parts[1]) if parts[1].isdigit() else 0,
                "text": parts[2].strip()[:200],
            })
        if len(refs) >= 30:
            break

    return json.dumps({"symbol": symbol, "references": refs, "count": len(refs)})


def _hover(path: str, line: int, column: int, symbol: str) -> str:
    """Get type/documentation info for a symbol."""
    if not line:
        return json.dumps({"error": "line parameter required for hover"})

    ext = Path(path).suffix.lower()

    if ext == ".py":
        try:
            import jedi
            cwd = _find_project_root(path)
            script = jedi.Script(path=path, project=jedi.Project(path=cwd))
            hovers = script.help(line=line, column=max(0, column - 1))
            results = []
            for h in hovers:
                results.append({
                    "name": h.name,
                    "type": h.type,
                    "description": h.description,
                    "docstring": (h.docstring() or "")[:500],
                    "module": h.module_name,
                })
            return json.dumps({"hover": results, "tool": "jedi"})
        except ImportError:
            return json.dumps({"error": "jedi not installed (pip install jedi)"})
        except Exception as e:
            return json.dumps({"error": f"Hover lookup failed: {e}"})

    return json.dumps({"error": f"Hover not supported for {ext}"})


def _symbols(path: str, line: int, column: int, symbol: str) -> str:
    """List symbols (classes, functions, variables) in a file."""
    ext = Path(path).suffix.lower()

    if ext == ".py":
        try:
            import jedi
            cwd = _find_project_root(path)
            script = jedi.Script(path=path, project=jedi.Project(path=cwd))
            names = script.get_names(all_scopes=True, definitions=True)
            symbols = []
            for n in names:
                if n.type in ("class", "function", "statement"):
                    symbols.append({
                        "name": n.name,
                        "type": n.type,
                        "line": n.line,
                        "description": n.description,
                    })
            return json.dumps({"symbols": symbols, "file": path, "tool": "jedi"})
        except ImportError as _exc:
            logger.warning("lsp: jedi not installed, falling back to regex for symbols: %s", _exc)
            pass

    # Fallback: regex-based symbol extraction
    return _regex_symbols(path, ext)


# --- Helpers ---

def _try_tool(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess | None:
    """Try running a tool, return None if not found."""
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=30, cwd=cwd,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Tool timed out: %s", args[0])
        return None


def _find_project_root(path: str) -> str:
    """Find the project root by looking for common markers."""
    current = Path(path).parent if Path(path).is_file() else Path(path)
    markers = {"pyproject.toml", "setup.py", "package.json", "go.mod", "Cargo.toml", ".git"}
    for parent in [current] + list(current.parents):
        if any((parent / m).exists() for m in markers):
            return str(parent)
    return str(current)


def _extract_symbol(path: str, line: int, column: int) -> str:
    """Extract the symbol name at a given position in a file."""
    import re
    try:
        with open(path) as f:
            lines = f.readlines()
        if line <= 0 or line > len(lines):
            return ""
        text = lines[line - 1]
        col = max(0, column - 1)
        # Find word boundary around column
        match = re.search(r'\b\w+\b', text[max(0, col - 20):col + 20])
        return match.group(0) if match else ""
    except Exception:
        return ""


def _grep_definition(symbol: str, cwd: str, ext: str) -> str:
    """Grep-based fallback for finding definitions."""
    patterns = {
        ".py": [f"(def|class)\\s+{symbol}", f"{symbol}\\s*="],
        ".ts": [f"(function|class|const|let|var|interface|type)\\s+{symbol}"],
        ".tsx": [f"(function|class|const|let|var|interface|type)\\s+{symbol}"],
        ".js": [f"(function|class|const|let|var)\\s+{symbol}"],
        ".go": [f"(func|type|var|const)\\s+{symbol}"],
    }
    search_patterns = patterns.get(ext, [f"\\b{symbol}\\b"])

    for pattern in search_patterns:
        result = _try_tool(
            ["grep", "-rn", "--include", f"*{ext}", "-E", pattern, "."],
            cwd=cwd,
        )
        if result and result.stdout.strip():
            defs = []
            for line in result.stdout.split("\n")[:10]:
                parts = line.strip().split(":", 2)
                if len(parts) >= 3:
                    defs.append({
                        "file": parts[0],
                        "line": int(parts[1]) if parts[1].isdigit() else 0,
                        "text": parts[2].strip()[:200],
                    })
            return json.dumps({"definitions": defs, "tool": "grep"})

    return json.dumps({"definitions": [], "message": f"No definition found for '{symbol}'"})


def _regex_symbols(path: str, ext: str) -> str:
    """Extract symbols using regex patterns."""
    import re
    patterns = {
        ".py": re.compile(r'^(?:def|class|async def)\s+(\w+)', re.MULTILINE),
        ".ts": re.compile(r'^(?:export\s+)?(?:function|class|interface|type|const|let|var)\s+(\w+)', re.MULTILINE),
        ".js": re.compile(r'^(?:export\s+)?(?:function|class|const|let|var)\s+(\w+)', re.MULTILINE),
        ".go": re.compile(r'^(?:func|type|var|const)\s+(\w+)', re.MULTILINE),
    }
    pattern = patterns.get(ext)
    if not pattern:
        return json.dumps({"error": f"Symbol extraction not supported for {ext}"})

    try:
        text = Path(path).read_text()
        matches = pattern.finditer(text)
        symbols = []
        for m in matches:
            line_num = text[:m.start()].count("\n") + 1
            symbols.append({"name": m.group(1), "line": line_num})
        return json.dumps({"symbols": symbols, "file": path, "tool": "regex"})
    except Exception as e:
        return json.dumps({"error": f"Symbol extraction failed: {e}"})


def _parse_pyright_output(result: subprocess.CompletedProcess, path: str) -> str:
    """Parse pyright JSON output into diagnostics."""
    try:
        data = json.loads(result.stdout)
        diagnostics = []
        for diag in data.get("generalDiagnostics", []):
            diagnostics.append({
                "severity": diag.get("severity", "error"),
                "message": diag.get("message", ""),
                "file": diag.get("file", path),
                "line": diag.get("range", {}).get("start", {}).get("line", 0),
                "column": diag.get("range", {}).get("start", {}).get("character", 0),
                "rule": diag.get("rule", ""),
            })
        return json.dumps({
            "diagnostics": diagnostics[:MAX_DIAGNOSTICS],
            "tool": "pyright",
            "summary": data.get("summary", {}),
        })
    except json.JSONDecodeError:
        # Parse text output as fallback
        return _parse_text_diagnostics(result.stdout + result.stderr, path, "pyright")


def _parse_mypy_output(result: subprocess.CompletedProcess, path: str) -> str:
    """Parse mypy text output into diagnostics."""
    return _parse_text_diagnostics(result.stdout + result.stderr, path, "mypy")


def _parse_tsc_output(result: subprocess.CompletedProcess, path: str) -> str:
    """Parse tsc text output into diagnostics."""
    return _parse_text_diagnostics(result.stdout + result.stderr, path, "tsc")


def register(registry):
    """Register LSP tool into the tool registry."""
    registry.register(
        name="lsp",
        func=handle_lsp,
        schema=LSP_SCHEMA,
        toolset="lsp",
    )


def _parse_text_diagnostics(output: str, path: str, tool: str) -> str:
    """Parse line-based diagnostic output."""
    diagnostics = []
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Common pattern: file:line:col: severity: message
        parts = line.split(":", 4)
        if len(parts) >= 4:
            severity = "error" if "error" in parts[3].lower() else "warning"
            diagnostics.append({
                "severity": severity,
                "message": parts[-1].strip() if len(parts) > 4 else parts[3].strip(),
                "file": parts[0],
                "line": int(parts[1]) if parts[1].strip().isdigit() else 0,
                "column": int(parts[2]) if parts[2].strip().isdigit() else 0,
            })
        elif line and not line.startswith("Found") and not line.startswith("Success"):
            diagnostics.append({"severity": "info", "message": line, "file": path})

    return json.dumps({"diagnostics": diagnostics[:MAX_DIAGNOSTICS], "tool": tool})
