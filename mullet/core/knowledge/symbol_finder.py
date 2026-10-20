"""Symbol finder — fast regex-based function/class/method lookup across a codebase.

Scans source files for symbol definitions using regex patterns.
Supports Python, TypeScript/JavaScript, Go, and PHP.
Returns structured SymbolMatch results with name, type, location, and signature.
No external dependencies (no tree-sitter required).
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SymbolMatch:
    """A symbol found in source code."""
    name: str
    type: str  # "function", "class", "method", "async_function", "async_method", "interface", "struct", "trait"
    file_path: str
    line_number: int
    signature: str
    parent_class: str | None = None


# ── Language config ──────────────────────────────────────────────────────────

_LANG_EXTENSIONS: dict[str, list[str]] = {
    'python': ['.py'],
    'typescript': ['.ts', '.tsx'],
    'javascript': ['.js', '.jsx', '.mjs', '.cjs'],
    'go': ['.go'],
    'php': ['.php', '.module', '.inc'],
}

SUPPORTED_LANGUAGES = set(_LANG_EXTENSIONS.keys())


def _extensions_for(language: str) -> list[str]:
    """Get file extensions for a language."""
    if language not in _LANG_EXTENSIONS:
        raise ValueError(f"Unsupported language: {language}. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}")
    return _LANG_EXTENSIONS[language]


# ── Python scanner ─────────���─────────────────────────────────────────────────

_PY_CLASS = re.compile(
    r'^(?P<indent>\s*)class\s+(?P<name>[a-zA-Z_]\w*)(?P<bases>\(.*?\))?:'
)
_PY_FUNC = re.compile(
    r'^(?P<indent>\s*)(?P<async>async\s+)?def\s+(?P<name>[a-zA-Z_]\w*)\((?P<params>[^)]*)\)(?:\s*->\s*(?P<return>[^:]+))?:'
)


def _scan_python_file(file_path: str) -> list[SymbolMatch]:
    """Extract all symbols from a Python file."""
    symbols: list[SymbolMatch] = []
    current_class: str | None = None
    class_indent: int = -1

    try:
        with open(file_path, errors='replace') as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return []

    for i, line in enumerate(lines):
        cls_match = _PY_CLASS.match(line)
        if cls_match:
            indent = len(cls_match.group('indent'))
            name = cls_match.group('name')
            bases = cls_match.group('bases') or ''
            sig = f"class {name}{bases}:"
            symbols.append(SymbolMatch(
                name=name, type="class", file_path=file_path,
                line_number=i + 1, signature=sig,
            ))
            current_class = name
            class_indent = indent
            continue

        func_match = _PY_FUNC.match(line)
        if func_match:
            indent = len(func_match.group('indent'))
            is_async = bool(func_match.group('async'))
            name = func_match.group('name')
            params = func_match.group('params')
            ret = func_match.group('return')

            prefix = "async def" if is_async else "def"
            sig = f"{prefix} {name}({params})"
            if ret:
                sig += f" -> {ret.strip()}"
            sig += ":"

            is_method = current_class is not None and indent > class_indent
            if is_method:
                sym_type = "async_method" if is_async else "method"
                parent = current_class
            else:
                sym_type = "async_function" if is_async else "function"
                parent = None
                if indent <= class_indent:
                    current_class = None
                    class_indent = -1

            symbols.append(SymbolMatch(
                name=name, type=sym_type, file_path=file_path,
                line_number=i + 1, signature=sig, parent_class=parent,
            ))
            continue

        stripped = line.rstrip()
        if stripped and not stripped.lstrip().startswith('#') and current_class is not None:
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= class_indent and stripped.strip():
                current_class = None
                class_indent = -1

    return symbols


# ── TypeScript / JavaScript scanner ─────���────────────────────────────────────

_TS_CLASS = re.compile(
    r'^(?:export\s+)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_$]\w*)'
    r'(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{'
)
_TS_INTERFACE = re.compile(
    r'^(?:export\s+)?interface\s+(?P<name>[A-Za-z_$]\w*)'
    r'(?:\s+extends\s+[\w,\s]+)?\s*\{'
)
_TS_FUNCTION = re.compile(
    r'^(?:export\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$]\w*)\s*'
    r'(?:<[^>]*>)?\s*\((?P<params>[^)]*)\)'
)
_TS_ARROW_CONST = re.compile(
    r'^(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$]\w*)\s*'
    r'(?::\s*[^=]+)?\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$]\w*)\s*=>'
)
_TS_METHOD = re.compile(
    r'^\s+(?:(?:public|private|protected|static|readonly|async|abstract|override)\s+)*'
    r'(?P<name>[A-Za-z_$]\w*)\s*(?:<[^>]*>)?\s*\((?P<params>[^)]*)\)'
)
_TS_TYPE_ALIAS = re.compile(
    r'^(?:export\s+)?type\s+(?P<name>[A-Za-z_$]\w*)(?:<[^>]*>)?\s*='
)
_TS_ENUM = re.compile(
    r'^(?:export\s+)?(?:const\s+)?enum\s+(?P<name>[A-Za-z_$]\w*)\s*\{'
)


def _scan_ts_file(file_path: str) -> list[SymbolMatch]:
    """Extract symbols from a TypeScript or JavaScript file."""
    symbols: list[SymbolMatch] = []
    current_class: str | None = None
    brace_depth = 0
    class_brace_depth = 0

    try:
        with open(file_path, errors='replace') as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
            continue

        # Track brace depth for class scope
        brace_depth += line.count('{') - line.count('}')

        # Check if we've left a class
        if current_class is not None and brace_depth <= class_brace_depth:
            current_class = None

        # Class
        m = _TS_CLASS.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="class", file_path=file_path,
                line_number=i + 1, signature=stripped.rstrip('{').strip(),
            ))
            current_class = name
            class_brace_depth = brace_depth - 1
            continue

        # Interface
        m = _TS_INTERFACE.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="interface", file_path=file_path,
                line_number=i + 1, signature=stripped.rstrip('{').strip(),
            ))
            continue

        # Enum
        m = _TS_ENUM.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="enum", file_path=file_path,
                line_number=i + 1, signature=stripped.rstrip('{').strip(),
            ))
            continue

        # Type alias
        m = _TS_TYPE_ALIAS.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="type_alias", file_path=file_path,
                line_number=i + 1, signature=stripped.rstrip(';').strip(),
            ))
            continue

        # Function declaration
        m = _TS_FUNCTION.match(stripped)
        if m:
            name = m.group('name')
            is_async = 'async' in stripped.split('function')[0]
            sym_type = "async_function" if is_async else "function"
            symbols.append(SymbolMatch(
                name=name, type=sym_type, file_path=file_path,
                line_number=i + 1, signature=stripped.split('{')[0].strip(),
            ))
            continue

        # Arrow function const
        m = _TS_ARROW_CONST.match(stripped)
        if m:
            name = m.group('name')
            is_async = 'async' in stripped.split('=>')[0]
            sym_type = "async_function" if is_async else "function"
            symbols.append(SymbolMatch(
                name=name, type=sym_type, file_path=file_path,
                line_number=i + 1, signature=stripped.split('=>')[0].strip(),
            ))
            continue

        # Method inside class
        if current_class is not None:
            m = _TS_METHOD.match(line)
            if m:
                name = m.group('name')
                # Skip constructor-like keywords
                if name in ('if', 'for', 'while', 'switch', 'catch', 'return', 'new', 'throw'):
                    continue
                is_async = 'async' in line[:line.index(name)]
                sym_type = "async_method" if is_async else "method"
                symbols.append(SymbolMatch(
                    name=name, type=sym_type, file_path=file_path,
                    line_number=i + 1, signature=line.strip().split('{')[0].strip(),
                    parent_class=current_class,
                ))

    return symbols


# ── Go scanner ─────────��─────────────────────────────────────────────────────

_GO_FUNC = re.compile(
    r'^func\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)'
)
_GO_METHOD = re.compile(
    r'^func\s+\(\s*\w+\s+\*?(?P<receiver>[A-Za-z_]\w*)\s*\)\s*(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)'
)
_GO_STRUCT = re.compile(
    r'^type\s+(?P<name>[A-Za-z_]\w*)\s+struct\s*\{'
)
_GO_INTERFACE = re.compile(
    r'^type\s+(?P<name>[A-Za-z_]\w*)\s+interface\s*\{'
)
_GO_TYPE_ALIAS = re.compile(
    r'^type\s+(?P<name>[A-Za-z_]\w*)\s+(?!struct|interface)\S'
)


def _scan_go_file(file_path: str) -> list[SymbolMatch]:
    """Extract symbols from a Go file."""
    symbols: list[SymbolMatch] = []

    try:
        with open(file_path, errors='replace') as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue

        # Method (receiver function) — must check before plain func
        m = _GO_METHOD.match(stripped)
        if m:
            receiver = m.group('receiver')
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="method", file_path=file_path,
                line_number=i + 1, signature=stripped.split('{')[0].strip(),
                parent_class=receiver,
            ))
            continue

        # Function
        m = _GO_FUNC.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="function", file_path=file_path,
                line_number=i + 1, signature=stripped.split('{')[0].strip(),
            ))
            continue

        # Struct
        m = _GO_STRUCT.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="struct", file_path=file_path,
                line_number=i + 1, signature=f"type {name} struct",
            ))
            continue

        # Interface
        m = _GO_INTERFACE.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="interface", file_path=file_path,
                line_number=i + 1, signature=f"type {name} interface",
            ))
            continue

        # Type alias
        m = _GO_TYPE_ALIAS.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="type_alias", file_path=file_path,
                line_number=i + 1, signature=stripped,
            ))

    return symbols


# ── PHP scanner ───���──────────────────────────────────────────────────────────

_PHP_CLASS = re.compile(
    r'^(?:(?:abstract|final)\s+)?class\s+(?P<name>[A-Za-z_]\w*)'
    r'(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w\\,\s]+)?\s*\{'
)
_PHP_INTERFACE = re.compile(
    r'^interface\s+(?P<name>[A-Za-z_]\w*)(?:\s+extends\s+[\w\\,\s]+)?\s*\{'
)
_PHP_TRAIT = re.compile(
    r'^trait\s+(?P<name>[A-Za-z_]\w*)\s*\{'
)
_PHP_FUNCTION = re.compile(
    r'^function\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)'
)
_PHP_METHOD = re.compile(
    r'^\s+(?:(?:public|private|protected|static|abstract|final)\s+)*'
    r'function\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^)]*)\)'
)


def _scan_php_file(file_path: str) -> list[SymbolMatch]:
    """Extract symbols from a PHP file."""
    symbols: list[SymbolMatch] = []
    current_class: str | None = None
    brace_depth = 0
    class_brace_depth = 0

    try:
        with open(file_path, errors='replace') as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('//') or stripped.startswith('#') or stripped.startswith('/*'):
            continue

        brace_depth += line.count('{') - line.count('}')

        if current_class is not None and brace_depth <= class_brace_depth:
            current_class = None

        # Class
        m = _PHP_CLASS.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="class", file_path=file_path,
                line_number=i + 1, signature=stripped.rstrip('{').strip(),
            ))
            current_class = name
            class_brace_depth = brace_depth - 1
            continue

        # Interface
        m = _PHP_INTERFACE.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="interface", file_path=file_path,
                line_number=i + 1, signature=stripped.rstrip('{').strip(),
            ))
            current_class = name
            class_brace_depth = brace_depth - 1
            continue

        # Trait
        m = _PHP_TRAIT.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="trait", file_path=file_path,
                line_number=i + 1, signature=stripped.rstrip('{').strip(),
            ))
            current_class = name
            class_brace_depth = brace_depth - 1
            continue

        # Method inside class
        if current_class is not None:
            m = _PHP_METHOD.match(line)
            if m:
                name = m.group('name')
                symbols.append(SymbolMatch(
                    name=name, type="method", file_path=file_path,
                    line_number=i + 1, signature=stripped.split('{')[0].strip(),
                    parent_class=current_class,
                ))
                continue

        # Top-level function
        m = _PHP_FUNCTION.match(stripped)
        if m:
            name = m.group('name')
            symbols.append(SymbolMatch(
                name=name, type="function", file_path=file_path,
                line_number=i + 1, signature=stripped.split('{')[0].strip(),
            ))

    return symbols


# ── Scanner dispatch ─────────────────────────────────────────────────────────

_SCANNERS: dict[str, callable] = {
    'python': _scan_python_file,
    'typescript': _scan_ts_file,
    'javascript': _scan_ts_file,
    'go': _scan_go_file,
    'php': _scan_php_file,
}


def _get_scanner(language: str):
    """Get the scanner function for a language."""
    if language not in _SCANNERS:
        raise ValueError(f"Unsupported language: {language}. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}")
    return _SCANNERS[language]


def detect_language(file_path: str) -> str | None:
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()
    for lang, exts in _LANG_EXTENSIONS.items():
        if ext in exts:
            return lang
    return None


# ── Public API ──────��─────────────────────────────��──────────────────────────

def list_symbols(file_path: str, language: str = 'python') -> list[SymbolMatch]:
    """List all symbols in a single file."""
    scanner = _get_scanner(language)
    return scanner(file_path)


def find_symbol(
    name: str,
    root_dir: str,
    language: str = 'python',
    *,
    exact: bool = True,
    max_results: int = 100,
) -> list[SymbolMatch]:
    """Find symbols by name across a directory tree.

    Args:
        name: Symbol name to search for.
        root_dir: Root directory to scan.
        language: Source language ('python', 'typescript', 'javascript', 'go', 'php').
        exact: If True, match name exactly. If False, substring match.
        max_results: Maximum number of results to return.
    """
    scanner = _get_scanner(language)
    extensions = _extensions_for(language)

    matches: list[SymbolMatch] = []
    name_lower = name.lower()

    for ext in extensions:
        for fpath in _walk_files(root_dir, ext):
            for sym in scanner(fpath):
                if exact:
                    if sym.name == name:
                        matches.append(sym)
                else:
                    if name_lower in sym.name.lower():
                        matches.append(sym)
                if len(matches) >= max_results:
                    return matches

    return matches


def find_symbols_by_type(
    root_dir: str,
    sym_type: str,
    language: str = 'python',
) -> list[SymbolMatch]:
    """Find all symbols of a given type (class, function, method, struct, etc.)."""
    scanner = _get_scanner(language)
    extensions = _extensions_for(language)

    return [
        sym
        for ext in extensions
        for fpath in _walk_files(root_dir, ext)
        for sym in scanner(fpath)
        if sym.type == sym_type
    ]


def find_all_symbols(
    root_dir: str,
    language: str | None = None,
) -> list[SymbolMatch]:
    """Find all symbols across all supported languages (or a specific one)."""
    if language:
        scanner = _get_scanner(language)
        extensions = _extensions_for(language)
        return [
            sym
            for ext in extensions
            for fpath in _walk_files(root_dir, ext)
            for sym in scanner(fpath)
        ]

    all_symbols: list[SymbolMatch] = []
    for lang in SUPPORTED_LANGUAGES:
        scanner = _SCANNERS[lang]
        for ext in _extensions_for(lang):
            for fpath in _walk_files(root_dir, ext):
                all_symbols.extend(scanner(fpath))
    return all_symbols


def _walk_files(root_dir: str, extension: str) -> list[str]:
    """Walk directory tree and return files matching extension."""
    results: list[str] = []
    root = Path(root_dir)
    if not root.is_dir():
        return []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith('.') and d not in {
                '__pycache__', 'node_modules', '.git', 'venv', '.venv',
                'vendor', 'dist', 'build', '.tox',
            }
        ]
        for fname in filenames:
            if fname.endswith(extension):
                results.append(os.path.join(dirpath, fname))

    return sorted(results)
