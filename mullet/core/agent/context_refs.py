"""Context reference parsing and resolution.

Supports @file:path, @folder:path, @git:N, @diff, @staged, @url:url
syntax for injecting external context into agent prompts.

Ported from Hermes agent engine, adapted for Mullet.
"""

import logging
import mimetypes
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Match @diff, @staged, @file:path, @folder:path, @git:N, @url:url
REFERENCE_PATTERN = re.compile(
    r"(?<![\w/])@(?:(?P<simple>diff|staged)\b|(?P<kind>file|folder|git|url):(?P<value>\S+))"
)

# Sensitive paths that should never be injected
_BLOCKED_DIRS = {".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gh", ".history"}
_BLOCKED_FILES = {
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    ".bash_history", ".zsh_history", ".netrc", ".pgpass",
    ".npmrc", ".pypirc", ".env",
}

# Code fence language mapping
_LANG_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "jsx", ".tsx": "tsx", ".rs": "rust", ".go": "go",
    ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c",
    ".rb": "ruby", ".php": "php", ".sh": "bash", ".zsh": "bash",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml",
    ".md": "markdown", ".sql": "sql", ".html": "html", ".css": "css",
    ".xml": "xml", ".swift": "swift", ".kt": "kotlin",
}

MAX_FOLDER_ENTRIES = 200


@dataclass
class ContextReference:
    """A parsed @ reference from user input."""
    raw: str
    kind: str  # file, folder, git, url, diff, staged
    target: str = ""
    start: int = 0
    end: int = 0
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class ContextReferenceResult:
    """Result of resolving context references."""
    message: str
    original_message: str
    references: list[ContextReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    injected_tokens: int = 0
    expanded: bool = False
    blocked: bool = False


def parse_context_references(message: str) -> list[ContextReference]:
    """Extract all @ references from user input."""
    refs = []
    for match in REFERENCE_PATTERN.finditer(message):
        simple = match.group("simple")
        kind = match.group("kind")
        value = match.group("value") or ""

        # Strip trailing punctuation from value
        if value and value[-1] in ".,;:!?)]}":
            value = value[:-1]

        if simple:
            refs.append(ContextReference(
                raw=match.group(0), kind=simple,
                start=match.start(), end=match.end(),
            ))
        elif kind and value:
            # Parse line range for file refs: @file:path:10-20
            line_start = line_end = None
            if kind == "file" and ":" in value:
                parts = value.rsplit(":", 1)
                range_match = re.match(r"(\d+)(?:-(\d+))?$", parts[1])
                if range_match:
                    value = parts[0]
                    line_start = int(range_match.group(1))
                    line_end = int(range_match.group(2)) if range_match.group(2) else line_start

            refs.append(ContextReference(
                raw=match.group(0), kind=kind, target=value,
                start=match.start(), end=match.end(),
                line_start=line_start, line_end=line_end,
            ))
    return refs


def resolve_context_references(
    message: str,
    cwd: str | None = None,
    context_length: int = 128_000,
) -> ContextReferenceResult:
    """Resolve all @ references in a message.

    Reads files, fetches git diffs, lists folders. Appends resolved
    content to the message and strips the @ tokens.

    Args:
        message: User input that may contain @ references.
        cwd: Working directory for path resolution. Defaults to os.getcwd().
        context_length: Model context window size for token budget.

    Returns:
        ContextReferenceResult with expanded message and metadata.
    """
    refs = parse_context_references(message)
    if not refs:
        return ContextReferenceResult(message=message, original_message=message)

    cwd = cwd or os.getcwd()
    hard_limit = context_length // 2  # 50% of context
    soft_limit = context_length // 4  # 25% of context

    context_blocks: list[str] = []
    warnings: list[str] = []
    injected_tokens = 0

    for ref in refs:
        content, error = _expand_reference(ref, cwd)
        if error:
            warnings.append(error)
            continue
        if content:
            # Scan for prompt injection before injecting
            from mullet.core.agent.injection_guard import scan_content, sanitize_content
            findings = scan_content(content, source=ref.raw)
            if findings:
                content, _ = sanitize_content(content, source=ref.raw)
                categories = set(f["category"] for f in findings)
                warnings.append(f"Injection patterns in {ref.raw}: {', '.join(categories)}")

            tokens_est = len(content) // 4
            injected_tokens += tokens_est

            if injected_tokens > hard_limit:
                warnings.append(f"Context too large ({injected_tokens} tokens > {hard_limit} limit). Skipping remaining references.")
                return ContextReferenceResult(
                    message=message, original_message=message,
                    references=refs, warnings=warnings,
                    injected_tokens=injected_tokens,
                    blocked=True,
                )
            context_blocks.append(content)

    if injected_tokens > soft_limit:
        warnings.append(f"Large context injection ({injected_tokens} tokens). Consider using fewer references.")

    # Strip @ tokens from message
    clean_message = message
    for ref in reversed(refs):  # Reverse to preserve positions
        clean_message = clean_message[:ref.start] + clean_message[ref.end:]
    clean_message = re.sub(r"\s{2,}", " ", clean_message).strip()

    # Append context blocks
    if context_blocks:
        expanded_message = clean_message + "\n\n" + "\n\n".join(context_blocks)
    else:
        expanded_message = clean_message

    return ContextReferenceResult(
        message=expanded_message,
        original_message=message,
        references=refs,
        warnings=warnings,
        injected_tokens=injected_tokens,
        expanded=bool(context_blocks),
    )


def _expand_reference(ref: ContextReference, cwd: str) -> tuple[str | None, str | None]:
    """Resolve a single reference. Returns (content, error)."""
    if ref.kind == "file":
        return _expand_file(ref, cwd)
    elif ref.kind == "folder":
        return _expand_folder(ref, cwd)
    elif ref.kind in ("diff", "staged"):
        return _expand_git_diff(ref, cwd)
    elif ref.kind == "git":
        return _expand_git_log(ref, cwd)
    elif ref.kind == "url":
        return _expand_url(ref)
    return None, f"Unknown reference kind: {ref.kind}"


def _expand_file(ref: ContextReference, cwd: str) -> tuple[str | None, str | None]:
    """Read a file and return formatted content."""
    path = _resolve_path(ref.target, cwd)
    if not path:
        return None, f"Path blocked: {ref.target}"
    if not path.exists():
        return None, f"File not found: {ref.target}"
    if not path.is_file():
        return None, f"Not a file: {ref.target}"
    if _is_binary(path):
        return None, f"Binary file skipped: {ref.target}"

    # Size limit: 500KB per file (prevents memory issues with large dumps)
    MAX_FILE_SIZE = 500_000
    try:
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return None, f"File too large: {ref.target} ({file_size//1024}KB, max {MAX_FILE_SIZE//1024}KB)"
    except OSError:
        pass

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return None, f"Read error: {ref.target}: {e}"

    # Extract line range if specified
    if ref.line_start is not None:
        lines = text.split("\n")
        start = max(0, ref.line_start - 1)
        end = (ref.line_end + 1) if ref.line_end else start + 1  # Inclusive end
        text = "\n".join(lines[start:end])

    lang = _LANG_MAP.get(path.suffix, "")
    header = f"**{ref.target}**" + (f" (lines {ref.line_start}-{ref.line_end})" if ref.line_start else "")
    return f"{header}\n```{lang}\n{text}\n```", None


def _expand_folder(ref: ContextReference, cwd: str) -> tuple[str | None, str | None]:
    """List a folder's contents."""
    path = _resolve_path(ref.target, cwd)
    if not path:
        return None, f"Path blocked: {ref.target}"
    if not path.exists() or not path.is_dir():
        return None, f"Directory not found: {ref.target}"

    entries = []
    count = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in sorted(dirs) if not d.startswith(".") and d != "__pycache__"]
        rel = os.path.relpath(root, path)
        if rel == ".":
            rel = ""
        for f in sorted(files):
            if count >= MAX_FOLDER_ENTRIES:
                entries.append(f"... and more ({count}+ files)")
                break
            fpath = os.path.join(rel, f) if rel else f
            entries.append(fpath)
            count += 1
        if count >= MAX_FOLDER_ENTRIES:
            break

    listing = "\n".join(entries)
    return f"**{ref.target}/** ({count} files)\n```\n{listing}\n```", None


def _expand_git_diff(ref: ContextReference, cwd: str) -> tuple[str | None, str | None]:
    """Get git diff or staged changes."""
    cmd = ["git", "diff"]
    if ref.kind == "staged":
        cmd.append("--staged")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=cwd)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "not a git repository" in stderr.lower():
                return None, f"@{ref.kind}: not a git repository"
            return None, f"git diff failed: {stderr.split(chr(10))[0][:100]}"
        diff = result.stdout
        if not diff.strip():
            return f"**@{ref.kind}**: No changes.", None
        if len(diff) > 30_000:
            diff = diff[:30_000] + f"\n[truncated — {len(diff)} chars total]"
        return f"**@{ref.kind}**\n```diff\n{diff}\n```", None
    except subprocess.TimeoutExpired:
        return None, "git diff timed out"
    except FileNotFoundError:
        return None, "git not found"


def _expand_git_log(ref: ContextReference, cwd: str) -> tuple[str | None, str | None]:
    """Get recent git log with patches."""
    try:
        count = min(max(1, int(ref.target)), 10)
    except ValueError:
        return None, f"Invalid git count: {ref.target} (use @git:N where N is 1-10)"

    try:
        result = subprocess.run(
            ["git", "log", f"-{count}", "-p", "--stat"],
            capture_output=True, text=True, timeout=30, cwd=cwd,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "not a git repository" in stderr.lower():
                return None, f"@git:{count}: not a git repository"
            return None, f"git log failed: {stderr.split(chr(10))[0][:100]}"
        output = result.stdout
        if not output.strip():
            return f"**@git:{count}**: No commits found.", None
        if len(output) > 30_000:
            output = output[:30_000] + f"\n[truncated — {len(output)} chars total]"
        return f"**@git:{count}** (last {count} commits)\n```diff\n{output}\n```", None
    except subprocess.TimeoutExpired:
        return None, "git log timed out"
    except FileNotFoundError:
        return None, "git not found"


def _expand_url(ref: ContextReference) -> tuple[str | None, str | None]:
    """Fetch URL content."""
    import urllib.request
    url = ref.target
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mullet/0.2"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        if len(content) > 20_000:
            content = content[:20_000] + f"\n[truncated — {len(content)} chars total]"
        return f"**@url:{ref.target}**\n```\n{content}\n```", None
    except Exception as e:
        return None, f"URL fetch failed: {ref.target}: {e}"


def _resolve_path(target: str, cwd: str) -> Path | None:
    """Resolve a path relative to cwd with security checks."""
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = Path(cwd) / path
    path = path.resolve()

    # Security: block sensitive directories and files
    for part in path.parts:
        if part.lower() in _BLOCKED_DIRS:
            return None
    if path.name.lower() in _BLOCKED_FILES:
        return None
    # Block all .env variants (.env.local, .env.production, etc.)
    if path.name.lower().startswith(".env"):
        return None

    return path


def _is_binary(path: Path) -> bool:
    """Check if a file is binary."""
    # Known text MIME types that don't start with "text/"
    _TEXT_MIMES = {
        "application/json", "application/xml", "application/javascript",
        "application/sql", "application/x-sh", "application/x-yaml",
        "application/toml", "application/graphql",
    }
    mime, _ = mimetypes.guess_type(str(path))
    if mime and not mime.startswith("text/") and mime not in _TEXT_MIMES:
        return True
    # Check first 4KB for null bytes
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        return b"\x00" in chunk
    except Exception as _exc:
        logger.warning("agent.context_refs: failed to check file %s for binary content: %s", path, _exc)
        return True
