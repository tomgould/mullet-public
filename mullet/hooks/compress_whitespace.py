"""Claude Code UserPromptSubmit hook: compress redundant whitespace in prompts.

Collapses runs of 2+ spaces into one within prose sections of the prompt.
Code blocks (triple-backtick fences), leading line indentation, and markdown
table lines are left untouched.

Outputs {"prompt": "<compressed>"} to stdout so Claude Code replaces the
original prompt text. Exits 0 unconditionally — never blocks a prompt.
Run as `mullet hook compress-whitespace` (UserPromptSubmit).
"""

import json
import re
import sys
import time
from pathlib import Path

CODE_FENCE = re.compile(r"(```[\s\S]*?```)", re.DOTALL)
TABLE_LINE = re.compile(r"^\s*\|")
MULTI_SPACE = re.compile(r" {2,}")


def _compress_prose(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        if TABLE_LINE.match(line):
            out.append(line)
            continue
        stripped = line.lstrip(" ")
        indent = line[: len(line) - len(stripped)]
        stripped = MULTI_SPACE.sub(" ", stripped)
        out.append(indent + stripped)
    return "".join(out)


_FENCE_MARKER = re.compile(r"(?:^|\n)```")


def compress(prompt: str) -> str:
    # Guard against unclosed fences: if there's an odd number of ``` markers
    # at line boundaries, skip compression to avoid corrupting code content.
    if len(_FENCE_MARKER.findall(prompt)) % 2 != 0:
        return prompt
    # Split on triple-backtick fences; odd indices are code blocks.
    parts = CODE_FENCE.split(prompt)
    for i, part in enumerate(parts):
        if i % 2 == 0:  # prose segment
            parts[i] = _compress_prose(part)
    return "".join(parts)


def main() -> None:
    hook_start = time.time()
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
        prompt = data.get("prompt", data.get("user_prompt", ""))

        if not prompt:
            sys.exit(0)

        compressed = compress(prompt)
        saved = len(prompt) - len(compressed)

        if saved > 0:
            elapsed_ms = int((time.time() - hook_start) * 1000)
            print(
                f"[hook:{Path(__file__).stem}] saved {saved} chars ({elapsed_ms}ms)",
                file=sys.stderr,
            )
            print(json.dumps({"prompt": compressed}))

    except Exception as exc:
        print(f"[hook:compress_whitespace] error: {exc}", file=sys.stderr)
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
