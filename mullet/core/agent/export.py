"""Conversation export — save agent conversations to shareable formats.

Export formats:
- JSON: full conversation with metadata
- Markdown: human-readable with tool call formatting
"""

import json
import time
from pathlib import Path
from mullet.core.agent.types import AgentResult


def export_json(result: AgentResult, task: str, output_path: str | None = None) -> str:
    """Export conversation as JSON."""
    data = {
        "mullet_version": __import__("mullet").__version__,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": task,
        "completed": result.completed,
        "provider": result.provider_used,
        "tier": result.tier_used,
        "tokens": {
            "input": result.tokens_used.input_tokens,
            "output": result.tokens_used.output_tokens,
            "total": result.tokens_used.total_tokens,
        },
        "tool_calls": result.tool_calls_made,
        "iterations": result.iterations_used,
        "messages": result.messages,
        "final_response": result.final_response,
    }

    json_str = json.dumps(data, indent=2, default=str)

    if output_path:
        Path(output_path).write_text(json_str)

    return json_str


def export_markdown(result: AgentResult, task: str, output_path: str | None = None) -> str:
    """Export conversation as human-readable Markdown."""
    lines = [
        "# Mullet Conversation Export",
        "",
        f"**Task:** {task}",
        f"**Provider:** {result.provider_used} ({result.tier_used or '?'})",
        f"**Status:** {'Completed' if result.completed else 'Incomplete'}",
        f"**Tokens:** {result.tokens_used.total_tokens:,}",
        f"**Tool calls:** {result.tool_calls_made}",
        f"**Iterations:** {result.iterations_used}",
        "",
        "---",
        "",
    ]

    for msg in result.messages:
        role = msg.get("role", "?")

        if role == "system":
            lines.append("### System Prompt")
            content = msg.get("content", "")
            if len(content) > 200:
                lines.append(f"```\n{content[:200]}...\n```")
            else:
                lines.append(f"```\n{content}\n```")
            lines.append("")

        elif role == "user":
            lines.append("### User")
            lines.append(msg.get("content", ""))
            lines.append("")

        elif role == "assistant":
            lines.append("### Assistant")
            content = msg.get("content", "")
            if content:
                lines.append(content)

            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("function", {}).get("name", "?")
                    args = tc.get("function", {}).get("arguments", "")
                    lines.append(f"\n**Tool call: `{name}`**")
                    if len(args) > 200:
                        lines.append(f"```json\n{args[:200]}...\n```")
                    else:
                        lines.append(f"```json\n{args}\n```")
            lines.append("")

        elif role == "tool":
            lines.append("### Tool Result")
            content = msg.get("content", "")
            if len(content) > 500:
                lines.append(f"```\n{content[:500]}...\n```")
            else:
                lines.append(f"```\n{content}\n```")
            lines.append("")

    lines.extend([
        "---",
        "",
        "### Final Response",
        "",
        result.final_response,
    ])

    md = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(md)

    return md
