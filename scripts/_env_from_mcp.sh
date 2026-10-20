#!/usr/bin/env bash
# Exports API keys from .mcp.json into the current shell so `mullet --provider ...`
# can reach Kimi / Gemini without the user having to pre-export in their session.
#
# Safe to source from any Bash call: no keys are emitted to stdout, and the
# file it reads (.mcp.json) is the same place Claude Code MCP servers read
# them from.
#
# Usage: source scripts/_env_from_mcp.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(python3 -c "
import json, sys
try:
    with open('${PROJECT_ROOT}/.mcp.json') as f:
        env = json.load(f).get('mcpServers', {}).get('mullet', {}).get('env', {})
except Exception as e:
    print(f'echo \"[_env_from_mcp] {e}\" >&2', file=sys.stderr)
    sys.exit(0)
for k in ('KIMI_API_KEY', 'MOONSHOT_API_KEY', 'GOOGLE_API_KEY', 'OPENROUTER_API_KEY'):
    if k in env and env[k]:
        print(f'export {k}={env[k]!r}')
")"

# Positive signal for manual_tool_guard that MCP is configured for this session.
# (Claude Code may still not have the MCP server registered; the guard also
# checks recent history. Setting this avoids the Agent-fallback message when
# MCP is actually available.)
export MULLET_MCP_ENABLED=1
