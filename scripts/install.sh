#!/usr/bin/env bash
# Mullet one-liner installer
# Usage: curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash
set -euo pipefail
echo "Installing Mullet..."
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$OS" in linux|darwin) ;; *) echo "Error: unsupported OS '$OS' — Linux or macOS required." >&2; exit 1 ;; esac
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        ver=$($cmd -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        major=${ver%%.*} minor=${ver#*.}
        if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; }; then
            PYTHON_CMD=$cmd; break
        fi
    fi
done
[ -z "$PYTHON_CMD" ] && { echo "Error: Python >= 3.10 is required. Install it first (e.g. apt install python3.10 / brew install python@3.10)." >&2; exit 1; }
VENV_DIR="$HOME/.local/share/mullet-venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi
SRC_DIR="$HOME/.local/share/mullet-src"
if [ -d "$SRC_DIR/.git" ]; then
    echo "Updating existing source clone..."
    git -C "$SRC_DIR" pull --quiet
    "$VENV_DIR/bin/pip" install --quiet -e "$SRC_DIR"
else
    if "$VENV_DIR/bin/pip" install --quiet mullet 2>/dev/null; then
        echo "Installed mullet from PyPI."
    else
        echo "Cloning source repo..."
        git clone --quiet https://github.com/tomgould/mullet-public.git "$SRC_DIR"
        "$VENV_DIR/bin/pip" install --quiet -e "$SRC_DIR"
    fi
fi
mkdir -p "$HOME/.local/bin"
for bin in mullet mullet-mcp mullet-hud mullet-statusline; do
    src="$VENV_DIR/bin/$bin"
    [ -f "$src" ] && ln -sf "$src" "$HOME/.local/bin/$bin"
done
echo "Success! Mullet installed."
echo "Run \`mullet doctor\` to verify your install, then \`mullet mcp\` to start the MCP server."
