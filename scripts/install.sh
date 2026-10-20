#!/usr/bin/env bash
# Mullet one-liner installer
# Usage: curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash
#
# Installs from the GitHub repository https://github.com/tomgould/mullet-public
# into its own virtualenv. It never installs from PyPI: the "mullet" package
# there is an unrelated project. Run it again to update.
set -euo pipefail
REPO_URL="https://github.com/tomgould/mullet-public.git"
echo "Installing Mullet..."
command -v git >/dev/null 2>&1 || { echo "Error: git is required. Install it first (e.g. apt install git / xcode-select --install)." >&2; exit 1; }
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
    echo "Updating existing source clone in $SRC_DIR..."
    git -C "$SRC_DIR" pull --quiet --ff-only
else
    echo "Cloning $REPO_URL into $SRC_DIR..."
    git clone --quiet "$REPO_URL" "$SRC_DIR"
fi
# Editable install: the Claude Code hooks run from the clone's scripts/ folder.
"$VENV_DIR/bin/pip" install --quiet -e "$SRC_DIR"
mkdir -p "$HOME/.local/bin"
for bin in mullet mullet-mcp mullet-hud mullet-statusline mullet-onboard; do
    src="$VENV_DIR/bin/$bin"
    [ -f "$src" ] && ln -sf "$src" "$HOME/.local/bin/$bin"
done
echo "Success! Mullet installed."
echo "Run \`mullet doctor\` to verify your install, then \`mullet mcp\` to start the MCP server."
