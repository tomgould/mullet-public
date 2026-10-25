#!/usr/bin/env bash
# Mullet installer for Linux and macOS.
# Usage: curl -fsSL https://raw.githubusercontent.com/tomgould/mullet-public/main/scripts/install.sh | bash
#
# Installs Mullet from the public GitHub repository as a tool of its own, with
# `uv tool install` when uv is available and `pipx install` otherwise. The
# install is not editable, so Mullet never runs from a tree it may be editing.
# It never installs from PyPI: the "mullet" package there is an unrelated
# project. Run it again to update.
#
# Optional settings:
#   MULLET_REF=<branch or tag>   install that ref instead of the default branch
#   MULLET_EXTRAS=gemini,...     install extras (gemini, bedrock, postgres, all)
#   MULLET_FORCE=1               replace a `mullet` command installed some other way
set -euo pipefail

REPO_URL="https://github.com/tomgould/mullet-public.git"

fail() { echo "Error: $*" >&2; exit 1; }

case "$(uname -s)" in
    Linux|Darwin) ;;
    *) fail "Mullet supports Linux and macOS only (on Windows, use WSL)." ;;
esac
command -v git >/dev/null 2>&1 || fail "git is required. Install it first (apt install git, or xcode-select --install)."

SPEC="git+${REPO_URL}"
[ -n "${MULLET_REF:-}" ] && SPEC="${SPEC}@${MULLET_REF}"
[ -n "${MULLET_EXTRAS:-}" ] && SPEC="mullet[${MULLET_EXTRAS}] @ ${SPEC}"

# An older version of this script made a virtualenv and linked its commands
# into ~/.local/bin; drop those links so the new install can take their place.
OLD_VENV="$HOME/.local/share/mullet-venv"
for bin in mullet mullet-mcp mullet-hud mullet-statusline mullet-onboard; do
    link="$HOME/.local/bin/$bin"
    if [ -L "$link" ] && case "$(readlink "$link")" in "$OLD_VENV"/*) true ;; *) false ;; esac; then
        rm -f "$link"
        echo "Removed the old installer's link $link"
    fi
done

FORCE=()
[ "${MULLET_FORCE:-}" = "1" ] && FORCE=(--force)

IN_THE_WAY="If another mullet command is in the way, run this again with MULLET_FORCE=1."
if command -v uv >/dev/null 2>&1; then
    echo "Installing Mullet from $SPEC ..."
    # --reinstall-package fetches the latest commit when Mullet is already installed.
    uv tool install ${FORCE[@]+"${FORCE[@]}"} --python 3.12 --reinstall-package mullet "$SPEC" \
        || fail "uv could not install Mullet (see above). $IN_THE_WAY"
    BIN_DIR="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
    PATH_HINT="uv tool update-shell"
elif command -v pipx >/dev/null 2>&1; then
    echo "Installing Mullet from $SPEC ..."
    # pipx leaves an installed package alone unless forced, so an update forces.
    if pipx list --short 2>/dev/null | grep -q '^mullet '; then
        FORCE=(--force)
    fi
    pipx install ${FORCE[@]+"${FORCE[@]}"} "$SPEC" \
        || fail "pipx could not install Mullet (see above). $IN_THE_WAY"
    BIN_DIR="${PIPX_BIN_DIR:-$HOME/.local/bin}"
    PATH_HINT="pipx ensurepath"
else
    fail "this installer needs uv or pipx. Install uv (https://docs.astral.sh/uv/) or pipx (https://pipx.pypa.io/), then run it again."
fi

echo
echo "Mullet is installed."
if ! command -v mullet >/dev/null 2>&1; then
    echo "$BIN_DIR is not on your PATH yet: run \`$PATH_HINT\`, then open a new terminal."
fi
if [ -d "$OLD_VENV" ] || [ -d "$HOME/.local/share/mullet-src" ]; then
    echo "The old installer's $OLD_VENV and $HOME/.local/share/mullet-src are no longer used; delete them when you like."
fi
cat <<'EOF'

Next steps:
  mullet doctor          check Python, Claude Code, git and the job sandbox
  mullet setup-claude    register Mullet with Claude Code (--dry-run shows the changes first)
EOF
