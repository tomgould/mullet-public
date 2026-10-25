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
# Options:
#   --version X.Y.Z   install the release tag vX.Y.Z
#   --ref <ref>       install a branch, tag or commit (default: the default branch)
#   --dry-run         print the install command without running it
#   -h, --help        show this help
# With curl: curl -fsSL <url> | bash -s -- --version X.Y.Z
#
# Optional settings:
#   MULLET_REF=<branch, tag or sha>   same as --ref (the option wins)
#   MULLET_EXTRAS=gemini,...     install extras (gemini, bedrock, postgres, all)
#   MULLET_FORCE=1               replace a `mullet` command installed some other way
set -euo pipefail

REPO_URL="https://github.com/tomgould/mullet-public.git"

fail() { echo "Error: $*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage: install.sh [--version X.Y.Z | --ref <branch|tag|sha>] [--dry-run]

  --version X.Y.Z   install the release tag vX.Y.Z
  --ref <ref>       install a branch, tag or commit (default: the default branch)
  --dry-run         print the install command without running it
  -h, --help        show this help
USAGE
}

valid_ref() {
    case "$1" in
        ""|-*|*[!A-Za-z0-9._/-]*) return 1 ;;
    esac
}

REF="${MULLET_REF:-}"
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --version)
            [ $# -ge 2 ] || fail "--version needs a value such as 2.0.0."
            case "$2" in
                [0-9]*.[0-9]*.[0-9]*) ;;
                *) fail "invalid version '$2': expected X.Y.Z." ;;
            esac
            valid_ref "$2" || fail "invalid version '$2': use only letters, digits and . _ / -"
            REF="v$2"; shift 2 ;;
        --ref)
            [ $# -ge 2 ] || fail "--ref needs a value."
            valid_ref "$2" || fail "invalid ref '$2': use only letters, digits and . _ / - and do not start with -."
            REF="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; fail "unknown argument '$1'." ;;
    esac
done
if [ -n "$REF" ]; then
    valid_ref "$REF" || fail "invalid ref '$REF': use only letters, digits and . _ / - and do not start with -."
fi

case "$(uname -s)" in
    Linux|Darwin) ;;
    *) fail "Mullet supports Linux and macOS only (on Windows, use WSL)." ;;
esac
command -v git >/dev/null 2>&1 || fail "git is required. Install it first (apt install git, or xcode-select --install)."

SPEC="git+${REPO_URL}"
[ -n "$REF" ] && SPEC="${SPEC}@${REF}"
if [ -n "$REF" ]; then SHOWN_REF="$REF"; else SHOWN_REF="the repository default branch"; fi
[ -n "${MULLET_EXTRAS:-}" ] && SPEC="mullet[${MULLET_EXTRAS}] @ ${SPEC}"

if [ "$DRY_RUN" = "1" ]; then
    if command -v uv >/dev/null 2>&1; then
        echo "Would run: uv tool install --python 3.12 --reinstall-package mullet \"$SPEC\""
    elif command -v pipx >/dev/null 2>&1; then
        echo "Would run: pipx install \"$SPEC\""
    else
        echo "Would run: uv tool install or pipx install of \"$SPEC\" (neither is installed)"
    fi
    echo "Ref: $SHOWN_REF"
    exit 0
fi

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
echo "Mullet is installed (ref: $SHOWN_REF)."
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
