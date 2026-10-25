#!/usr/bin/env bash
# Release helper: bump version, tag and build.
# Mullet is distributed from GitHub, not PyPI: the "mullet" name on PyPI
# belongs to an unrelated project, so don't upload these builds there.
#
# Usage: scripts/release.sh [--dry-run] <version>
#   --dry-run   validate the version and print the steps without changing anything
set -euo pipefail

usage() { echo "usage: $0 [--dry-run] <version>  (for example 2.0.1 or 2.1.0-rc1)" >&2; }

DRY_RUN=0
NEW_VERSION=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            if [ -n "$NEW_VERSION" ]; then usage; exit 1; fi
            NEW_VERSION="$arg" ;;
    esac
done
[ -n "$NEW_VERSION" ] || { usage; exit 1; }

if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.]+)?$ ]]; then
    echo "Error: invalid version '$NEW_VERSION': expected X.Y.Z or X.Y.Z-suffix." >&2
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "Would set version $NEW_VERSION in pyproject.toml and mullet/__init__.py"
    echo "Would run: python3 -m build"
    echo "Would commit \"chore(release): v$NEW_VERSION\" and tag v$NEW_VERSION"
    exit 0
fi

# Update version in pyproject.toml + mullet/__init__.py
sed -i "s/^version = .*/version = \"$NEW_VERSION\"/" pyproject.toml
sed -i "s/__version__ = .*/__version__ = \"$NEW_VERSION\"/" mullet/__init__.py

# Build
python3 -m build

# Tag
git add pyproject.toml mullet/__init__.py
git commit -m "chore(release): v$NEW_VERSION"
git tag "v$NEW_VERSION"

echo "Built dist/mullet-$NEW_VERSION-*.whl"
echo "Next: git push && git push --tags"
