#!/bin/bash
# Release helper: bump version, tag and build.
# Mullet is distributed from GitHub, not PyPI: the "mullet" name on PyPI
# belongs to an unrelated project, so don't upload these builds there.
set -e
NEW_VERSION="$1"
[ -z "$NEW_VERSION" ] && { echo "usage: $0 <version>"; exit 1; }

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
