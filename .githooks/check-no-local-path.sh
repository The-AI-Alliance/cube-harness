#!/bin/sh
# Block commits that leave a local path = "..." source in any pyproject.toml.
if git diff --cached -- '*pyproject.toml' | grep -qE '^\+\s*path\s*=\s*"'; then
    echo "❌ A pyproject.toml contains a local path = \"...\" source."
    echo "   Local paths break for reviewers with a different folder structure."
    echo ""
    echo "   Instead:"
    echo "   1. Revert the path = source to the PyPI/git version."
    echo "   2. Add a Depends-on: line to your PR description:"
    echo "      Depends-on: cube-standard/<branch-name>"
    echo "   3. Reviewers run: make review PR=<n>"
    exit 1
fi
