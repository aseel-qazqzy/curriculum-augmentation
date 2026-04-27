#!/usr/bin/env bash
# Auto-format a Python file with ruff (preferred) or black (fallback).
# Usage: format_py.sh <file_path>
set -euo pipefail

FILE="$1"
[[ "$FILE" == *.py ]] || exit 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RUFF="$PROJECT_DIR/venv/bin/ruff"
BLACK="/Library/Frameworks/Python.framework/Versions/3.12/bin/black"

if [ -x "$RUFF" ]; then
    "$RUFF" format --quiet "$FILE"
elif [ -x "$BLACK" ]; then
    "$BLACK" --quiet "$FILE"
fi
