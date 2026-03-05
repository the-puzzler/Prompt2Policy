#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv venv "${VENV_DIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Install project + dev tooling into the virtual environment.
uv pip install -e ".[dev]"

echo "Environment ready at ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
