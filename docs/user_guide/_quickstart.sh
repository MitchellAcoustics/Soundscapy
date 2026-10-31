#!/usr/bin/env sh
# Quick start for macOS and Linux: install uv if needed, create a project in the
# current directory, add soundscapy and JupyterLab, then run the first-plot script.
# SOUNDSCAPY_SPEC overrides the requirement (used by CI before the first release).
set -eu

SPEC="${SOUNDSCAPY_SPEC:-soundscapy==1.0.0.dev1}"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
    export PATH
fi

uv init --bare
uv add "$SPEC" jupyterlab
uv run python _quickstart.py
