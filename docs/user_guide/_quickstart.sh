#!/usr/bin/env sh
# Quick start for macOS and Linux: install uv if needed, create a project in the
# current directory, add soundscapy and JupyterLab, then run _quickstart.py.
# SOUNDSCAPY_SPEC overrides the requirement (used by CI before the first release).
set -eu

SPEC="${SOUNDSCAPY_SPEC:-soundscapy==1.0.0.dev1}"

if [ ! -f _quickstart.py ]; then
    echo "_quickstart.py not found: save the code from the install page as _quickstart.py in this folder, then run this again." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl not found: install curl, or install uv yourself (https://docs.astral.sh/uv/), then run this again." >&2
        exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
    export PATH
fi

[ -f pyproject.toml ] || uv init --bare
uv add "$SPEC" jupyterlab
uv run python _quickstart.py
