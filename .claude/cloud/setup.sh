#!/usr/bin/env bash
# Cloud environment setup script for soundscapy.
#
# Claude Code cloud sessions do not install the plugins that .claude/settings.json
# enables (https://code.claude.com/docs/en/cloud-environments). This script installs
# them before Claude Code starts. Paste it into the environment's Setup script field
# (cloud environment menu in the session title bar, then Edit). Keep the plugin list
# in step with enabledPlugins in .claude/settings.json.
#
# Plugins are installed at user scope so the repository's settings file is not
# rewritten. The cloud image pre-installs code-review, pr-review-toolkit and
# skill-creator; settings.json switches those off, because built-in commands and
# vendored copies replace them. A failed install is reported but does not fail the session.
set -uo pipefail

# pyproject.toml sets [tool.uv] required-version = ">=0.13.6". With an older uv on PATH
# every `uv run` in the repository fails, including the hooks that guard each Bash call,
# so the session cannot run commands at all. The cloud image has shipped uv 0.11.32.
# `uv self update` fails on a uv that was not installed standalone, so install a new uv
# as a uv tool instead. It goes to uv's tool bin directory (~/.local/bin by default), which
# is on PATH in the cloud image; this is what worked on 2026-10-31.
uv_min=0.13.6
uv_current=$(uv --version 2>/dev/null | awk '{print $2}')
if [ -z "$uv_current" ] || [ "$(printf '%s\n%s\n' "$uv_min" "$uv_current" | sort -V | head -n1)" != "$uv_min" ]; then
  if uv tool install --force "uv>=${uv_min}" >/dev/null 2>&1 && hash -r; then
    echo "installed uv $(uv --version 2>/dev/null | awk '{print $2}') (was ${uv_current:-missing})"
  else
    echo "FAILED to install uv>=${uv_min} (found ${uv_current:-none}); uv run will refuse this project" >&2
  fi
fi

marketplaces=(
  anthropics/claude-plugins-official
  trailofbits/skills
  astral-sh/claude-code-plugins
)
plugins=(
  modern-python@trailofbits
  astral@astral-sh
  superpowers@claude-plugins-official
)

failed=0
for m in "${marketplaces[@]}"; do
  claude plugin marketplace add "$m" >/dev/null 2>&1 || claude plugin marketplace update >/dev/null 2>&1 || true
done
for p in "${plugins[@]}"; do
  if claude plugin install "$p" --scope user -y >/dev/null 2>&1; then
    echo "installed $p"
  else
    echo "FAILED $p" >&2
    failed=1
  fi
done
[ "$failed" -eq 0 ] || echo "setup: some plugins failed to install; see above" >&2
exit 0
