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
# rewritten. A failed install is reported but does not fail the session.
set -uo pipefail

marketplaces=(
  anthropics/claude-plugins-official
  trailofbits/skills
  astral-sh/claude-code-plugins
  posit-dev/skills
)
plugins=(
  modern-python@trailofbits
  astral@astral-sh
  posit-dev@posit-dev-skills
  github@posit-dev-skills
  quarto@posit-dev-skills
  code-review@claude-plugins-official
  pr-review-toolkit@claude-plugins-official
  superpowers@claude-plugins-official
  skill-creator@claude-plugins-official
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
