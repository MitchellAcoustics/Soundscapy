#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""PreToolUse hook: block ``git push`` from agents in local sessions only.

Cloud sessions (``CLAUDE_CODE_REMOTE=true``) push their own branches as part of
their workflow, so they are allowed. On a developer's machine the developer
pushes, or approves a push, so the agent's Bash call is denied with a reason.
"""

from __future__ import annotations

import json
import os
import re
import sys

# `git push`, including forms such as `git -C repo push` and `git -c k=v push`.
GIT_PUSH = re.compile(r"\bgit\b[^;&|\n]*\bpush\b")


def main() -> int:
    if os.environ.get("CLAUDE_CODE_REMOTE") == "true":
        return 0
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    if not GIT_PUSH.search(command):
        return 0
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Agents do not push from local sessions in this repository. "
                    "Ask the user to push, or to approve this push."
                ),
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
