# Phase 0 Toolchain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the soundscapy rewrite's toolchain on an empty package, so that `uv run task check` and `uv run task docs` pass and a wheel installs with no compiler.

**Architecture:** One `pyproject.toml` carries every tool's configuration (house style). go-task runs every entry point through `uv run`. prek runs the git hooks. GitHub Actions runs the same `task` targets. great-docs builds the docs from `docs/`. No runtime dependency and no package code beyond `__version__` exist at the end of this plan; M1 slice S1 starts the package.

**Tech Stack:** uv 0.13.6+ with `uv_build`, ruff 0.17.3, ty 0.0.95, pytest 9.2, hypothesis, import-linter 2.15, prek 0.5.9, go-task-bin 3.54, great-docs (git pin `af17c76`), quarto-cli 1.10.20, GitHub Actions.

**Spec:** `design/specs/2026-10-31-phase0-m1-design.md`, Part A (§5) and §11.1, §11.4. Read §5 in full before starting; this plan cites its subsections instead of repeating normative blocks.

## Global Constraints

- uv only: never bare `python`, `pip` or `uv pip` (a shim refuses them); use `uv run`, `uvx`.
- Python floor `>=3.11`; ruff `target-version = "py311"`; ty `python-version = "3.11"` under `[tool.ty.environment]`.
- Build backend `uv_build>=0.13.6,<0.14`; `[tool.uv] required-version = ">=0.13.6"`.
- Every dev and test dependency has a lower bound (spec §11.4).
- The `pyproject.toml` blocks in spec §5.2 and §5.2.1 are normative: copy them verbatim, then add `[tool.ty.src] exclude = ["scripts"]` (spec §11.4).
- Task names in spec §5.3 are normative: AGENTS.md, the Stop hook and CI call them.
- `task check:fast` must stay offline and finish in seconds.
- `filterwarnings = ["error"]`; never weaken a lint or type rule to pass; a narrowly scoped ignore needs a comment with the reason.
- Commit `uv.lock` with every dependency change and say why in the message.
- Commit messages end with the session's attribution lines.
- Work on branch `soundscapy-rewrite`; push there.

## Review Focus

1. **A contributor on Windows without `task` on PATH.** Expected: `uv run task check` works after `uv sync`; the Stop hook falls back to `uv run task check:fast`. Pinned by Task 3's hook test.
2. **A Windows checkout rewriting line endings of hashed data.** Expected: files under `src/soundscapy/_data/` and test fixtures keep their bytes. Pinned by Task 1's `git check-attr` step.
3. **`import soundscapy` printing anything** (a warning, a log line). Expected: silent. Pinned by Task 1's `test_import_is_silent`.
4. **A contributor running plain `uv run pytest -m network`.** Expected: the network test runs; plain `pytest` deselects it. Pinned by Task 3's marker meta-test.
5. **`task docs` reaching the network.** Expected: it builds offline. Pinned by Task 5's offline build step.

---

### Task 1: Package skeleton, project file and lock

**Files:**
- Create: `pyproject.toml` (spec §5.2 block plus §5.2.1 block plus `[tool.ty.src] exclude = ["scripts"]`), `uv.lock`, `src/soundscapy/__init__.py`, `src/soundscapy/py.typed`, `.gitattributes` (spec §5.1.1, verbatim), `tests/test_package.py`
- Modify: `.gitignore` (append the entries listed in spec §5.1)

**Interfaces:**
- Produces: `soundscapy.__version__: str`; `soundscapy.__all__ == ["__version__"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_package.py
import importlib.metadata
import subprocess
import sys

import soundscapy


def test_version_matches_metadata():
    assert soundscapy.__version__ == importlib.metadata.version("soundscapy")


def test_import_is_silent():
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import soundscapy"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
```

- [ ] **Step 2: Create `pyproject.toml` and `.gitattributes`, then `uv sync`**

Run: `uv sync`
Expected: creates `.venv` and `uv.lock`; exit 0. If `go-task-bin` or another pinned floor is not yet on PyPI, use the newest release and note it in the commit message.

- [ ] **Step 3: Run the tests to see them fail**

Run: `uv run pytest tests/test_package.py -v`
Expected: FAIL (no `soundscapy` package).

- [ ] **Step 4: Implement `src/soundscapy/__init__.py`**

A numpy-style module docstring (one summary line, one paragraph); `__version__ = importlib.metadata.version("soundscapy")`; `__all__ = ["__version__"]`. Empty `py.typed`.

- [ ] **Step 5: Run the tests to see them pass, and check line-ending attributes**

Run: `uv run pytest tests/test_package.py -v`
Expected: 2 passed.
Run: `git check-attr text eol -- src/soundscapy/_data/profiles/sal-2026/1.toml pyproject.toml`
Expected: the `_data` path reports `text: unset`; `pyproject.toml` reports `text: set` and `eol: lf`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitattributes .gitignore src tests
git commit -m "build: package skeleton, pyproject and lock (Phase 0)"
```

The message body lists the dev and test dependencies and why each is there (house style).

### Task 2: Taskfile and the lint, type and import gates

**Files:**
- Create: `Taskfile.yml` (spec §5.3, every task name and command, global `env: {MPLBACKEND: Agg}`, a `desc` on every task)
- Test: `tests/test_project_config.py`

**Interfaces:**
- Consumes: Task 1's `pyproject.toml`.
- Produces: the task names of spec §5.3, used by every later task and by CI.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_project_config.py
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_dev_and_test_dependency_has_a_floor():
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]
    for name in ("dev", "test"):
        for requirement in groups[name]:
            assert re.search(r"(>=|==)", requirement), f"{name}: {requirement} has no lower bound"


def test_taskfile_defines_the_normative_tasks():
    text = (ROOT / "Taskfile.yml").read_text()
    for task in ("setup", "fmt", "lint", "typecheck", "test", "test:cov", "test:reference",
                 "test:network", "check:fast", "check", "docs", "docs:papers", "docs:preview",
                 "build", "repo-review"):
        assert re.search(rf"^  {re.escape(task)}:", text, re.M), task
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_project_config.py -v`
Expected: FAIL on the Taskfile test (file missing).

- [ ] **Step 3: Write `Taskfile.yml`**

`version: '3'`. Commands exactly as spec §5.3. `docs:papers` runs fallback step 1 of spec §5.9 for now; Task 5 settles it.

- [ ] **Step 4: Run the gates**

Run: `uv run task check:fast`
Expected: ruff format check, ruff check, `lint-imports` (all contracts kept or skipped as absent) and `ty check` pass. The `.claude/hooks` files may fail ruff: that is Task 3's first step, so run `uv run ruff check src tests` here and expect it clean.
Run: `uv run task check`
Expected: tests pass and coverage reports 100% of `src/soundscapy/__init__.py`.

- [ ] **Step 5: Commit**

```bash
git add Taskfile.yml tests/test_project_config.py
git commit -m "build: Taskfile with the house-style task names (Phase 0)"
```

### Task 3: Hook scripts, Stop-hook fallback and the marker meta-test

**Files:**
- Modify: `.claude/hooks/guard_push.py`, `post_edit.py`, `stop_check.py` (format only), then `stop_check.py` (spec §5.4.1)
- Create: `tests/test_markers.py`, `tests/test_hooks.py`

**Interfaces:**
- Consumes: Task 2's `check:fast` task.

- [ ] **Step 1: Reformat the hook scripts, alone, as its own commit**

Run: `uv run ruff format .claude/hooks && uv run ruff check .claude/hooks`
Expected: 2 files reformatted; the check is clean under the per-file ignore of spec §5.2.

```bash
git add .claude/hooks
git commit -m "style: ruff format hook scripts (one-off, no behaviour change)"
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_markers.py
import subprocess
import sys
import textwrap


def test_later_m_overrides_addopts(tmp_path):
    (tmp_path / "pyproject.toml").write_text(textwrap.dedent("""
        [tool.pytest.ini_options]
        addopts = ["--strict-markers", "-m", "not reference and not network"]
        markers = ["network: needs a download", "reference: slow"]
    """))
    (tmp_path / "test_x.py").write_text(textwrap.dedent("""
        import pytest
        def test_plain(): pass
        @pytest.mark.network
        def test_net(): pass
    """))
    def run(*args):
        return subprocess.run([sys.executable, "-m", "pytest", "-q", *args],
                              cwd=tmp_path, capture_output=True, text=True, check=False).stdout
    assert "1 passed, 1 deselected" in run()
    assert "1 passed, 1 deselected" in run("-m", "network")
```

```python
# tests/test_hooks.py
import os
import shutil
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "stop_check.py"


def test_stop_hook_falls_back_to_uv_run_task(tmp_path, monkeypatch):
    # PATH holds only uv, so `task` is missing and the hook must use `uv run task check:fast`.
    uv_dir = tmp_path / "bin"
    uv_dir.mkdir()
    (uv_dir / "uv").symlink_to(shutil.which("uv"))
    result = subprocess.run([sys.executable, str(HOOK)], input="{}", capture_output=True,
                            text=True, env={**os.environ, "PATH": str(uv_dir)}, cwd=HOOK.parents[2], check=False)
    assert "neither `task` nor `uv`" not in result.stderr
```

Skip `test_stop_hook_falls_back_to_uv_run_task` on Windows with a reason (symlinks need privileges there); the manual run in Step 5 covers it.

- [ ] **Step 3: Run them to see the hook test fail**

Run: `uv run pytest tests/test_markers.py tests/test_hooks.py -v`
Expected: the marker test PASSES (pytest's own behaviour, pinned here); the hook test FAILS (the hook returns early without `task`).

- [ ] **Step 4: Edit `stop_check.py` per spec §5.4.1 and run the tests**

Run: `uv run pytest tests/test_markers.py tests/test_hooks.py -v`
Expected: both pass.

- [ ] **Step 5: Manual check, then commit**

Run: `env PATH="$(dirname "$(command -v uv)")" "$(command -v python3)" .claude/hooks/stop_check.py < /dev/null; echo $?`
Expected: it runs `uv run task check:fast` and exits 0. Paste the output into the commit body.

```bash
git add .claude/hooks/stop_check.py tests/test_markers.py tests/test_hooks.py
git commit -m "fix(hooks): stop hook runs uv run task when task is not on PATH"
```

### Task 4: prek hooks

**Files:**
- Create: `.pre-commit-config.yaml` (spec §5.5, verbatim)

- [ ] **Step 1: Install and run the hooks**

Run: `uv run prek install && uv run prek run --all-files`
Expected: every hook passes (the first run fetches ruff-pre-commit, uv-pre-commit and actionlint-py). Fix whitespace or end-of-file findings in files this plan created; for files it did not create, do not reformat them (AGENTS.md): add a narrowly scoped `exclude` with a comment instead.

- [ ] **Step 2: Check the large-file hook admits the planned data size**

Run: `head -c 400000 /dev/urandom > /tmp/x.bin && uv run prek run check-added-large-files --files /tmp/x.bin; rm /tmp/x.bin`
Expected: passes (400 KB < 600 KB).

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "build: prek hooks (Phase 0)"
```

### Task 5: Docs skeleton and the freeze check

**Files:**
- Create: `docs/great-docs.yml` (spec §5.9, verbatim), `docs/references.bib` (empty), `docs/user_guide/01-install.qmd`, `docs/user_guide/_quickstart.sh`, `_quickstart.ps1`, `_quickstart.py`, `docs/tutorials/00-check-install.qmd`, `docs/papers/index.qmd`, `tests/test_docs_structure.py`
- Modify: `Taskfile.yml` (`docs:papers` once the freeze check settles), `.claude/skills/house-style/reference.md` (record the freeze result)

**Interfaces:**
- Produces: `docs/user_guide/_quickstart.py` (prints `soundscapy.__version__` at Phase 0; S1 replaces it); the `guide-section` values `How-to`, `Explanation`, `Reference`.

- [ ] **Step 1: Write the failing structure test**

```python
# tests/test_docs_structure.py
import re
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "user_guide"
ALLOWED = {"How-to", "Explanation", "Reference"}


def test_every_user_guide_page_has_a_known_guide_section():
    pages = sorted(GUIDE.glob("*.qmd"))
    assert pages
    for page in pages:
        match = re.search(r"^guide-section:\s*\"?([^\"\n]+)\"?\s*$", page.read_text(), re.M)
        assert match and match.group(1) in ALLOWED, page.name
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/test_docs_structure.py -v`
Expected: FAIL (no pages).

- [ ] **Step 3: Write the pages per spec §5.9**

`01-install.qmd` sets `guide-section: "How-to"` and pulls the three `_quickstart.*` files in with `{{< include >}}` inside display blocks. The shell scripts install uv if absent, create a project in the current directory, run `uv add 'soundscapy==1.0.0.dev1' jupyterlab`, then `uv run python _quickstart.py`; they never launch JupyterLab (spec §11.4). Before the first release the scripts accept a `SOUNDSCAPY_SPEC` environment variable overriding the requirement (the nightly job sets it to the git URL, spec §5.6.2).

- [ ] **Step 4: Run the structure test, then build offline**

Run: `uv run pytest tests/test_docs_structure.py -v`
Expected: PASS.
Run: `uv sync --group docs && unshare -rn uv run --offline task docs || uv run --offline task docs`
Expected: the site builds in `docs/_site/` with every cell executed and no network use (if `unshare` is unavailable, the `--offline` run is the check). If the build fails without Quarto, confirm `uv run --group docs quarto --version` prints 1.10.20.

- [ ] **Step 5: Run the freeze check (spec §5.9)**

Create a throwaway `docs/papers/_freeze_probe.qmd` with `freeze: true` and a cell printing a timestamp; build once; run fallback step 1, then step 2 if needed. Record in `reference.md` which step re-executed the page, naming commit `af17c76`. Set `docs:papers` in the Taskfile to that step. Delete the probe and its `_freeze` entry.

- [ ] **Step 6: Commit**

```bash
git add docs tests/test_docs_structure.py Taskfile.yml .claude/skills/house-style/reference.md
git commit -m "docs: great-docs skeleton, install page and freeze check (Phase 0)"
```

### Task 6: Governance, citation, README and agent docs

**Files:**
- Create: `GOVERNANCE.md`, `CITATION.cff`, `CONTRIBUTING.md`, `.github/CODEOWNERS` (spec §5.1, §5.8)
- Modify: `README.md`, `AGENTS.md` (Status paragraph and the `uv run task` line), `.claude/skills/house-style/SKILL.md` (every edit in spec §11.1's house-style row), `.claude/skills/great-docs/`, `write-user-guide/`, `revise-docstrings/` (re-copied at `af17c76`), `.claude/skills/VENDORED.md`

- [ ] **Step 1: Write the governance files and README per spec §5.8 and §5.1**

CODEOWNERS lines: `/src/soundscapy/_data/ @MitchellAcoustics`, `/tests/science/ @MitchellAcoustics`, `/tests/regression/ @MitchellAcoustics`, `/design/ @MitchellAcoustics`.

- [ ] **Step 2: Validate the citation file**

Run: `uvx cffconvert@2.0.0 --validate`
Expected: "Citation metadata are valid".

- [ ] **Step 3: Re-copy the vendored great-docs skills at `af17c76`**

Run: `git clone --filter=blob:none https://github.com/posit-dev/great-docs /tmp/gd && git -C /tmp/gd checkout af17c76d6ab2c775e438c6804c3b1b6feceef57e` then copy `docs/skills/great-docs`, `write-user-guide`, `revise-docstrings` over the vendored directories; bump the three `VENDORED.md` rows to `af17c76d6ab2c775e438c6804c3b1b6feceef57e`.
Expected: `git diff --stat .claude/skills` shows only those three directories and `VENDORED.md`.

- [ ] **Step 4: Apply the house-style edits and the AGENTS.md Status rewrite**

House-style: every item in spec §11.1's house-style and reference.md rows. AGENTS.md: the Status paragraph says the toolchain, CI and docs skeleton exist and `task check` is the definition of done; add "If `task` is not on PATH, run `uv run task <name>`; go-task comes with the dev group."

- [ ] **Step 5: Run the gate and commit**

Run: `uv run task check`
Expected: PASS.

```bash
git add GOVERNANCE.md CITATION.cff CONTRIBUTING.md .github/CODEOWNERS README.md AGENTS.md .claude/skills
git commit -m "docs: governance, citation, contributing; re-vendor great-docs skills at af17c76"
```

### Task 7: CI workflows

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/nightly.yml`, `.github/workflows/release.yml` (spec §5.6)

- [ ] **Step 1: Resolve the checkout action SHA**

Run: `git ls-remote https://github.com/actions/checkout refs/tags/v7.0.1`
Expected: one SHA; pin `actions/checkout@<sha> # v7.0.1`. setup-uv is `astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2 # v10.3.3`.

- [ ] **Step 2: Write the three workflows per spec §5.6.1 to §5.6.3**

Jobs and matrices exactly as the spec's tables: `check`, `lowest` (job-level `UV_RESOLUTION: lowest-direct`), `docs`, `citation`, `install` (wheel step with `shell: bash`), `install-win-arm` (`continue-on-error: true`); nightly `reference`, `network`, `docs:papers` with the freeze-diff issue step (`permissions: {contents: read, issues: write}`) and the `quickstart` job; `release.yml` on tags `v*` with environment `pypi` and `id-token: write`. Every job `permissions: contents: read` unless stated.

- [ ] **Step 3: Lint the workflows**

Run: `uv run prek run actionlint --all-files`
Expected: passes.

- [ ] **Step 4: Commit, push, and watch the run**

```bash
git add .github/workflows
git commit -m "ci: check, lowest, docs, citation and install jobs; nightly; release (Phase 0)"
git push -u origin soundscapy-rewrite
```

Expected: the `ci.yml` run on GitHub is green for `check`, `docs`, `citation`, `install` and `lowest`. Fix any red job in a follow-up commit; never skip or disable a job to get green.

### Task 8: repo-review and the Phase 0 exit

**Files:**
- Modify: `pyproject.toml` (`[tool.repo-review] ignore`, each id with a reason comment), `.claude/skills/house-style/SKILL.md` (the same reasons)

- [ ] **Step 1: Run repo-review**

Run: `uv run task repo-review`
Expected: a list of failing checks.

- [ ] **Step 2: Fix what fits house style; ignore the rest with reasons**

For each failing check: fix it if house style agrees; otherwise add its id to `[tool.repo-review] ignore` with a trailing comment and the same reason in the house-style skill.

- [ ] **Step 3: Verify the exit criteria of spec §5.10**

Run: `uv run task repo-review; echo $?` → `0`. Run: `uv run task check` → PASS. Confirm on GitHub that `ci.yml` is green on the pushed head. Each remaining §5.10 row is shown by a file or test from Tasks 1 to 7.

Evidence (recorded at close-out): `tests/test_hooks.py` stands in for a manual Stop-hook run (it runs `.claude/hooks/stop_check.py` with `uv` but no `task` on PATH, and with neither). The `uv run task repo-review; echo $?` output showing exit 0 is recorded in the body of the Phase 0 close-out commit.

- [ ] **Step 4: Commit and push**

```bash
git add pyproject.toml .claude/skills/house-style/SKILL.md
git commit -m "build: repo-review clean with justified ignores (Phase 0 exit)"
git push origin soundscapy-rewrite
```

---

## After the plan

Run `/simplify` on the Phase 0 changes, re-run `uv run task check`, then `/code-review` and address the findings (AGENTS.md Review). Then write the plan for M1 slice S1 from spec §9 and §6.
