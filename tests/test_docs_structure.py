"""Structure checks for the docs sources (FR-12.3's four-part structure) and docstrings."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "user_guide"
ALLOWED = {"How-to", "Explanation", "Reference"}
# Any Sphinx role, including the forms great-docs' interlinks filter parses (:func:`x`,
# :py:func:`x`, :external+numpy:py:class:`x`). Scoped to src/: design/ quotes pip's
# ``--only-binary :all:``, which a pattern this broad would also match.
SPHINX_ROLE = re.compile(r":[a-z]+(?:[+:][a-z]+)*:`")


def test_every_user_guide_page_has_a_known_guide_section():
    pages = sorted(GUIDE.glob("*.qmd"))
    assert pages
    for page in pages:
        match = re.search(r"^guide-section:\s*\"?([^\"\n]+)\"?\s*$", page.read_text(), re.MULTILINE)
        assert match, page.name
        assert match.group(1) in ALLOWED, page.name


def test_install_page_pins_the_release_and_includes_the_quickstart_files():
    text = (GUIDE / "01-install.qmd").read_text()
    assert "soundscapy==1.0.0.dev1" in text
    assert "%pip install 'soundscapy==1.0.0.dev1'" in text
    for name in ("_quickstart.py", "_quickstart.sh", "_quickstart.ps1"):
        # great-docs expands a code-file include into its own fenced block.
        assert re.search(rf"^\{{\{{< include {re.escape(name)}[ >]", text, re.MULTILINE), name


def test_quickstart_is_the_first_plot():
    text = (GUIDE / "_quickstart.py").read_text()
    for fragment in ("load_isd(subset=True)", 'angles="equal"', 'by="location"'):
        assert fragment in text, fragment
    assert 'savefig("first-plot.png")' in text


def test_install_page_says_the_notebook_cell_shows_the_plot():
    # The notebook cells are the script without pyplot's import and the savefig line: a
    # notebook shows the pyplot figure itself. The prose must not call them "the same code".
    text = (GUIDE / "01-install.qmd").read_text()
    script = (GUIDE / "_quickstart.py").read_text()
    notebook = script.replace("import matplotlib.pyplot as plt\n\n", "").replace(
        'plt.savefig("first-plot.png")\n', ""
    )
    assert notebook != script
    assert "the same code" not in text
    assert text.count(f"```python\n{notebook}```") == 2
    assert text.count("shows the plot instead of saving it") == 2


def test_quickstart_runs(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(GUIDE / "_quickstart.py")],
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
    )
    # stderr is not asserted: matplotlib's first-run font-cache message may appear there.
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert (tmp_path / "first-plot.png").exists()


def test_docstrings_use_no_sphinx_roles():
    # Under great-docs' numpy parser a role is not converted: ":func:" is printed as text,
    # also in page descriptions and llms.txt, and the build gives no warning. Write a code
    # span instead (house-style reference.md, "Cross-references").
    sources = sorted((ROOT / "src" / "soundscapy").rglob("*.py"))
    assert sources
    hits = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in sources
        # Explicit UTF-8: Windows CI would otherwise decode with the locale code page, which
        # fails on characters such as Greek rho, whose UTF-8 bytes cp1252 leaves undefined.
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if SPHINX_ROLE.search(line)
    ]
    assert not hits, "\n".join(hits)


def test_sphinx_role_pattern_matches_roles_and_nothing_else():
    # Without this, a broken pattern would let the guard above pass on any source tree.
    roles = (
        ":func:`read_survey`",
        ":class:`~soundscapy.Survey`",
        ":py:exc:`ValueError`",
        ":external+numpy:py:class:`numpy.ndarray`",
        "[](:func:`read_survey`)",
    )
    others = (
        "``read_survey()``",
        "`Survey`",
        "[](`~soundscapy.read_survey`)",
        "pip install --only-binary :all: soundscapy",
        '{"a":`b`}',
    )
    assert all(SPHINX_ROLE.search(text) for text in roles)
    assert not any(SPHINX_ROLE.search(text) for text in others)
