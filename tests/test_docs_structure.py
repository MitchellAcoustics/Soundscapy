"""Structure checks for the docs sources (FR-12.3's four-part structure)."""

import os
import re
import subprocess
import sys
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "user_guide"
ALLOWED = {"How-to", "Explanation", "Reference"}


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
    # The notebook cells end in .figure, which shows the plot, where the script saves it;
    # the prose must not call them "the same code".
    text = (GUIDE / "01-install.qmd").read_text()
    script = (GUIDE / "_quickstart.py").read_text()
    notebook = script.replace('.figure.savefig("first-plot.png")', ".figure")
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
