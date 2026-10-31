"""Structure checks for the docs sources (FR-12.3's four-part structure)."""

import re
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
