"""
Test Web Frontend Cleanliness:
1. Asserts no emoji characters exist in web templates or frontend scripts.
2. Asserts all JavaScript in HTML templates is separated into external .js files.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "web" / "templates"
STATIC_DIR = REPO_ROOT / "web" / "static"

EMOJI_PATTERN = re.compile(
    r'[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf][\ufe00-\ufe0f]?'
)


def test_no_inline_scripts_in_templates():
    """Verify that no HTML templates contain inline JavaScript logic."""
    for html_file in sorted(TEMPLATES_DIR.glob("*.html")):
        text = html_file.read_text(encoding="utf-8")
        scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", text, re.DOTALL)
        for attrs, body in scripts:
            assert "src=" in attrs, (
                f"Template {html_file.name} contains inline script ({len(body.strip())} chars). "
                f"JavaScript must be separated into a standalone file under web/static/."
            )
            assert not body.strip(), (
                f"Template {html_file.name} has non-empty script body despite src attribute."
            )


def test_no_emojis_in_templates():
    """Verify that no HTML templates contain emoji characters."""
    for html_file in sorted(TEMPLATES_DIR.glob("*.html")):
        text = html_file.read_text(encoding="utf-8")
        emojis = EMOJI_PATTERN.findall(text)
        assert not emojis, (
            f"Template {html_file.name} contains emoji characters: {emojis}. "
            f"Web UI must not contain emojis."
        )


def test_no_emojis_in_static_assets():
    """Verify that no frontend JS or CSS files contain emoji characters."""
    for asset_file in sorted(STATIC_DIR.glob("*")):
        if asset_file.suffix in (".js", ".css") and "vendor" not in asset_file.name:
            text = asset_file.read_text(encoding="utf-8")
            emojis = EMOJI_PATTERN.findall(text)
            assert not emojis, (
                f"Asset {asset_file.name} contains emoji characters: {emojis}."
            )


def test_login_js_exists_and_referenced():
    """Verify that login.js exists and is referenced in login.html."""
    login_js = STATIC_DIR / "login.js"
    assert login_js.is_file(), "web/static/login.js does not exist"

    login_html = TEMPLATES_DIR / "login.html"
    content = login_html.read_text(encoding="utf-8")
    assert "login.js" in content, "web/templates/login.html does not reference login.js"
