#!/usr/bin/env python3
"""
Automated Web Frontend Sanitizer:
1. Extracts inline JavaScript from HTML templates into standalone static .js files.
2. Automatically removes all emoji characters from web templates and static assets.
3. Verifies zero inline scripts and zero emojis across web/templates and web/static.
"""

import os
import re
import sys
from pathlib import Path

# Target directories
REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "web" / "templates"
STATIC_DIR = REPO_ROOT / "web" / "static"

# Comprehensive regex matching emoji symbols and optional variation selectors
EMOJI_PATTERN = re.compile(
    r'[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf][\ufe00-\ufe0f]?\s*'
)


def extract_inline_scripts(templates_dir: Path, static_dir: Path):
    """Finds inline script blocks in HTML templates and extracts them to static JS files."""
    login_html_path = templates_dir / "login.html"
    login_js_path = static_dir / "login.js"

    content = login_html_path.read_text(encoding="utf-8")

    # Match <script> block inside login.html
    script_match = re.search(r"<script>(.*?)</script>", content, re.DOTALL)
    if script_match:
        raw_js = script_match.group(1).strip()
        # Clean any emoji markers from feedback calls
        cleaned_js = EMOJI_PATTERN.sub("", raw_js)
        # Clean up empty string concatenation like `'' + `
        cleaned_js = re.sub(r"setFeedback\(''\s*\+\s*", "setFeedback(", cleaned_js)
        cleaned_js = re.sub(r"setFeedback\(\s*'\s*'\s*\+\s*", "setFeedback(", cleaned_js)

        # Write out to login.js
        login_js_path.write_text(cleaned_js + "\n", encoding="utf-8")
        print(f"[EXTRACT] Extracted {len(cleaned_js)} chars of JS into {login_js_path}")

        # Replace inline script with script src tag
        new_block = (
            '{% block scripts %}\n'
            '<script src="{{ url_for(\'static\', path=\'login.js\').path }}" defer></script>\n'
            '{% endblock %}'
        )
        content = re.sub(
            r"\{%\s*block\s+scripts\s*%\}\s*<script>.*?</script>\s*\{%\s*endblock\s*%\}",
            new_block,
            content,
            flags=re.DOTALL,
        )
        login_html_path.write_text(content, encoding="utf-8")
        print(f"[UPDATE] Updated {login_html_path} to use static/login.js")


def strip_emojis_from_file(file_path: Path):
    """Strips all emoji characters from a single file and cleans formatting."""
    content = file_path.read_text(encoding="utf-8")
    matches = EMOJI_PATTERN.findall(content)
    if not matches:
        return 0

    # Clean empty span if present: e.g. <span class="fs-4">⚠️</span>
    cleaned = re.sub(
        r'<span\s+class="fs-4">[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf][\ufe00-\ufe0f]?</span>\s*',
        '<span class="badge bg-warning text-dark me-2">提示</span>\n    ',
        content,
    )

    # Clean all other emojis
    cleaned = EMOJI_PATTERN.sub("", cleaned)

    # Specific cleanups for clean typography
    cleaned = re.sub(r"setFeedback\(''\s*\+\s*", "setFeedback(", cleaned)

    file_path.write_text(cleaned, encoding="utf-8")
    print(f"[CLEAN] Removed {len(matches)} emoji(s) from {file_path.name}: {matches}")
    return len(matches)


def audit_cleanliness(templates_dir: Path, static_dir: Path):
    """Verifies that no inline scripts remain in templates and no emojis remain in web assets."""
    print("\n--- Auditing Web Assets Cleanliness ---")
    errors = 0

    # 1. Audit templates for inline scripts
    for html_file in sorted(templates_dir.glob("*.html")):
        text = html_file.read_text(encoding="utf-8")
        scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", text, re.DOTALL)
        for attrs, body in scripts:
            if "src=" not in attrs and body.strip():
                print(f"[ERROR] Inline script found in {html_file.name} ({len(body)} chars)")
                errors += 1

    # 2. Audit all templates and static JS/CSS for emojis
    files_to_check = list(templates_dir.glob("*.html")) + [
        f for f in static_dir.glob("*") if f.suffix in (".js", ".css") and "vendor" not in f.name
    ]

    for f in sorted(files_to_check):
        text = f.read_text(encoding="utf-8")
        emojis_found = EMOJI_PATTERN.findall(text)
        if emojis_found:
            print(f"[ERROR] Emojis still found in {f.name}: {emojis_found}")
            errors += 1

    if errors == 0:
        print("[SUCCESS] All web templates and JS assets are 100% free of emojis and inline scripts!")
    else:
        print(f"[FAILURE] Found {errors} violation(s).")
    return errors == 0


def main():
    if not TEMPLATES_DIR.exists():
        print(f"Error: {TEMPLATES_DIR} does not exist.")
        sys.exit(1)

    extract_inline_scripts(TEMPLATES_DIR, STATIC_DIR)

    total_stripped = 0
    for html_file in sorted(TEMPLATES_DIR.glob("*.html")):
        total_stripped += strip_emojis_from_file(html_file)

    for js_file in sorted(STATIC_DIR.glob("*.js")):
        total_stripped += strip_emojis_from_file(js_file)

    print(f"\nTotal emoji instances stripped: {total_stripped}")

    success = audit_cleanliness(TEMPLATES_DIR, STATIC_DIR)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
