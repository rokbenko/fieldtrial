"""Print the CHANGELOG.md section of one version (used for GitHub release notes).

    python scripts/changelog_section.py 0.1.0rc1

Exits with status 1 when the version has no section, so a release cannot go out without
notes.
"""

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).parents[1] / "CHANGELOG.md"


def section(text: str, version: str) -> str | None:
    """The body of the ``## [version]`` section, without its heading."""
    pattern = re.compile(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", re.S | re.M)
    match = pattern.search(text)
    if match is None:
        return None
    body = re.sub(r"^\[[^\]]+\]: .*$", "", match.group(1), flags=re.M)  # link references
    return body.strip()


def main(argv: list[str]) -> int:
    """Print the section, or explain why there is none."""
    if len(argv) != 2:
        print("usage: changelog_section.py VERSION", file=sys.stderr)
        return 2
    version = argv[1].removeprefix("v")
    body = section(CHANGELOG.read_text(encoding="utf-8"), version)
    if not body:
        print(f"CHANGELOG.md has no '## [{version}]' section with content", file=sys.stderr)
        return 1
    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
