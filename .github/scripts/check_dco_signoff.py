import re
import sys
from pathlib import Path

SIGNED_OFF_RE = re.compile(r"^Signed-off-by:\s+.+\s+<[^<>]+>\s*$", re.MULTILINE)


def main() -> int:
    msg_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not msg_file or not msg_file.exists():
        print("dco-signoff: missing commit message file path", file=sys.stderr)
        return 2

    msg = msg_file.read_text(encoding="utf-8", errors="replace")
    if not SIGNED_OFF_RE.search(msg):
        print("ERROR: DCO sign-off required.", file=sys.stderr)
        print(
            "Add a line like: Signed-off-by: Your Name <you@example.com>",
            file=sys.stderr,
        )
        print('Tip: use `git commit -s -m "..."`', file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
