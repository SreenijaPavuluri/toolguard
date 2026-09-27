"""`toolguard scan <file>` command."""
import argparse
import json
import sys

from . import Guard


def main(argv=None):
    ap = argparse.ArgumentParser(prog="toolguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan a file (or - for stdin) for prompt injection")
    s.add_argument("file")
    s.add_argument("--model", default=None)
    a = ap.parse_args(argv)
    text = sys.stdin.read() if a.file == "-" else open(a.file, encoding="utf-8", errors="replace").read()
    r = Guard.load(a.model).scan(text)
    excerpt = text[r.span[0]:r.span[1]][:300] if r.span else ""
    print(json.dumps({"file": a.file, "is_injection": r.is_injection, "score": round(r.score, 4), "threshold": round(r.threshold, 4),
                      "span": r.span, "top_window_excerpt": excerpt}, indent=2))
    return 1 if r.is_injection else 0


if __name__ == "__main__":
    sys.exit(main())
