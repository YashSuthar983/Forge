#!/usr/bin/env python3
"""SOR Python REPL — project entry point.

  cd /home/yash/Desktop/Sih/sor
  python3 scripts/sor_repl.py
  python3 scripts/sor_repl.py --one-shot
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT / "bindings"
if str(BINDINGS) not in sys.path:
    sys.path.insert(0, str(BINDINGS))

from python.sor_api import run_one_shot, run_repl  # noqa: E402


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--one-shot":
        run_one_shot()
    else:
        run_repl()


if __name__ == "__main__":
    main()
