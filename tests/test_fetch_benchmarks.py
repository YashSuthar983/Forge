#!/usr/bin/env python3
"""Offline tests for scripts/fetch_benchmarks.py."""
from __future__ import annotations

import importlib.util
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "fetch_benchmarks", ROOT / "scripts" / "fetch_benchmarks.py")
assert SPEC and SPEC.loader
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata = root / "miplib2017" / "benchmark-v2.test"
        metadata.parent.mkdir(parents=True)
        metadata.write_text("# comment\na.mps.gz\nb.mps.gz\n", encoding="utf-8")

        assert fetch.suite_names("miplib2017", root) == ["a", "b"]
        assert fetch.select_instances("netlib", ["afiro.mps", "afiro"], root) == ["afiro"]
        assert fetch.select_instances("miplib-easy", ["p0033"], root) == ["p0033"]
        assert fetch.looks_like_mps(b"* comment\nNAME X\nROWS\n")
        assert not fetch.looks_like_mps(b"<html>not a model</html>")
        try:
            fetch.select_instances("netlib", ["not-a-model"], root)
        except ValueError as error:
            assert "use --list" in str(error)
        else:
            raise AssertionError("unknown model was accepted")

        stream = io.StringIO()
        with redirect_stdout(stream):
            assert fetch.main(["--root", str(root), "--suite", "netlib", "--list"]) == 0
        assert "netlib (93)" in stream.getvalue()
        assert "afiro" in stream.getvalue()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
