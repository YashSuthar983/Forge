#!/usr/bin/env python3
"""Freeze a benchmark suite into a manifest.

    scripts/freeze_manifest.py netlib benchmarks/netlib/mps \
        --reference-run benchmarks/results/reference-highs-netlib.jsonl

Section 3.1 of LP_HIGHS_BEAT_EXECUTION_PLAN_20260910.md: a frozen manifest
records, per model, the source, a SHA-256, dimensions, nonzeros, input
warnings, the relaxation policy, reference status and objective, and whether
the model is eligible for scoring.

The point is not bookkeeping. A performance claim is meaningless unless the
corpus it was measured on is pinned: "Netlib-93" is not a specification, a
manifest with 93 checksums is. Any later change to the corpus must create a NEW
manifest version, because aggregates computed against two different corpora are
not comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "CMakeLists.txt").exists())
sys.path.insert(0, str(ROOT / "scripts"))

MANIFEST_VERSION = 1


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_shape(path: Path) -> dict:
    """Rows, columns, nonzeros and any reader warning, via SOR's own reader.

    Deliberately SOR's reader and not a Python re-implementation: the manifest
    must describe the model the solver will actually see, including the
    fixed-format fallback and any coefficient the small-value threshold drops.
    """
    import subprocess
    exe = ROOT / "build" / "sor_solve"
    for cand in (ROOT / "build-det" / "sor_solve", exe):
        if cand.exists():
            exe = cand
            break
    if not exe.exists():
        return {"rows": None, "cols": None, "nnz": None,
                "warnings": ["sor_solve not built; shape not recorded"]}
    try:
        out = subprocess.run(
            [str(exe), str(path), "--max-iter", "1", "--time-limit", "1"],
            capture_output=True, text=True, timeout=600).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return {"rows": None, "cols": None, "nnz": None,
                "warnings": [f"{type(e).__name__}: {e}"]}
    rows = cols = nnz = None
    warnings: list[str] = []
    for line in out.splitlines():
        if line.startswith("rows x cols:"):
            body = line.split(":", 1)[1].split()
            try:
                rows, cols, nnz = int(body[0]), int(body[2]), int(body[4])
            except (IndexError, ValueError):
                pass
        low = line.lower()
        if "warning" in low or "dropped" in low:
            warnings.append(line.strip())
    return {"rows": rows, "cols": cols, "nnz": nnz, "warnings": warnings}


def load_reference(path: Path | None) -> dict:
    """instance -> (status, objective) from a committed solver sweep."""
    if path is None or not path.exists():
        return {}
    ref: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("record") not in ("run", "aggregate"):
            continue
        if not str(rec.get("solver", "")).lower().startswith("highs"):
            continue
        ref[rec["instance"]] = {"status": rec.get("status"),
                                "objective": rec.get("objective")}
    return ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", help="suite name, e.g. netlib / ladder-lp / miplib2017-lp")
    ap.add_argument("directory", help="directory holding the models")
    ap.add_argument("--tier", default="unspecified")
    ap.add_argument("--source-url", default="")
    ap.add_argument("--source-version", default="")
    ap.add_argument("--relaxation", choices=("none", "relax_integrality"),
                    default="none",
                    help="how an integer model is turned into the LP that is scored")
    ap.add_argument("--reference-run", type=Path, default=None,
                    help="a committed JSONL sweep supplying reference objectives")
    ap.add_argument("--select", type=Path, default=None,
                    help="file listing the model basenames to include, one per line")
    ap.add_argument("--patterns", default="*.mps,*.mps.gz,*.qps,*.lp")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    directory = Path(args.directory)
    if not directory.is_absolute():
        directory = ROOT / directory
    if not directory.is_dir():
        print(f"error: {directory} is not a directory", file=sys.stderr)
        return 2

    wanted = None
    if args.select is not None:
        wanted = {ln.strip() for ln in args.select.read_text().splitlines() if ln.strip()}

    files: list[Path] = []
    for pat in args.patterns.split(","):
        files.extend(sorted(directory.rglob(pat.strip())))
    seen, models = set(), []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        if wanted is not None and f.name not in wanted:
            continue
        models.append(f)
    if wanted is not None:
        missing = sorted(wanted - {m.name for m in models})
        if missing:
            print(f"error: {len(missing)} model(s) in --select not found, "
                  f"first: {missing[:3]}", file=sys.stderr)
            return 2

    reference = load_reference(args.reference_run)
    entries = []
    for path in models:
        shape = model_shape(path)
        ref = reference.get(path.name, {})
        obj = ref.get("objective")
        status = ref.get("status")
        # A model is scoring-eligible only when the manifest can state what the
        # right answer is. Without a reference objective a timing is unchecked,
        # and an unchecked timing cannot support a claim.
        eligible = bool(status) and obj is not None
        entries.append({
            "name": path.name,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_of(path),
            "bytes": path.stat().st_size,
            "gzipped": path.name.endswith(".gz"),
            "rows": shape["rows"],
            "cols": shape["cols"],
            "nnz": shape["nnz"],
            "input_warnings": shape["warnings"],
            "relaxation": args.relaxation,
            "reference_status": status,
            "reference_objective": obj,
            "scoring_eligible": eligible,
        })

    doc = {
        "manifest_version": MANIFEST_VERSION,
        "suite": args.suite,
        "tier": args.tier,
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": args.source_url,
        "source_version": args.source_version,
        "relaxation_policy": args.relaxation,
        "reference_run": (str(args.reference_run.relative_to(ROOT))
                          if args.reference_run and args.reference_run.is_absolute()
                          and str(args.reference_run).startswith(str(ROOT))
                          else (str(args.reference_run) if args.reference_run else None)),
        "model_count": len(entries),
        "scoring_eligible_count": sum(1 for e in entries if e["scoring_eligible"]),
        "models": entries,
    }
    out = args.out or (ROOT / "benchmarks" / "manifests" / f"{args.suite}-v{MANIFEST_VERSION}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    print(f"{out}: {len(entries)} model(s), "
          f"{doc['scoring_eligible_count']} scoring-eligible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
