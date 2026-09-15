#!/usr/bin/env python3
"""Count the presolve reductions an MPS model would offer, without solving it.

This is a measurement tool, not a presolver: it reads models and reports how
many of each classical Andersen & Andersen (1995) reduction is *available* in
the original problem, so a presolve implementation can be aimed at the
reductions that actually occur in the benchmark sets.

    # one CSV row per Netlib instance, to stdout
    scripts/analyze_presolve_opportunities.py benchmarks/netlib/mps

    # both formats, to files
    scripts/analyze_presolve_opportunities.py benchmarks/netlib/mps \\
        --csv presolve_ops.csv --json presolve_ops.json

Counted per instance:

    singleton_columns    columns with exactly one structural entry in A
    singleton_rows       rows with exactly one structural entry
    doubleton_equations  equality rows with exactly two structural entries
    duplicate_rows       rows parallel to an earlier row (excess over one
                         representative per parallel group)
    forcing_rows         rows whose implied activity range touches a finite
                         side exactly, so every variable in them is forced
    empty_columns        columns with no structural entry in A

The counts are INDEPENDENT single-pass counts on the ORIGINAL model. They are
not what a presolve loop would remove, because reductions cascade: eliminating
a singleton column can create a new singleton row, and a doubleton equation is
also counted as such even if a duplicate-row rule would have removed it first.
Read them as an upper bound on first-round opportunity, per rule.

MODEL SEMANTICS match src/io/src/mps.cpp exactly, so the counts describe the
problem SOR actually builds: free-format tokenization with a fixed-format
retry (Netlib's `forplan` has row names containing spaces), only the first N
row is the objective (later N rows are discarded), RANGES per the MPS
specification, the negative-UP-bound quirk, duplicate COLUMNS entries summed,
and explicit zeros retained as structural entries.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

INF = float("inf")

# Depth-independent: walk up to whichever directory holds CMakeLists.txt.
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "CMakeLists.txt").exists())

_SECTIONS = {"NAME", "OBJSENSE", "OBJSENS", "ROWS", "COLUMNS", "RHS", "RANGES",
             "BOUNDS", "QUADOBJ", "QMATRIX", "ENDATA"}


class MpsError(ValueError):
    """A model that this reader cannot interpret."""


@dataclass
class Lp:
    """A model in the same shape src/io/src/mps.cpp assembles."""
    name: str = ""
    maximize: bool = False
    row_names: list[str] = field(default_factory=list)
    col_names: list[str] = field(default_factory=list)
    # rows[i] is {column index: coefficient}; duplicates already summed.
    rows: list[dict[int, float]] = field(default_factory=list)
    row_lo: list[float] = field(default_factory=list)
    row_hi: list[float] = field(default_factory=list)
    col_lo: list[float] = field(default_factory=list)
    col_hi: list[float] = field(default_factory=list)
    is_integer: list[bool] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.col_names)

    @property
    def nnz(self) -> int:
        return sum(len(r) for r in self.rows)

    def column_lengths(self) -> list[int]:
        counts = [0] * self.n_cols
        for r in self.rows:
            for j in r:
                counts[j] += 1
        return counts


# --------------------------------------------------------------------------
# MPS reading
# --------------------------------------------------------------------------
def _num(token: str, line_no: int) -> float:
    try:
        return float(token)
    except ValueError as e:
        raise MpsError(f"line {line_no}: not a number: {token!r}") from e


# Fixed-format MPS field columns, 0-based starts and lengths -- the same table
# split_fixed() uses in src/io/src/mps.cpp. Empty fields are dropped so the
# token list has free-format semantics while names containing SPACES survive.
_FIXED_BEG = (1, 4, 14, 24, 39, 49)
_FIXED_LEN = (2, 8, 8, 12, 8, 12)


def split_fixed(line: str) -> list[str]:
    out = []
    for beg, ln in zip(_FIXED_BEG, _FIXED_LEN):
        if len(line) <= beg:
            break
        f = line[beg:beg + ln].strip(" \t")
        if f:
            out.append(f)
    return out


def read_mps(path: Path, fixed_format: bool = False) -> Lp:
    """Read an MPS file into an Lp. Raises MpsError on anything the C++ reader
    would also reject. Use read_mps_auto() for the free-then-fixed retry."""
    split = split_fixed if fixed_format else str.split
    lp = Lp()
    row_kind: dict[str, str] = {}      # row name -> N/L/G/E
    row_index: dict[str, int] = {}     # constraint name -> index into lp.rows
    col_index: dict[str, int] = {}
    rhs: dict[int, float] = {}
    ranges: dict[int, float] = {}
    lo_set: list[bool] = []
    hi_set: list[bool] = []
    obj_name: str | None = None
    section = ""
    in_integer_marker = False
    saw_endata = False

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, 1):
            if not raw.strip() or raw.lstrip().startswith("*"):
                continue
            if raw[0] not in " \t":
                # Section headers are whitespace-split in BOTH formats.
                f = raw.split()
                key = f[0].upper()
                if key not in _SECTIONS:
                    # Non-strict, like the C++ reader: ignore what we don't know.
                    section = ""
                    continue
                section = key
                if key == "NAME":
                    lp.name = f[1] if len(f) > 1 else path.stem
                elif key in ("OBJSENSE", "OBJSENS"):
                    if len(f) > 1 and f[1].upper().startswith("MAX"):
                        lp.maximize = True
                elif key == "ENDATA":
                    saw_endata = True
                    break
                continue

            f = split(raw.rstrip("\n").rstrip("\r"))
            if not f:
                continue

            if section == "ROWS":
                if len(f) < 2:
                    raise MpsError(f"line {line_no}: ROWS entry needs kind and name")
                kind, name = f[0].upper(), f[1]
                if kind not in ("N", "L", "G", "E"):
                    raise MpsError(f"line {line_no}: unknown row kind {f[0]!r}")
                if name in row_kind:
                    raise MpsError(f"line {line_no}: duplicate row name {name!r}")
                row_kind[name] = kind
                if kind == "N":
                    # Only the FIRST N row is the objective; later ones are free
                    # rows whose coefficients the reader discards.
                    if obj_name is None:
                        obj_name = name
                else:
                    row_index[name] = len(lp.rows)
                    lp.rows.append({})
                    lp.row_names.append(name)

            elif section == "COLUMNS":
                if any("MARKER" in t.upper() for t in f):
                    joined = " ".join(f).upper()
                    if "INTORG" in joined:
                        in_integer_marker = True
                    if "INTEND" in joined:
                        in_integer_marker = False
                    continue
                if len(f) < 3:
                    raise MpsError(f"line {line_no}: COLUMNS entry needs col, row, value")
                name = f[0]
                j = col_index.get(name)
                if j is None:
                    j = col_index[name] = len(lp.col_names)
                    lp.col_names.append(name)
                    lp.col_lo.append(0.0)
                    lp.col_hi.append(INF)
                    lp.is_integer.append(False)
                    lo_set.append(False)
                    hi_set.append(False)
                if in_integer_marker:
                    lp.is_integer[j] = True
                for k in range(1, len(f) - 1, 2):
                    rname = f[k]
                    if rname not in row_kind:
                        raise MpsError(f"line {line_no}: unknown row {rname!r}")
                    v = _num(f[k + 1], line_no)
                    i = row_index.get(rname)
                    if i is None:
                        continue  # objective or a discarded free row
                    # Duplicate entries sum, matching sparse::from_triplets.
                    lp.rows[i][j] = lp.rows[i].get(j, 0.0) + v

            elif section in ("RHS", "RANGES"):
                # The leading token may be a set name, or may be omitted.
                start = 1 if len(f) % 2 == 1 else 0
                target = rhs if section == "RHS" else ranges
                for k in range(start, len(f) - 1, 2):
                    rname = f[k]
                    if rname not in row_kind:
                        raise MpsError(
                            f"line {line_no}: unknown row {rname!r} in {section}")
                    v = _num(f[k + 1], line_no)
                    i = row_index.get(rname)
                    if i is not None:
                        target[i] = v

            elif section == "BOUNDS":
                if len(f) < 2:
                    raise MpsError(f"line {line_no}: BOUNDS entry needs a type and column")
                btype = f[0].upper()
                # Field 2 is a bound-set name when field 3 names the column.
                ci, vi = 1, 2
                if len(f) > 2 and f[2] in col_index and f[1] not in col_index:
                    ci, vi = 2, 3
                if f[ci] not in col_index:
                    raise MpsError(
                        f"line {line_no}: unknown column {f[ci]!r} in BOUNDS")
                j = col_index[f[ci]]
                needs_value = btype in ("UP", "LO", "FX", "UI", "LI")
                if needs_value and vi >= len(f):
                    raise MpsError(
                        f"line {line_no}: bound type {btype} needs a value")
                v = _num(f[vi], line_no) if needs_value else 0.0
                if btype in ("UP", "UI"):
                    lp.col_hi[j], hi_set[j] = v, True
                    # Documented MPS quirk: a negative UP on a column still at
                    # the default lower bound implies lower = -inf.
                    if v < 0.0 and not lo_set[j]:
                        lp.col_lo[j] = -INF
                    if btype == "UI":
                        lp.is_integer[j] = True
                elif btype in ("LO", "LI"):
                    lp.col_lo[j], lo_set[j] = v, True
                    if btype == "LI":
                        lp.is_integer[j] = True
                elif btype == "FX":
                    lp.col_lo[j] = lp.col_hi[j] = v
                    lo_set[j] = hi_set[j] = True
                elif btype == "FR":
                    lp.col_lo[j], lp.col_hi[j] = -INF, INF
                    lo_set[j] = hi_set[j] = True
                elif btype == "MI":
                    lp.col_lo[j], lo_set[j] = -INF, True
                elif btype == "PL":
                    lp.col_hi[j], hi_set[j] = INF, True
                elif btype == "BV":
                    lp.col_lo[j], lp.col_hi[j] = 0.0, 1.0
                    lo_set[j] = hi_set[j] = True
                    lp.is_integer[j] = True
                else:
                    raise MpsError(f"line {line_no}: unknown bound type {f[0]!r}")

            elif section in ("OBJSENSE", "OBJSENS"):
                if f[0].upper().startswith("MAX"):
                    lp.maximize = True

            # NAME/QUADOBJ/QMATRIX payload lines carry nothing this tool needs.

    # A missing ENDATA is only a warning in the C++ reader, so it is not fatal
    # here either. A model with neither rows nor columns IS fatal: that is what
    # a file whose section headers were never recognised looks like, and
    # reporting it as a genuine 0x0 model would silently understate a sweep.
    if not lp.rows and not lp.col_names:
        raise MpsError("no ROWS/COLUMNS data found"
                       + ("" if saw_endata else " (and no ENDATA record)"))

    for i in range(lp.n_rows):
        r, kind = rhs.get(i, 0.0), row_kind[lp.row_names[i]]
        if i not in ranges:
            lo, hi = {"L": (-INF, r), "G": (r, INF), "E": (r, r)}[kind]
        else:
            R = ranges[i]
            a = abs(R)
            if kind == "L":
                lo, hi = r - a, r
            elif kind == "G":
                lo, hi = r, r + a
            else:  # E
                lo, hi = (r, r + R) if R >= 0.0 else (r + R, r)
        lp.row_lo.append(lo)
        lp.row_hi.append(hi)

    return lp


def read_mps_auto(path: Path) -> tuple[Lp, bool]:
    """Free format, retrying in fixed format. Returns (lp, used_fixed_format).

    Same fallback as io::read_mps_file_auto: fixed format only ever helps with
    names containing spaces, so the free-format error is the informative one
    when both fail.
    """
    try:
        return read_mps(path, fixed_format=False), False
    except MpsError as free_err:
        try:
            return read_mps(path, fixed_format=True), True
        except MpsError as fixed_err:
            raise MpsError(f"parse failed in both formats. free: {free_err} | "
                           f"fixed: {fixed_err}") from free_err


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------
def _close(a: float, b: float, tol: float) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= tol * (1.0 + max(abs(a), abs(b)))


def parallel_key(entries: dict[int, float]) -> tuple:
    """A hashable signature that is equal for parallel rows.

    Rows are normalized by their first nonzero coefficient in column order, so
    a row and any nonzero scalar multiple of it produce the same key. The
    pattern is part of the key exactly (no rounding); only the ratios are
    rounded, to 12 significant digits, which is well inside the precision an
    MPS file carries.
    """
    cols = sorted(entries)
    pivot = 0.0
    for j in cols:
        if entries[j] != 0.0:
            pivot = entries[j]
            break
    if pivot == 0.0:
        # All-zero row (explicit zeros only): parallel to every other such row
        # with the same pattern.
        return (tuple(cols), None)
    return (tuple(cols), tuple(float(f"{entries[j] / pivot:.12g}") for j in cols))


def activity_bounds(entries: dict[int, float], col_lo: list[float],
                    col_hi: list[float]) -> tuple[float, float]:
    """(min, max) of sum a_j x_j over the column bounds. +/-inf when unbounded."""
    lo = hi = 0.0
    for j, a in entries.items():
        if a == 0.0:
            continue  # an explicit zero contributes nothing, even against inf
        if a > 0.0:
            lo += a * col_lo[j]
            hi += a * col_hi[j]
        else:
            lo += a * col_hi[j]
            hi += a * col_lo[j]
        if lo == -INF and hi == INF:
            break
    return lo, hi


def analyze(lp: Lp, tol: float = 1e-9) -> dict[str, int]:
    """Count each reduction opportunity in `lp`."""
    col_len = lp.column_lengths()

    singleton_columns = sum(1 for c in col_len if c == 1)
    empty_columns = sum(1 for c in col_len if c == 0)

    singleton_rows = 0
    doubleton_equations = 0
    forcing_rows = 0
    seen: dict[tuple, int] = {}
    duplicate_rows = 0

    for i, entries in enumerate(lp.rows):
        n = len(entries)
        if n == 1:
            singleton_rows += 1
        lo, hi = lp.row_lo[i], lp.row_hi[i]
        if n == 2 and math.isfinite(lo) and lo == hi:
            doubleton_equations += 1

        if n:
            key = parallel_key(entries)
            if key in seen:
                duplicate_rows += 1
            else:
                seen[key] = i

            act_lo, act_hi = activity_bounds(entries, lp.col_lo, lp.col_hi)
            # Forcing: the reachable activity range touches a finite side of the
            # row exactly, so every variable in the row is pinned to one bound.
            # A row whose activity cannot reach its range at all is infeasible
            # rather than forcing, and is deliberately not counted here.
            if _close(act_lo, hi, tol) or _close(act_hi, lo, tol):
                forcing_rows += 1

    return {
        "singleton_columns": singleton_columns,
        "singleton_rows": singleton_rows,
        "doubleton_equations": doubleton_equations,
        "duplicate_rows": duplicate_rows,
        "forcing_rows": forcing_rows,
        "empty_columns": empty_columns,
    }


FIELDS = ["instance", "path", "rows", "cols", "nnz", "integer_cols",
          "singleton_columns", "singleton_rows", "doubleton_equations",
          "duplicate_rows", "forcing_rows", "empty_columns", "fixed_format",
          "error"]


def scan(path: Path, tol: float) -> dict[str, object]:
    """One output record for one model file. Never raises."""
    rec: dict[str, object] = {"instance": path.stem, "path": str(path),
                              "error": None}
    try:
        lp, rec["fixed_format"] = read_mps_auto(path)
    except (MpsError, OSError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        for k in FIELDS:
            rec.setdefault(k, None)
        return rec
    rec["rows"] = lp.n_rows
    rec["cols"] = lp.n_cols
    rec["nnz"] = lp.nnz
    rec["integer_cols"] = sum(lp.is_integer)
    rec.update(analyze(lp, tol))
    return rec


def collect_models(paths: list[str]) -> list[Path]:
    models: list[Path] = []
    for p in paths:
        q = Path(p)
        if not q.is_absolute() and not q.exists() and (ROOT / q).exists():
            q = ROOT / q
        if q.is_dir():
            models.extend(sorted(q.rglob("*.mps")))
        elif q.exists():
            models.append(q)
        else:
            print(f"warning: no such model or directory: {p}", file=sys.stderr)
    seen, out = set(), []
    for m in models:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def write_csv(records: list[dict[str, object]], fh) -> None:
    w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in records:
        w.writerow({k: r.get(k) for k in FIELDS})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Count presolve reduction opportunities in MPS models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("models", nargs="+",
                    help="model files, or directories to scan for .mps")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write CSV here (default: CSV to stdout)")
    ap.add_argument("--json", type=Path, default=None, help="write JSON here")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="relative tolerance for the forcing-row equality test")
    args = ap.parse_args(argv)

    if not math.isfinite(args.tol) or args.tol < 0.0:
        ap.error("--tol must be finite and nonnegative")

    models = collect_models(args.models)
    if not models:
        print("error: no models found", file=sys.stderr)
        return 2

    records = [scan(m, args.tol) for m in models]

    if args.csv is not None:
        with args.csv.open("w", newline="") as fh:
            write_csv(records, fh)
    if args.json is not None:
        with args.json.open("w") as fh:
            json.dump({"tool": "analyze_presolve_opportunities",
                       "tolerance": args.tol,
                       "instances": records}, fh, indent=2)
            fh.write("\n")
    if args.csv is None and args.json is None:
        write_csv(records, sys.stdout)

    failed = [r for r in records if r["error"]]
    for r in failed:
        print(f"error: {r['path']}: {r['error']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
