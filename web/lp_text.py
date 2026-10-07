"""Minimal human LP text → free MPS (demo modelling, not a full AMPL).

Supported (CPLEX-LP flavoured subset):

    Maximize
      3 x + 4 y
    Subject To
      c1: x + 2 y <= 10
      c2: 3 x + y >= 4
      c3: x + y = 5
    Bounds
      0 <= x <= 10
      y free
    Binary
      y
    General
      x
    End

Also accepts lowercase keywords and blank lines / # comments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class LpTextError(ValueError):
    pass


_TERM = re.compile(
    r"""
    (?P<sign>[+-])?\s*
    (?:(?P<coef>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\*?\s*)?
    (?P<var>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.X,
)

_SENSE = re.compile(
    r"(?P<body>.+?)\s*(?P<sense><=|>=|=|<|>)\s*(?P<rhs>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)

_BOUND = re.compile(
    r"""
    ^(?:
        (?P<lo>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|-\s*inf|inf)?\s*
        (?P<op1><=|>=|<|>)?\s*
        (?P<var>[A-Za-z_][A-Za-z0-9_]*)
        \s*(?P<op2><=|>=|<|>)?\s*
        (?P<hi>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\+?\s*inf|inf)?
        |
        (?P<var2>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<free>free)
    )\s*$
    """,
    re.X | re.I,
)


@dataclass
class _Model:
    maximize: bool = True
    obj: list[tuple[float, str]] = field(default_factory=list)
    rows: list[tuple[str, str, list[tuple[float, str]], float]] = field(default_factory=list)
    # sense in {L,G,E}
    col_lo: dict[str, float] = field(default_factory=dict)
    col_hi: dict[str, float] = field(default_factory=dict)
    integer: set[str] = field(default_factory=set)
    binary: set[str] = field(default_factory=set)
    vars: set[str] = field(default_factory=set)


def _parse_expr(text: str) -> list[tuple[float, str]]:
    s = text.strip()
    if not s:
        return []
    if s[0] not in "+-":
        s = "+ " + s
    terms: list[tuple[float, str]] = []
    pos = 0
    while pos < len(s):
        while pos < len(s) and s[pos].isspace():
            pos += 1
        if pos >= len(s):
            break
        m = _TERM.match(s, pos)
        if not m:
            raise LpTextError(f"Cannot parse expression near: {s[pos:pos+40]!r}")
        sign = -1.0 if m.group("sign") == "-" else 1.0
        coef = float(m.group("coef") or "1") * sign
        var = m.group("var")
        terms.append((coef, var))
        pos = m.end()
    return terms


def _strip_name(line: str) -> tuple[str | None, str]:
    if ":" in line:
        name, rest = line.split(":", 1)
        name = name.strip()
        if name and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            return name, rest.strip()
    return None, line.strip()


def parse_lp_text(text: str) -> _Model:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    if not lines:
        raise LpTextError("Empty model")

    m = _Model()
    section = None
    buf: list[str] = []
    row_i = 0

    def flush_obj() -> None:
        nonlocal buf
        if not buf:
            return
        expr = " ".join(buf)
        m.obj = _parse_expr(expr)
        for _, v in m.obj:
            m.vars.add(v)
        buf = []

    def flush_row() -> None:
        nonlocal buf, row_i
        if not buf:
            return
        line = " ".join(buf)
        name, rest = _strip_name(line)
        mm = _SENSE.match(rest)
        if not mm:
            raise LpTextError(f"Constraint needs <=, >=, or = : {line!r}")
        sense_tok = mm.group("sense")
        sense = {"<=": "L", "<": "L", ">=": "G", ">": "G", "=": "E"}[sense_tok]
        body = _parse_expr(mm.group("body"))
        rhs = float(mm.group("rhs"))
        for _, v in body:
            m.vars.add(v)
        row_i += 1
        m.rows.append((name or f"r{row_i}", sense, body, rhs))
        buf = []

    keywords = {
        "maximize": "obj",
        "max": "obj",
        "minimize": "obj",
        "min": "obj",
        "subject to": "st",
        "such that": "st",
        "s.t.": "st",
        "st": "st",
        "bounds": "bounds",
        "bound": "bounds",
        "binary": "binary",
        "binaries": "binary",
        "general": "general",
        "generals": "general",
        "integer": "general",
        "integers": "general",
        "end": "end",
    }

    i = 0
    while i < len(lines):
        low = lines[i].lower()
        # multi-word headers
        matched = None
        for k, sec in keywords.items():
            if low == k or low.startswith(k + " "):
                matched = (k, sec)
                break
        if matched:
            k, sec = matched
            if section == "obj":
                flush_obj()
            elif section == "st":
                flush_row()
            if sec == "obj":
                m.maximize = k.startswith("max")
                rest = lines[i][len(k) :].strip()
                section = "obj"
                buf = [rest] if rest else []
            elif sec == "end":
                section = None
            else:
                section = sec
                # remainder on same line after keyword
                rest = lines[i][len(k) :].strip()
                if rest and section == "st":
                    buf = [rest]
                    flush_row()
                elif rest and section == "bounds":
                    _apply_bound(m, rest)
                elif rest and section in {"binary", "general"}:
                    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rest):
                        m.vars.add(tok)
                        (m.binary if section == "binary" else m.integer).add(tok)
            i += 1
            continue

        if section == "obj":
            buf.append(lines[i])
        elif section == "st":
            # new constraint if previous finished sense OR starts with name:
            if buf and _SENSE.search(" ".join(buf)):
                flush_row()
            buf.append(lines[i])
            if _SENSE.search(" ".join(buf)):
                flush_row()
        elif section == "bounds":
            _apply_bound(m, lines[i])
        elif section == "binary":
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", lines[i]):
                m.vars.add(tok)
                m.binary.add(tok)
        elif section == "general":
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", lines[i]):
                m.vars.add(tok)
                m.integer.add(tok)
        else:
            raise LpTextError(f"Unexpected line (need Maximize/Minimize first): {lines[i]!r}")
        i += 1

    if section == "obj":
        flush_obj()
    elif section == "st":
        flush_row()

    if not m.obj and not m.rows:
        raise LpTextError("Model has no objective or constraints")
    if not m.vars:
        raise LpTextError("Model has no variables")

    for v in m.binary:
        m.col_lo.setdefault(v, 0.0)
        m.col_hi.setdefault(v, 1.0)
        m.integer.add(v)

    return m


def _apply_bound(m: _Model, line: str) -> None:
    s = line.strip().rstrip(";")
    mm = _BOUND.match(s)
    if not mm:
        # x >= 0  /  x <= 5
        m2 = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|>|<|=)\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$",
            s,
        )
        if not m2:
            raise LpTextError(f"Bad Bounds line: {line!r}")
        var, op, val = m2.group(1), m2.group(2), float(m2.group(3))
        m.vars.add(var)
        if op in (">=", ">"):
            m.col_lo[var] = val
        elif op in ("<=", "<"):
            m.col_hi[var] = val
        else:
            m.col_lo[var] = val
            m.col_hi[var] = val
        return

    if mm.group("free"):
        var = mm.group("var2")
        m.vars.add(var)
        m.col_lo[var] = float("-inf")
        m.col_hi[var] = float("inf")
        return

    var = mm.group("var")
    m.vars.add(var)
    lo, hi = mm.group("lo"), mm.group("hi")
    op1, op2 = mm.group("op1"), mm.group("op2")

    def num(x: str | None) -> float | None:
        if x is None:
            return None
        t = x.replace(" ", "").lower()
        if t in {"inf", "+inf"}:
            return float("inf")
        if t in {"-inf"}:
            return float("-inf")
        return float(t)

    # patterns: 0 <= x <= 10  |  x <= 10  |  0 <= x
    if lo is not None and op1 and op2 and hi is not None:
        m.col_lo[var] = num(lo)  # type: ignore
        m.col_hi[var] = num(hi)  # type: ignore
    elif lo is not None and op1 and not op2:
        # 0 <= x
        m.col_lo[var] = num(lo)  # type: ignore
    elif op2 and hi is not None:
        # x <= 10
        if op2 in ("<=", "<"):
            m.col_hi[var] = num(hi)  # type: ignore
        else:
            m.col_lo[var] = num(hi)  # type: ignore
    else:
        raise LpTextError(f"Bad Bounds line: {line!r}")


def to_mps(model: _Model, name: str = "TEXTLP") -> str:
    cols = sorted(model.vars)
    # default bounds: 0 <= x  (unless free/set)
    lines = [f"NAME          {name}"]
    if model.maximize:
        # sor_solve reads OBJSENSE natively, so objective, bounds and gaps all
        # come back in the user's own sense.
        lines += ["OBJSENSE", "    MAX"]
    lines += ["ROWS", " N  COST"]
    for rname, sense, _, _ in model.rows:
        lines.append(f" {sense}  {rname}")

    lines.append("COLUMNS")
    int_marker = False
    # Emit integer block if any integers
    ints = sorted(model.integer)
    conts = [c for c in cols if c not in model.integer]

    def emit_col(v: str) -> None:
        # objective
        obj_coef = sum(c for c, name_ in model.obj if name_ == v)
        entries: list[tuple[str, float]] = []
        if abs(obj_coef) > 0:
            entries.append(("COST", obj_coef))
        for rname, _, body, _ in model.rows:
            coef = sum(c for c, name_ in body if name_ == v)
            if abs(coef) > 0:
                entries.append((rname, coef))
        if not entries:
            entries.append(("COST", 0.0))
        for i in range(0, len(entries), 2):
            chunk = entries[i : i + 2]
            if len(chunk) == 1:
                r, a = chunk[0]
                lines.append(f"    {v:<8}  {r:<8}  {a!r}")
            else:
                (r1, a1), (r2, a2) = chunk
                lines.append(f"    {v:<8}  {r1:<8}  {a1!r}   {r2:<8}  {a2!r}")

    if ints:
        lines.append("    MARK0000  'MARKER'                 'INTORG'")
        for v in ints:
            emit_col(v)
        lines.append("    MARK0001  'MARKER'                 'INTEND'")
    for v in conts:
        emit_col(v)

    lines.append("RHS")
    for rname, _, _, rhs in model.rows:
        lines.append(f"    RHS1      {rname:<8}  {rhs!r}")

    lines.append("BOUNDS")
    for v in cols:
        lo = model.col_lo.get(v, 0.0)
        hi = model.col_hi.get(v, float("inf"))
        if lo == float("-inf") and hi == float("inf"):
            lines.append(f" FR BND1      {v}")
        else:
            if lo != 0.0 and lo != float("-inf"):
                lines.append(f" LO BND1      {v:<8}  {lo!r}")
            if lo == float("-inf"):
                lines.append(f" MI BND1      {v}")
            if hi != float("inf"):
                lines.append(f" UP BND1      {v:<8}  {hi!r}")
            elif v in model.binary:
                lines.append(f" UP BND1      {v:<8}  1")
    lines.append("ENDATA")
    return "\n".join(lines) + "\n"


def lp_text_to_mps(text: str, name: str = "TEXTLP") -> tuple[str, dict]:
    model = parse_lp_text(text)
    meta = {
        "maximize": model.maximize,
        "n_vars": len(model.vars),
        "n_constraints": len(model.rows),
        "n_integer": len(model.integer),
        "suggested_engine": "milp" if model.integer else "simplex",
    }
    return to_mps(model, name=name), meta


SAMPLE = """Maximize
  3 x + 4 y
Subject To
  c1: x + 2 y <= 14
  c2: 3 x + y <= 18
Bounds
  x >= 0
  y >= 0
General
  x
Binary
  y
End
"""
