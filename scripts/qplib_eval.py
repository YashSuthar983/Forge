#!/usr/bin/env python3
"""Independently evaluate a point against ANY QPLIB instance: objective, and
the worst violation of constraint rows (linear and quadratic), variable
bounds and integrality -- the three terms of QPLIB's SOLINFEASIBILITY.

A second reader, written from the QPLIB format description (Furini et al.,
"QPLIB: a library of quadratic programming instances", MPC 2019, appendix,
and qplib.zib.de/doc.html) and sharing no code with the C++ parser, so a
mis-read model cannot pass both.  Used to audit a claim that disagrees with
QPLIB's published point, and by scripts/qplib_parse_all.py.

    python3 scripts/qplib_eval.py MODEL.qplib SOLUTION        # SOR --solution-out file
    python3 scripts/qplib_eval.py MODEL.qplib QPLIB_xxxx.sol  # published QPLIB point
    python3 scripts/qplib_eval.py MODEL.qplib --zero          # evaluate x = 0
    add --machine for one "PYEVAL key=value ..." line.

All 3-letter types are handled: objective L/D/C/Q, variables C/B/M/I/G,
constraints N/B/L/D/C/Q.
"""
import math
import os
import sys


def tokens(path):
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                yield line.split()


def read(path):
    t = tokens(path)
    nxt = lambda: next(t)
    name = nxt()[0]
    kind = nxt()[0].upper()
    ob, vt, ct = kind
    sense = nxt()[0].lower()
    n = int(nxt()[0])
    # Constraint letters N (none) and B (bounds only) have no m, no A and
    # no row bounds; D/C/Q have all of those PLUS per-row Hessians.
    has_rows = ct in "LDCQ"
    m = int(nxt()[0]) if has_rows else 0
    q = []  # (i, j, v) as listed; see evaluate() for their weight
    if ob != "L":  # a linear objective has no Hessian block at all
        for _ in range(int(nxt()[0])):
            i, j, v = nxt(); q.append((int(i) - 1, int(j) - 1, float(v)))
    c = [float(nxt()[0])] * n
    for _ in range(int(nxt()[0])):
        i, v = nxt(); c[int(i) - 1] = float(v)
    const = float(nxt()[0])
    qc = []  # (row, i, j, v): constraint Hessian quadruples
    if ct in "DCQ":
        for _ in range(int(nxt()[0])):
            k, i, j, v = nxt()
            qc.append((int(k) - 1, int(i) - 1, int(j) - 1, float(v)))
    a = []
    if has_rows:
        for _ in range(int(nxt()[0])):
            i, j, v = nxt(); a.append((int(i) - 1, int(j) - 1, float(v)))
    inf = float(nxt()[0])  # float() maps 1.79769313486232E+308 to inf: fine

    def vec(k):
        d = [float(nxt()[0])] * k
        for _ in range(int(nxt()[0])):
            i, v = nxt(); d[int(i) - 1] = float(v)
        return d

    cl = vec(m) if has_rows else []
    cu = vec(m) if has_rows else []
    # Type codes: 0 continuous, 1 integer, 2 an EXPLICIT binary (its bounds
    # are forced to [0, 1] whatever the file lists, mirroring the C++
    # reader's QplibVarType::Binary and tests/data/qcqp_tiny.qplib, which
    # exercises exactly this: var 2 carries code 2 with listed bounds
    # [0, 1e20]).  In practice code 2 is never used: QPLIB's own
    # instancedata.csv splits NBINVARS from NINTVARS by BOUNDS, not by code
    # -- QPLIB_3562 is counted 7 binary + 56 integer and all 63 of its
    # variables carry code 1, while QPLIB_3496's code-1 variables are
    # bounded [0, 4] and counted integer -- so a code-1 variable is binary
    # here too when its listed bounds are already exactly [0, 1]. No file in
    # the library uses a code outside {0, 1, 2}, so another code is an error
    # here rather than a guess.
    if vt == "B":  # all binary: bounds and types omitted
        xl, xu, typ = [0.0] * n, [1.0] * n, [1] * n
    else:
        xl = vec(n)
        xu = vec(n)
        if vt in "MG":
            typ = [int(float(nxt()[0]))] * n
            for _ in range(int(nxt()[0])):
                i, v = nxt(); typ[int(i) - 1] = int(float(v))
        else:
            typ = [1 if vt == "I" else 0] * n
    if any(k not in (0, 1, 2) for k in typ):
        sys.exit(f"{name}: variable type code outside {{0, 1, 2}}")
    for j, k in enumerate(typ):
        if k == 2:  # explicit binary: bounds are [0, 1], listed or not
            xl[j], xu[j] = 0.0, 1.0
    # Trailer: start point, (row duals), bound duals, names.
    vec(n)
    if has_rows:
        vec(m)
    vec(n)
    vnames = {}
    for _ in range(int(nxt()[0])):
        i, nm = nxt()[:2]; vnames[int(i) - 1] = nm
    for _ in range(int(nxt()[0])):
        nxt()
    leftover = sum(1 for _ in t)
    return dict(name=name, kind=kind, sense=sense, n=n, m=m, q=q, c=c, const=const,
                qc=qc, a=a, inf=inf, cl=cl, cu=cu, xl=xl, xu=xu, typ=typ,
                vnames=vnames, leftover=leftover)


def evaluate(p, x):
    # math.fsum: exact summation, so the check does not depend on term order
    # (some rows have 1e5 mixed-sign terms and are compared at 1e-6 rel).
    # Every listed term contributes 0.5 * v * x_i * x_j, off-diagonal ones
    # included -- the format stores ONE triangle of Q and does not symmetrise
    # it (doc.html: Q^i is lower-left triangular, objective 1/2 x'Q^0 x).
    # Settled empirically too: the full-weight reading gives ~0 on
    # QPLIB_8515 at a KKT point, the half-weight one QPLIB's published 320.0.
    # The same rule applies to constraint Hessians (1/2 x'Q^i x in the doc);
    # checked against every published solution point by qplib_parse_all.py.
    terms = [p["const"]] + [ci * xi for ci, xi in zip(p["c"], x)]
    terms += [0.5 * v * x[i] * x[j] for i, j, v in p["q"]]
    obj = math.fsum(terms)
    inf = p["inf"]
    bound = integ = 0.0
    for j in range(p["n"]):
        if p["xl"][j] > -inf: bound = max(bound, p["xl"][j] - x[j])
        if p["xu"][j] < inf: bound = max(bound, x[j] - p["xu"][j])
        if p["typ"][j] != 0: integ = max(integ, abs(x[j] - round(x[j])))
    rows = [[] for _ in range(p["m"])]
    for i, j, v in p["a"]:
        rows[i].append(v * x[j])
    isq = [False] * p["m"]
    for k, i, j, v in p["qc"]:
        rows[k].append(0.5 * v * x[i] * x[j]); isq[k] = True
    row = qrow = 0.0
    for i in range(p["m"]):
        ax = math.fsum(rows[i])
        v = 0.0
        if p["cl"][i] > -inf: v = max(v, p["cl"][i] - ax)
        if p["cu"][i] < inf: v = max(v, ax - p["cu"][i])
        row = max(row, v)
        if isq[i]: qrow = max(qrow, v)
    return obj, dict(rows=row, qrows=qrow, bounds=bound, integrality=integ)


def default_var_name(p, j):
    """The GAMS name QPLIB's .sol files use, when nothing better is known.

    GAMS prefixes by KIND, not by the file's type code: b for a binary (an
    integer bounded [0, 1]), i for a general integer, x for a continuous
    variable.  The index is j + 2, because the GAMS model of most instances
    declares `objvar` first and the instance's variable 1 is the second.
    This is a fallback: it is right for the majority family only, so
    read_point prefers a QPLIB_xxxx.varnames sidecar (see
    scripts/fetch_qplib_varnames.py) whenever one is there.
    """
    t = p["typ"][j]
    if t == 0:
        pre = "x"
    elif p["xl"][j] == 0.0 and p["xu"][j] == 1.0:
        pre = "b"
    else:
        pre = "i"
    return f"{pre}{j + 2}"


def load_varnames(model_path, n):
    """QPLIB_xxxx.varnames beside the instance: the GAMS name of variable
    1..n, one per line, taken from the instance's own published GAMS source.
    Absent -> None, and read_point falls back to default_var_name."""
    side = model_path[:-6] + ".varnames" if model_path.endswith(".qplib") else None
    if not side or not os.path.exists(side):
        return None
    names = [l.strip() for l in open(side) if l.strip()]
    if len(names) != n:
        sys.exit(f"{side}: {len(names)} names for {n} variables")
    return names


def read_point(p, path, model_path=None):
    """SOR's --solution-out file (an "x <n> v1 ... vn" line) or a published
    QPLIB .sol ("name value" lines, "objvar" = the published objective,
    unlisted variables = 0)."""
    lines = [l.split() for l in open(path)]
    for f in lines:
        if len(f) > 2 and f[0] == "x":
            return [float(v) for v in f[2:]], None
    x = [0.0] * p["n"]
    objvar = None
    side = load_varnames(model_path, p["n"]) if model_path else None
    names = {}
    for j in range(p["n"]):
        # The .qplib's own name override wins, then the GAMS sidecar, then
        # the fallback guess.
        nm = p["vnames"].get(j) or (side[j] if side else None) or default_var_name(p, j)
        names[nm] = j
    for f in lines:
        if not f:
            continue
        if f[0] == "objvar":
            objvar = float(f[1])
            # "objvar" is usually a GAMS variable the QPLIB model does not
            # have, and the line only states the objective.  In a few
            # instances it IS variable 1 of the model (QPLIB_10035 and its
            # three siblings), and dropping the line would leave that
            # variable at 0 and make the point look infeasible by exactly
            # the published objective.  Fall through when the name resolves.
            if "objvar" not in names:
                continue
        if f[0] not in names:
            sys.exit(f"{path}: unknown variable {f[0]} (name/index mapping disagrees)")
        x[names[f[0]]] = float(f[1])
    return x, objvar


def main():
    args = [s for s in sys.argv[1:] if s != "--machine"]
    machine = len(args) != len(sys.argv) - 1
    p = read(args[0])
    objvar = None
    if args[1] == "--zero":
        x = [0.0] * p["n"]
    else:
        x, objvar = read_point(p, args[1], args[0])
        if len(x) != p["n"]:
            sys.exit("solution has the wrong length")
    obj, viol = evaluate(p, x)
    worst = max(viol["rows"], viol["bounds"], viol["integrality"])
    if machine:
        print(f"PYEVAL name={p['name']} class={p['kind']} obj={obj!r} "
              f"objvar={objvar!r} viol={worst!r} qcviol={viol['qrows']!r} "
              f"nqcentries={len(p['qc'])} leftover={p['leftover']}")
        return
    print(f"{p['name']}: sense {p['sense']}  objective {obj:.12e}  max violation {worst:.3e}"
          f"  (rows {viol['rows']:.3e}, quadratic rows {viol['qrows']:.3e}, bounds"
          f" {viol['bounds']:.3e}, integrality {viol['integrality']:.3e})")
    if objvar is not None:
        print(f"  objective stated in the .sol file: {objvar:.12e}")
    if p["leftover"]:
        print(f"  WARNING: {p['leftover']} unread lines after the trailer")


if __name__ == "__main__":
    main()
