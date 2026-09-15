#!/usr/bin/env python3
"""Tests for the presolve-opportunity scanner.

Every fixture is a tiny MPS file written here, so each assertion is about a
model whose reductions can be counted by hand.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sor_presolve_scan", ROOT / "scripts" / "analyze_presolve_opportunities.py")
assert SPEC is not None and SPEC.loader is not None
scan_mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scan_mod
SPEC.loader.exec_module(scan_mod)

INF = float("inf")


def mps(body: str) -> str:
    """Dedent a fixture and keep its leading-space column discipline."""
    return textwrap.dedent(body).lstrip("\n")


class MpsFixture:
    """A temporary .mps file, usable as a context manager."""

    def __init__(self, text: str, name: str = "fixture.mps") -> None:
        self.text, self.name = text, name

    def __enter__(self) -> Path:
        self._dir = tempfile.TemporaryDirectory()
        p = Path(self._dir.name) / self.name
        p.write_text(self.text)
        return p

    def __exit__(self, *exc: object) -> None:
        self._dir.cleanup()


def parse(text: str) -> scan_mod.Lp:
    with MpsFixture(text) as p:
        return scan_mod.read_mps(p)


def counts(text: str) -> dict[str, int]:
    return scan_mod.analyze(parse(text))


# A two-row, two-column model used as the base for several fixtures.
BASIC = mps("""
NAME          BASIC
ROWS
 N  COST
 L  R1
 G  R2
COLUMNS
    X         COST      1.0        R1        1.0
    X         R2        1.0
    Y         COST      2.0        R1        1.0
RHS
    RHS       R1        4.0        R2        1.0
ENDATA
""")


class ReaderTests(unittest.TestCase):
    def test_dimensions_and_bounds_defaults(self) -> None:
        lp = parse(BASIC)
        self.assertEqual((lp.n_rows, lp.n_cols, lp.nnz), (2, 2, 3))
        self.assertEqual(lp.row_names, ["R1", "R2"])
        self.assertEqual(lp.col_names, ["X", "Y"])
        # Default column bounds are [0, +inf); L/G rows are one-sided.
        self.assertEqual(lp.col_lo, [0.0, 0.0])
        self.assertEqual(lp.col_hi, [INF, INF])
        self.assertEqual(lp.row_lo, [-INF, 1.0])
        self.assertEqual(lp.row_hi, [4.0, INF])

    def test_objective_row_is_not_a_constraint(self) -> None:
        lp = parse(BASIC)
        self.assertNotIn("COST", lp.row_names)

    def test_later_n_rows_are_discarded(self) -> None:
        lp = parse(mps("""
            NAME
            ROWS
             N  COST
             N  FREE
             E  R1
            COLUMNS
                X         COST      1.0        FREE      9.0
                X         R1        1.0
            RHS
                RHS       R1        2.0
            ENDATA
            """))
        # Only the first N row is the objective; FREE's coefficient is dropped
        # rather than becoming a constraint entry.
        self.assertEqual(lp.n_rows, 1)
        self.assertEqual(lp.nnz, 1)

    def test_duplicate_column_entries_are_summed(self) -> None:
        lp = parse(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                X         R1        1.0
                X         R1        2.5
            RHS
                RHS       R1        1.0
            ENDATA
            """))
        self.assertEqual(lp.nnz, 1)
        self.assertEqual(lp.rows[0], {0: 3.5})

    def test_explicit_zero_stays_structural(self) -> None:
        # sparse::from_triplets keeps a zero value, so the scanner must too --
        # otherwise it would report a different sparsity than the solver sees.
        lp = parse(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                X         R1        0.0
                Y         R1        1.0
            RHS
                RHS       R1        1.0
            ENDATA
            """))
        self.assertEqual(lp.nnz, 2)
        self.assertEqual(scan_mod.analyze(lp)["empty_columns"], 0)

    def test_ranges_semantics(self) -> None:
        lp = parse(mps("""
            NAME
            ROWS
             N  COST
             L  RL
             G  RG
             E  REP
             E  REN
            COLUMNS
                X         RL        1.0        RG        1.0
                X         REP       1.0        REN       1.0
            RHS
                RHS       RL        10.0       RG        1.0
                RHS       REP       5.0        REN       5.0
            RANGES
                RNG       RL        4.0        RG        -4.0
                RNG       REP       2.0        REN       -2.0
            ENDATA
            """))
        # L: [rhs-|R|, rhs].  G: [rhs, rhs+|R|].
        self.assertEqual((lp.row_lo[0], lp.row_hi[0]), (6.0, 10.0))
        self.assertEqual((lp.row_lo[1], lp.row_hi[1]), (1.0, 5.0))
        # E: the SIGN of R picks the side the range extends to.
        self.assertEqual((lp.row_lo[2], lp.row_hi[2]), (5.0, 7.0))
        self.assertEqual((lp.row_lo[3], lp.row_hi[3]), (3.0, 5.0))

    def test_bound_types(self) -> None:
        lp = parse(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                A         R1        1.0
                B         R1        1.0
                C         R1        1.0
                D         R1        1.0
                E         R1        1.0
                F         R1        1.0
            RHS
                RHS       R1        1.0
            BOUNDS
             UP BND       A         5.0
             FX BND       B         3.0
             FR BND       C
             MI BND       D
             BV BND       E
             LO BND       F         -2.0
            ENDATA
            """))
        self.assertEqual((lp.col_lo[0], lp.col_hi[0]), (0.0, 5.0))
        self.assertEqual((lp.col_lo[1], lp.col_hi[1]), (3.0, 3.0))
        self.assertEqual((lp.col_lo[2], lp.col_hi[2]), (-INF, INF))
        self.assertEqual((lp.col_lo[3], lp.col_hi[3]), (-INF, INF))
        self.assertEqual((lp.col_lo[4], lp.col_hi[4]), (0.0, 1.0))
        self.assertTrue(lp.is_integer[4])
        self.assertEqual((lp.col_lo[5], lp.col_hi[5]), (-2.0, INF))

    def test_negative_up_bound_implies_free_below(self) -> None:
        lp = parse(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                A         R1        1.0
                B         R1        1.0
            RHS
                RHS       R1        1.0
            BOUNDS
             UP BND       A         -5.0
             LO BND       B         -9.0
             UP BND       B         -5.0
            ENDATA
            """))
        # A had no explicit lower bound, so the negative UP frees it below.
        self.assertEqual((lp.col_lo[0], lp.col_hi[0]), (-INF, -5.0))
        # B did, so the quirk must NOT fire.
        self.assertEqual((lp.col_lo[1], lp.col_hi[1]), (-9.0, -5.0))

    def test_integer_markers(self) -> None:
        lp = parse(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                MARKER                 'MARKER'                 'INTORG'
                A         R1        1.0
                MARKER                 'MARKER'                 'INTEND'
                B         R1        1.0
            RHS
                RHS       R1        1.0
            ENDATA
            """))
        self.assertEqual(lp.is_integer, [True, False])

    def test_rhs_on_objective_is_not_a_row(self) -> None:
        lp = parse(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                X         COST      1.0        R1        1.0
            RHS
                RHS       COST      7.0        R1        1.0
            ENDATA
            """))
        self.assertEqual(lp.n_rows, 1)
        self.assertEqual(lp.row_lo, [1.0])

    def test_objsense_maximize(self) -> None:
        for header in ("OBJSENSE\n    MAX\n", "OBJSENSE  MAXIMIZE\n"):
            lp = parse(header + mps("""
                ROWS
                 N  COST
                 E  R1
                COLUMNS
                    X         R1        1.0
                RHS
                    RHS       R1        1.0
                ENDATA
                """))
            self.assertTrue(lp.maximize, header)

    def test_comments_and_blank_lines_are_skipped(self) -> None:
        lp = parse(mps("""
            * a comment
            NAME

            ROWS
             N  COST
             E  R1
            * another
            COLUMNS
                X         R1        1.0
            RHS
                RHS       R1        1.0
            ENDATA
            """))
        self.assertEqual((lp.n_rows, lp.n_cols, lp.nnz), (1, 1, 1))


class FixedFormatTests(unittest.TestCase):
    # Column starts: f1 at 1, f2 at 4, f3 at 14, f4 at 24 (0-based).
    SPACED = ("NAME          SPACED\n"
              "ROWS\n"
              " E  DEDO3 1R\n"
              " E  DEDO3 2R\n"
              " N  COST\n"
              "COLUMNS\n"
              "    X         DEDO3 1R       1.0\n"
              "    X         DEDO3 2R       2.0\n"
              "RHS\n"
              "    RHS       DEDO3 1R       1.0\n"
              "ENDATA\n")

    def test_free_format_rejects_names_with_spaces(self) -> None:
        with MpsFixture(self.SPACED) as p:
            with self.assertRaises(scan_mod.MpsError):
                scan_mod.read_mps(p, fixed_format=False)

    def test_fixed_format_retry_succeeds(self) -> None:
        with MpsFixture(self.SPACED) as p:
            lp, used_fixed = scan_mod.read_mps_auto(p)
        self.assertTrue(used_fixed)
        self.assertEqual(lp.row_names, ["DEDO3 1R", "DEDO3 2R"])
        self.assertEqual((lp.n_rows, lp.n_cols, lp.nnz), (2, 1, 2))

    def test_plain_model_does_not_need_fixed_format(self) -> None:
        with MpsFixture(BASIC) as p:
            _, used_fixed = scan_mod.read_mps_auto(p)
        self.assertFalse(used_fixed)


class ErrorTests(unittest.TestCase):
    def _expect_error(self, text: str) -> str:
        with MpsFixture(text) as p:
            with self.assertRaises(scan_mod.MpsError) as ctx:
                scan_mod.read_mps_auto(p)
        return str(ctx.exception)

    def test_unknown_row_in_columns(self) -> None:
        self.assertIn("NOPE", self._expect_error(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                X         NOPE      1.0
            ENDATA
            """)))

    def test_duplicate_row_name(self) -> None:
        self.assertIn("duplicate row name", self._expect_error(mps("""
            NAME
            ROWS
             N  COST
             E  R1
             E  R1
            COLUMNS
                X         R1        1.0
            ENDATA
            """)))

    def test_unparsable_number(self) -> None:
        self.assertIn("not a number", self._expect_error(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                X         R1        one
            ENDATA
            """)))

    def test_unknown_row_kind(self) -> None:
        self.assertIn("unknown row kind", self._expect_error(mps("""
            NAME
            ROWS
             Q  R1
            COLUMNS
            ENDATA
            """)))

    def test_bound_without_required_value(self) -> None:
        self.assertIn("needs a value", self._expect_error(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                X         R1        1.0
            BOUNDS
             UP BND       X
            ENDATA
            """)))

    def test_empty_file_is_an_error(self) -> None:
        self.assertIn("no ROWS/COLUMNS", self._expect_error("* nothing here\n"))

    def test_unrecognised_layout_is_an_error_not_an_empty_model(self) -> None:
        # Every line indented, so no section header is ever recognised. This
        # must fail loudly: reporting it as a real 0x0 model would silently
        # understate a sweep.
        self.assertIn("no ROWS/COLUMNS", self._expect_error(
            "   ROWS\n    N  COST\n    E  R1\nENDATA\n"))

    def test_unknown_section_is_ignored_not_fatal(self) -> None:
        # The C++ reader is non-strict by default and warns instead of failing.
        lp = parse(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                X         R1        1.0
            SOSSECTION
                whatever  here
            RHS
                RHS       R1        1.0
            ENDATA
            """))
        self.assertEqual((lp.n_rows, lp.nnz), (1, 1))


class SingletonAndEmptyTests(unittest.TestCase):
    def test_singleton_and_empty_columns(self) -> None:
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             E  R1
             E  R2
            COLUMNS
                SINGLE    R1        1.0
                DOUBLE    R1        1.0        R2        1.0
                EMPTY     COST      1.0
            RHS
                RHS       R1        1.0        R2        1.0
            ENDATA
            """))
        # EMPTY appears only in the objective, so it has no entry in A.
        self.assertEqual(c["singleton_columns"], 1)
        self.assertEqual(c["empty_columns"], 1)

    def test_singleton_rows(self) -> None:
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             L  R1
             E  R2
            COLUMNS
                X         R1        1.0        R2        1.0
                Y         R2        1.0
            RHS
                RHS       R1        3.0        R2        1.0
            ENDATA
            """))
        self.assertEqual(c["singleton_rows"], 1)

    def test_empty_row_is_neither_singleton_nor_duplicate(self) -> None:
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             E  R1
             E  R2
            COLUMNS
                X         R1        1.0
            RHS
                RHS       R1        1.0
            ENDATA
            """))
        self.assertEqual(c["singleton_rows"], 1)
        self.assertEqual(c["duplicate_rows"], 0)
        self.assertEqual(c["forcing_rows"], 0)


class DoubletonEquationTests(unittest.TestCase):
    BASE = mps("""
        NAME
        ROWS
         N  COST
         {k1}  R1
         {k2}  R2
        COLUMNS
            X         R1        1.0        R2        1.0
            Y         R1        1.0        R2        1.0
            Z         R2        1.0
        RHS
            RHS       R1        1.0        R2        1.0
        ENDATA
        """)

    def test_only_equalities_of_length_two_count(self) -> None:
        # R1 is the doubleton; R2 has three entries.
        self.assertEqual(counts(self.BASE.format(k1="E", k2="E"))
                         ["doubleton_equations"], 1)
        # An inequality of length two is not a doubleton EQUATION.
        self.assertEqual(counts(self.BASE.format(k1="L", k2="E"))
                         ["doubleton_equations"], 0)

    def test_ranged_equality_of_width_zero_still_counts(self) -> None:
        # A RANGES entry of 0 leaves row_lo == row_hi, i.e. still an equation.
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                X         R1        1.0
                Y         R1        1.0
            RHS
                RHS       R1        1.0
            RANGES
                RNG       R1        0.0
            ENDATA
            """))
        self.assertEqual(c["doubleton_equations"], 1)

    def test_ranged_equality_of_positive_width_does_not_count(self) -> None:
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             E  R1
            COLUMNS
                X         R1        1.0
                Y         R1        1.0
            RHS
                RHS       R1        1.0
            RANGES
                RNG       R1        2.0
            ENDATA
            """))
        self.assertEqual(c["doubleton_equations"], 0)


class DuplicateRowTests(unittest.TestCase):
    def test_identical_and_scaled_rows_form_one_group(self) -> None:
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             L  R1
             L  R2
             L  R3
            COLUMNS
                X         R1        1.0        R2        1.0
                X         R3        2.0
                Y         R1        3.0        R2        3.0
                Y         R3        6.0
            RHS
                RHS       R1        1.0        R2        1.0
                RHS       R3        2.0
            ENDATA
            """))
        # Three parallel rows -> two are duplicates of the first.
        self.assertEqual(c["duplicate_rows"], 2)

    def test_negatively_scaled_rows_are_parallel(self) -> None:
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             L  R1
             G  R2
            COLUMNS
                X         R1        1.0        R2        -1.0
                Y         R1        2.0        R2        -2.0
            RHS
                RHS       R1        1.0        R2        -1.0
            ENDATA
            """))
        self.assertEqual(c["duplicate_rows"], 1)

    def test_same_pattern_different_ratio_is_not_a_duplicate(self) -> None:
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             L  R1
             L  R2
            COLUMNS
                X         R1        1.0        R2        1.0
                Y         R1        2.0        R2        3.0
            RHS
                RHS       R1        1.0        R2        1.0
            ENDATA
            """))
        self.assertEqual(c["duplicate_rows"], 0)

    def test_different_pattern_is_not_a_duplicate(self) -> None:
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             L  R1
             L  R2
            COLUMNS
                X         R1        1.0        R2        1.0
                Y         R1        1.0
                Z         R2        1.0
            RHS
                RHS       R1        1.0        R2        1.0
            ENDATA
            """))
        self.assertEqual(c["duplicate_rows"], 0)

    def test_row_bounds_do_not_affect_parallelism(self) -> None:
        # Parallel rows are a matrix property; a presolver still has to look at
        # them, whatever their right-hand sides are.
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             L  R1
             L  R2
            COLUMNS
                X         R1        1.0        R2        1.0
            RHS
                RHS       R1        1.0        R2        99.0
            ENDATA
            """))
        self.assertEqual(c["duplicate_rows"], 1)

    def test_parallel_key_normalizes_scale(self) -> None:
        k = scan_mod.parallel_key
        self.assertEqual(k({0: 1.0, 1: 2.0}), k({0: -4.0, 1: -8.0}))
        self.assertNotEqual(k({0: 1.0, 1: 2.0}), k({0: 1.0, 1: 2.5}))
        # A leading explicit zero must not be used as the normalizer.
        self.assertEqual(k({0: 0.0, 1: 1.0}), k({0: 0.0, 1: 7.0}))


class ForcingRowTests(unittest.TestCase):
    def _one_row(self, kind: str, rhs: float,
                 upper: float | None = None) -> dict[str, int]:
        # Built line by line rather than through mps(): an optional line at a
        # different indent would change what textwrap.dedent strips.
        lines = ["NAME",
                 "ROWS",
                 " N  COST",
                 f" {kind}  R1",
                 "COLUMNS",
                 "    X         R1        1.0",
                 "    Y         R1        1.0",
                 "RHS",
                 f"    RHS       R1        {rhs}",
                 "BOUNDS",
                 " UP BND       X         2.0"]
        if upper is not None:
            lines.append(f" UP BND       Y         {upper}")
        lines.append("ENDATA")
        return counts("\n".join(lines) + "\n")

    def test_min_activity_equal_to_upper_bound_is_forcing(self) -> None:
        # x, y >= 0 so min activity is 0; "x + y <= 0" pins both to zero.
        self.assertEqual(self._one_row("L", 0.0)["forcing_rows"], 1)

    def test_max_activity_equal_to_lower_bound_is_forcing(self) -> None:
        # x <= 2, y <= 3 so max activity is 5; "x + y >= 5" pins both up.
        self.assertEqual(self._one_row("G", 5.0, upper=3.0)["forcing_rows"], 1)

    def test_slack_row_is_not_forcing(self) -> None:
        self.assertEqual(self._one_row("L", 1.0)["forcing_rows"], 0)

    def test_infeasible_row_is_not_counted_as_forcing(self) -> None:
        # min activity 0 > -1: infeasible, which is a different finding.
        self.assertEqual(self._one_row("L", -1.0)["forcing_rows"], 0)

    def test_unbounded_activity_is_not_forcing(self) -> None:
        # y has no upper bound, so max activity is +inf and the G row cannot be
        # forcing however large its right-hand side is.
        self.assertEqual(self._one_row("G", 5.0)["forcing_rows"], 0)

    def test_equality_row_at_its_min_activity_is_forcing(self) -> None:
        self.assertEqual(self._one_row("E", 0.0)["forcing_rows"], 1)

    def test_explicit_zero_coefficient_does_not_poison_infinite_bounds(self) -> None:
        # 0 * inf is NaN if computed naively; the coefficient contributes
        # nothing and the row is still forcing through its other column.
        c = counts(mps("""
            NAME
            ROWS
             N  COST
             L  R1
            COLUMNS
                FREEVAR   R1        0.0
                X         R1        1.0
            RHS
                RHS       R1        0.0
            BOUNDS
             FR BND       FREEVAR
            ENDATA
            """))
        self.assertEqual(c["forcing_rows"], 1)

    def test_activity_bounds_helper(self) -> None:
        lo = [0.0, -INF]
        hi = [2.0, 3.0]
        self.assertEqual(scan_mod.activity_bounds({0: 1.0}, lo, hi), (0.0, 2.0))
        self.assertEqual(scan_mod.activity_bounds({0: -1.0}, lo, hi), (-2.0, 0.0))
        self.assertEqual(scan_mod.activity_bounds({1: 1.0}, lo, hi), (-INF, 3.0))
        self.assertEqual(scan_mod.activity_bounds({}, lo, hi), (0.0, 0.0))

    def test_tolerance_is_relative(self) -> None:
        lp = parse(mps("""
            NAME
            ROWS
             N  COST
             L  R1
            COLUMNS
                X         R1        1.0
            RHS
                RHS       R1        1.0e-9
            ENDATA
            """))
        # min activity 0 vs rhs 1e-9: within the default relative tolerance.
        self.assertEqual(scan_mod.analyze(lp, tol=1e-6)["forcing_rows"], 1)
        self.assertEqual(scan_mod.analyze(lp, tol=1e-12)["forcing_rows"], 0)


class CliTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = scan_mod.main(argv)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        return code, out.getvalue(), err.getvalue()

    def test_csv_to_stdout(self) -> None:
        with MpsFixture(BASIC, "basic.mps") as p:
            code, out, _ = self._run([str(p)])
        self.assertEqual(code, 0)
        rows = list(csv.DictReader(io.StringIO(out)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["instance"], "basic")
        self.assertEqual(rows[0]["rows"], "2")
        self.assertEqual(rows[0]["nnz"], "3")
        self.assertEqual(rows[0]["error"], "")

    def test_csv_and_json_files(self) -> None:
        with MpsFixture(BASIC, "basic.mps") as p:
            with tempfile.TemporaryDirectory() as d:
                csv_path, json_path = Path(d) / "a.csv", Path(d) / "a.json"
                code, out, _ = self._run(
                    [str(p), "--csv", str(csv_path), "--json", str(json_path)])
                self.assertEqual(code, 0)
                self.assertEqual(out, "")
                payload = json.loads(json_path.read_text())
                self.assertEqual(len(payload["instances"]), 1)
                self.assertEqual(payload["instances"][0]["singleton_columns"], 1)
                with csv_path.open() as fh:
                    self.assertEqual(list(csv.DictReader(fh))[0]["cols"], "2")

    def test_directory_scan_finds_every_mps(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.mps", "b.mps"):
                (Path(d) / name).write_text(BASIC)
            (Path(d) / "notes.txt").write_text("ignored")
            code, out, _ = self._run([d])
        self.assertEqual(code, 0)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(out)))), 2)

    def test_unreadable_model_is_reported_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "good.mps").write_text(BASIC)
            (Path(d) / "bad.mps").write_text("ROWS\n Q  R1\nENDATA\n")
            code, out, err = self._run([d])
        self.assertEqual(code, 1)       # a failed model fails the run
        rows = {r["instance"]: r for r in csv.DictReader(io.StringIO(out))}
        self.assertEqual(rows["good"]["error"], "")
        self.assertIn("unknown row kind", rows["bad"]["error"])
        # A broken model must not blank out the models that did parse.
        self.assertEqual(rows["good"]["rows"], "2")
        self.assertIn("bad.mps", err)

    def test_missing_path_is_a_warning_until_nothing_is_left(self) -> None:
        code, _, err = self._run(["/no/such/place"])
        self.assertEqual(code, 2)
        self.assertIn("no such model", err)

    def test_negative_tolerance_is_rejected(self) -> None:
        with MpsFixture(BASIC) as p:
            with self.assertRaises(SystemExit) as ctx:
                self._run([str(p), "--tol", "-1"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_nan_tolerance_is_rejected(self) -> None:
        with MpsFixture(BASIC) as p:
            with self.assertRaises(SystemExit) as ctx:
                self._run([str(p), "--tol", "nan"])
        self.assertNotEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
