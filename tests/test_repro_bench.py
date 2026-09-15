#!/usr/bin/env python3
"""Tests for the reproducible benchmark runner.

Two things are worth pinning down here: the compare.py command the wrapper
generates (it is the whole point of the wrapper, and a dropped flag would
silently measure something other than what was asked for), and the robust
statistics (a median/MAD that quietly mishandles zeros or a single sample would
report perfect stability on nonsense).

Nothing here runs a solver.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sor_repro_bench", ROOT / "scripts" / "repro_bench.py")
assert SPEC is not None and SPEC.loader is not None
rb = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rb
SPEC.loader.exec_module(rb)

compare_runs = rb.compare_runs
compare = rb.compare


def flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


class CommandTests(unittest.TestCase):
    def base(self, **kw: object) -> list[str]:
        args = dict(solvers="sor:simplex", time_limit=30.0, repetitions=5,
                    warmups=1, cpu=None, jsonl=Path("out.jsonl"))
        args.update(kw)
        return rb.build_compare_command(["model.mps"], **args)  # type: ignore[arg-type]

    def test_required_flags_are_all_present(self) -> None:
        cmd = self.base()
        self.assertEqual(cmd[0], sys.executable)
        self.assertTrue(cmd[1].endswith("compare.py"))
        self.assertIn("model.mps", cmd)
        self.assertEqual(flag(cmd, "--solvers"), "sor:simplex")
        self.assertEqual(flag(cmd, "--time-limit"), "30.0")
        self.assertEqual(flag(cmd, "--repetitions"), "5")
        self.assertEqual(flag(cmd, "--warmups"), "1")
        self.assertEqual(flag(cmd, "--jsonl"), "out.jsonl")

    def test_several_models_are_all_passed_through(self) -> None:
        cmd = rb.build_compare_command(
            ["a.mps", "b.mps", "dir/"], solvers="sor:simplex", time_limit=1.0,
            repetitions=1, warmups=0, cpu=None, jsonl=Path("o.jsonl"))
        for model in ("a.mps", "b.mps", "dir/"):
            self.assertIn(model, cmd)
        # Models must precede the first option, or argparse reads them as values.
        self.assertLess(cmd.index("dir/"), cmd.index("--solvers"))

    def test_cpu_pinning_is_only_added_when_requested(self) -> None:
        self.assertNotIn("--cpu", self.base())
        self.assertEqual(flag(self.base(cpu=3), "--cpu"), "3")
        # CPU 0 is a real CPU, not "unset".
        self.assertEqual(flag(self.base(cpu=0), "--cpu"), "0")

    def test_zero_warmups_and_one_repetition_are_still_emitted(self) -> None:
        cmd = self.base(warmups=0, repetitions=1)
        self.assertEqual(flag(cmd, "--warmups"), "0")
        self.assertEqual(flag(cmd, "--repetitions"), "1")

    def test_optional_passthroughs(self) -> None:
        cmd = self.base(tol=1e-8, method="dual", pricing="devex",
                        basis_update="ft", max_iter=1000, seed=7, limit=12)
        self.assertEqual(flag(cmd, "--tol"), "1e-08")
        self.assertEqual(flag(cmd, "--method"), "dual")
        self.assertEqual(flag(cmd, "--pricing"), "devex")
        self.assertEqual(flag(cmd, "--basis-update"), "ft")
        self.assertEqual(flag(cmd, "--max-iter"), "1000")
        self.assertEqual(flag(cmd, "--seed"), "7")
        self.assertEqual(flag(cmd, "--limit"), "12")

    def test_optional_passthroughs_absent_by_default(self) -> None:
        cmd = self.base()
        for name in ("--tol", "--method", "--pricing", "--basis-update",
                     "--max-iter", "--limit", "--allow-unchecked",
                     "--allow-unavailable"):
            self.assertNotIn(name, cmd, name)

    def test_allow_flags_are_bare_switches(self) -> None:
        cmd = self.base(allow_unchecked=True, allow_unavailable=True)
        self.assertIn("--allow-unchecked", cmd)
        self.assertIn("--allow-unavailable", cmd)
        # They take no value, so the next token must be another flag or nothing.
        idx = cmd.index("--allow-unchecked")
        self.assertTrue(idx + 1 == len(cmd) or cmd[idx + 1].startswith("--"))

    def test_generated_command_is_accepted_by_compare_py(self) -> None:
        # The strongest check available without running a solver: hand the
        # generated argv to compare.py's own parser and require it to parse.
        cmd = self.base(cpu=1, tol=1e-9, method="primal", pricing="dse",
                        basis_update="product", max_iter=99, seed=3, limit=4,
                        allow_unchecked=True, allow_unavailable=True)
        argv = cmd[2:]                       # drop python and the script path
        parser_args = self._parse_with_compare(argv)
        self.assertEqual(parser_args.repetitions, 5)
        self.assertEqual(parser_args.warmups, 1)
        self.assertEqual(parser_args.cpu, 1)
        self.assertEqual(parser_args.method, "primal")
        self.assertEqual(parser_args.pricing, "dse")
        self.assertTrue(parser_args.allow_unchecked)

    def _parse_with_compare(self, argv: list[str]):
        """Run compare.py's own argument parser over `argv` and stop there.

        compare.main() reads sys.argv and would go on to run solvers, so
        parse_args is stubbed to capture the namespace and raise SystemExit
        immediately after parsing. That still exercises the real parser,
        including its choices= and type= validation.
        """
        import argparse
        captured: dict[str, object] = {}
        real_parse = argparse.ArgumentParser.parse_args

        def spy(self, args=None, namespace=None):  # noqa: ANN001
            ns = real_parse(self, args, namespace)
            captured["ns"] = ns
            raise SystemExit(0)

        old_argv = sys.argv
        sys.argv = ["compare.py", *argv]
        argparse.ArgumentParser.parse_args = spy  # type: ignore[method-assign]
        try:
            with self.assertRaises(SystemExit) as ctx:
                compare.main()
            # A parse FAILURE exits 2 before the spy ever captures anything.
            self.assertEqual(ctx.exception.code, 0)
        finally:
            argparse.ArgumentParser.parse_args = real_parse  # type: ignore[method-assign]
            sys.argv = old_argv
        return captured["ns"]


class StatisticsTests(unittest.TestCase):
    def test_median_odd_and_even(self) -> None:
        self.assertEqual(rb.median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(rb.median([1.0, 2.0, 3.0, 4.0]), 2.5)
        self.assertEqual(rb.median([5.0]), 5.0)
        self.assertIsNone(rb.median([]))

    def test_mad_of_identical_samples_is_zero(self) -> None:
        self.assertEqual(rb.mad([2.0, 2.0, 2.0, 2.0, 2.0]), 0.0)
        self.assertEqual(rb.robust_cv([2.0] * 5), 0.0)

    def test_mad_resists_an_outlier_that_wrecks_the_stdev(self) -> None:
        # The whole reason MAD is used instead of stdev: one 100x sample from an
        # unrelated process must not dominate the dispersion. MAD is not
        # literally unchanged by the outlier -- replacing a sample shifts which
        # deviations sit in the middle -- but it stays on the same scale, while
        # the standard deviation moves by two orders of magnitude.
        import statistics as st
        noisy = [1.00, 1.01, 0.99, 1.00, 100.0]
        self.assertEqual(rb.median(noisy), 1.0)          # median unmoved
        self.assertLess(rb.robust_cv(noisy), 0.05)       # MAD stays tiny
        self.assertGreater(st.pstdev(noisy) / st.fmean(noisy), 1.0)  # stdev explodes

    def test_mad_scaling_matches_the_documented_constant(self) -> None:
        samples = [1.0, 1.0, 1.0, 2.0, 2.0]     # median 1, MAD 0 -> cv 0
        self.assertEqual(rb.robust_cv(samples), 0.0)
        samples = [0.5, 1.0, 1.0, 1.0, 1.5]     # median 1, MAD 0
        self.assertEqual(rb.mad(samples), 0.0)
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]     # median 3, MAD 1
        self.assertEqual(rb.mad(samples), 1.0)
        self.assertAlmostEqual(rb.robust_cv(samples), 1.4826 / 3.0)

    def test_single_sample_has_zero_dispersion(self) -> None:
        # Honest, but the CLI warns that fewer than 3 repetitions makes this
        # uninformative -- zero MAD here means "no evidence", not "stable".
        self.assertEqual(rb.mad([1.0]), 0.0)
        self.assertEqual(rb.robust_cv([1.0]), 0.0)

    def test_zero_median_yields_no_cv_instead_of_dividing_by_zero(self) -> None:
        self.assertEqual(rb.median([0.0, 0.0, 0.0]), 0.0)
        self.assertEqual(rb.mad([0.0, 0.0, 0.0]), 0.0)
        self.assertIsNone(rb.robust_cv([0.0, 0.0, 0.0]))
        self.assertIsNone(rb.robust_cv([0.0, 0.0, 1.0]))   # median still 0

    def test_negative_median_yields_no_cv(self) -> None:
        # Should never happen for a timing, but a ratio has no meaning here.
        self.assertIsNone(rb.robust_cv([-2.0, -1.0, -1.0]))

    def test_empty_input(self) -> None:
        self.assertIsNone(rb.mad([]))
        self.assertIsNone(rb.robust_cv([]))

    def test_none_samples_are_dropped(self) -> None:
        self.assertEqual(rb.median([1.0, None, 3.0]), 2.0)  # type: ignore[list-item]
        self.assertIsNone(rb.median([None, None]))  # type: ignore[list-item]


class EnvironmentTests(unittest.TestCase):
    def make_sysroot(self, *, governors: list[str] | None = None,
                     no_turbo: str | None = None, boost: str | None = None,
                     smt: str | None = None, aslr: str | None = None,
                     loadavg: str | None = None,
                     siblings: dict[int, str] | None = None,
                     isolated: str | None = None) -> Path:
        d = Path(self._dir.name)
        cpu = d / "sys/devices/system/cpu"
        proc = d / "proc"
        for p in (cpu, proc):
            p.mkdir(parents=True, exist_ok=True)
        for i, g in enumerate(governors or []):
            gd = cpu / f"cpu{i}/cpufreq"
            gd.mkdir(parents=True, exist_ok=True)
            (gd / "scaling_governor").write_text(g + "\n")
        if no_turbo is not None:
            (cpu / "intel_pstate").mkdir(parents=True, exist_ok=True)
            (cpu / "intel_pstate/no_turbo").write_text(no_turbo)
        if boost is not None:
            (cpu / "cpufreq").mkdir(parents=True, exist_ok=True)
            (cpu / "cpufreq/boost").write_text(boost)
        if smt is not None:
            (cpu / "smt").mkdir(parents=True, exist_ok=True)
            (cpu / "smt/control").write_text(smt)
        if isolated is not None:
            (cpu / "isolated").write_text(isolated)
        if aslr is not None:
            (proc / "sys/kernel").mkdir(parents=True, exist_ok=True)
            (proc / "sys/kernel/randomize_va_space").write_text(aslr)
        if loadavg is not None:
            (proc / "loadavg").write_text(loadavg)
        for n, s in (siblings or {}).items():
            td = cpu / f"cpu{n}/topology"
            td.mkdir(parents=True, exist_ok=True)
            (td / "thread_siblings_list").write_text(s + "\n")
        (proc / "cpuinfo").write_text(
            "processor\t: 0\nmodel name\t: Test CPU 9000\nflags\t\t: fpu\n")
        return d

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_reads_governors_model_and_load(self) -> None:
        root = self.make_sysroot(governors=["performance", "performance"],
                                 loadavg="0.12 0.30 0.44 1/500 1234")
        env = rb.capture_environment(root)
        self.assertEqual(env["governors"], ["performance"])   # deduplicated
        self.assertEqual(env["cpu_model"], "Test CPU 9000")
        self.assertAlmostEqual(env["loadavg_1min"], 0.12)

    def test_mixed_governors_are_all_reported(self) -> None:
        root = self.make_sysroot(governors=["performance", "powersave"])
        self.assertEqual(sorted(rb.capture_environment(root)["governors"]),
                         ["performance", "powersave"])

    def test_turbo_from_intel_pstate_no_turbo(self) -> None:
        # no_turbo == 0 means turbo is ON.
        self.assertIs(rb.capture_environment(self.make_sysroot(no_turbo="0"))
                      ["turbo_enabled"], True)
        self.assertIs(rb.capture_environment(self.make_sysroot(no_turbo="1"))
                      ["turbo_enabled"], False)

    def test_turbo_from_acpi_boost_has_the_opposite_polarity(self) -> None:
        self.assertIs(rb.capture_environment(self.make_sysroot(boost="1"))
                      ["turbo_enabled"], True)
        self.assertIs(rb.capture_environment(self.make_sysroot(boost="0"))
                      ["turbo_enabled"], False)

    def test_unknown_state_is_none_not_a_guess(self) -> None:
        env = rb.capture_environment(Path(self._dir.name))
        for key in ("governors", "turbo_enabled", "smt_control",
                    "isolated_cpus", "aslr", "loadavg_1min"):
            self.assertIsNone(env[key], key)

    def test_thread_siblings_only_read_for_the_pinned_cpu(self) -> None:
        root = self.make_sysroot(siblings={2: "2-3", 4: "4-5"})
        self.assertEqual(
            rb.capture_environment(root, cpu=2)["thread_siblings_of_pinned_cpu"],
            "2-3")
        self.assertIsNone(
            rb.capture_environment(root)["thread_siblings_of_pinned_cpu"])

    def test_malformed_loadavg_does_not_raise(self) -> None:
        root = self.make_sysroot(loadavg="not a load average")
        self.assertIsNone(rb.capture_environment(root)["loadavg_1min"])
        root = self.make_sysroot(loadavg="")
        self.assertIsNone(rb.capture_environment(root)["loadavg_1min"])


class WarningTests(unittest.TestCase):
    CLEAN = {"governors": ["performance"], "turbo_enabled": False,
             "smt_control": "off", "aslr": "0", "loadavg_1min": 0.01,
             "taskset_available": True, "thread_siblings_of_pinned_cpu": "2"}

    def warn(self, **overrides: object) -> list[str]:
        env = dict(self.CLEAN)
        env.update(overrides)
        cpu = overrides.pop("_cpu", 2)
        return rb.stability_warnings(env, cpu)  # type: ignore[arg-type]

    def test_a_clean_machine_produces_no_warnings(self) -> None:
        self.assertEqual(self.warn(), [])

    def test_non_performance_governor(self) -> None:
        out = self.warn(governors=["powersave"])
        self.assertEqual(len(out), 1)
        self.assertIn("governor", out[0])

    def test_turbo_enabled(self) -> None:
        self.assertTrue(any("turbo" in w for w in self.warn(turbo_enabled=True)))

    def test_unknown_turbo_is_not_warned_about(self) -> None:
        # None means "could not read it", which must not be reported as a fault.
        self.assertEqual(self.warn(turbo_enabled=None), [])

    def test_high_load(self) -> None:
        self.assertTrue(any("load average" in w for w in self.warn(loadavg_1min=3.0)))
        self.assertEqual(self.warn(loadavg_1min=0.4), [])

    def test_no_pinning(self) -> None:
        out = rb.stability_warnings(dict(self.CLEAN), None)
        self.assertTrue(any("--cpu" in w for w in out))

    def test_pinning_requested_without_taskset(self) -> None:
        out = self.warn(taskset_available=False)
        self.assertTrue(any("taskset" in w for w in out))

    def test_smt_on_with_pinning(self) -> None:
        out = self.warn(smt_control="on")
        self.assertTrue(any("SMT" in w for w in out))

    def test_aslr_enabled(self) -> None:
        self.assertTrue(any("ASLR" in w for w in self.warn(aslr="2")))

    def test_unknown_aslr_is_not_warned_about(self) -> None:
        self.assertEqual(self.warn(aslr=None), [])


def make_run(samples: dict[str, list[float]], *, status: str = "Optimal",
             proof: str | None = "ProvedOptimalFP",
             solver: str = "sor:simplex") -> "compare_runs.Run":
    records = []
    for instance, secs in samples.items():
        for rep, s in enumerate(secs):
            records.append({"record": "run", "solver": solver,
                            "instance": instance, "status": status,
                            "proof": proof, "objective": 1.0, "seconds": s,
                            "wall_s": s, "iterations": 10, "repetition": rep})
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        return compare_runs.load_run(p)


class SummarizeTests(unittest.TestCase):
    def test_stable_instance(self) -> None:
        run = make_run({"a.mps": [1.00, 1.01, 0.99, 1.00, 1.02]})
        st = rb.summarize(run, "sor:simplex", threshold=0.05, floor_s=0.005)[0]
        self.assertEqual(st.n, 5)
        self.assertAlmostEqual(st.median_s, 1.00)
        self.assertEqual(st.min_s, 0.99)
        self.assertEqual(st.max_s, 1.02)
        self.assertTrue(st.stable)
        self.assertTrue(st.certified)

    def test_unstable_instance(self) -> None:
        run = make_run({"a.mps": [1.0, 1.4, 0.7, 1.5, 0.6]})
        st = rb.summarize(run, "sor:simplex", threshold=0.05, floor_s=0.005)[0]
        self.assertFalse(st.stable)
        self.assertGreater(st.robust_cv, 0.05)

    def test_below_the_noise_floor_is_not_judged(self) -> None:
        # Two milliseconds with 50% spread is process startup, not the solver.
        run = make_run({"a.mps": [0.002, 0.003, 0.002, 0.004, 0.002]})
        st = rb.summarize(run, "sor:simplex", threshold=0.05, floor_s=0.005)[0]
        self.assertIsNone(st.stable)
        self.assertIsNotNone(st.robust_cv)

    def test_uncertified_result_is_flagged(self) -> None:
        run = make_run({"a.mps": [1.0] * 5}, status="Optimal", proof="BoundOnly")
        st = rb.summarize(run, "sor:simplex", threshold=0.05, floor_s=0.005)[0]
        self.assertFalse(st.certified)

    def test_timeout_keeps_its_samples_but_is_uncertified(self) -> None:
        run = make_run({"a.mps": [30.0] * 3}, status="timeout", proof=None)
        st = rb.summarize(run, "sor:simplex", threshold=0.05, floor_s=0.005)[0]
        self.assertFalse(st.certified)
        self.assertEqual(st.status, "timeout")
        self.assertEqual(st.n, 3)

    def test_instances_are_sorted(self) -> None:
        run = make_run({"c.mps": [1.0], "a.mps": [1.0], "b.mps": [1.0]})
        names = [s.instance for s in
                 rb.summarize(run, "sor:simplex", 0.05, 0.005)]
        self.assertEqual(names, ["a.mps", "b.mps", "c.mps"])


class CliTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = rb.main(argv)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_prints_the_command_and_runs_nothing(self) -> None:
        code, out, _ = self._run(["m.mps", "--dry-run", "--cpu", "1",
                                  "--repetitions", "7"])
        self.assertEqual(code, 0)
        self.assertIn("COMMAND", out)
        self.assertIn("compare.py", out)
        self.assertIn("--repetitions 7", out)
        self.assertIn("--cpu 1", out)
        self.assertIn("ENVIRONMENT", out)

    def test_dry_run_reports_machine_warnings_without_failing(self) -> None:
        # --dry-run must work even on a machine that would refuse to measure.
        code, out, _ = self._run(["m.mps", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("REPRODUCIBILITY", out)

    def test_invalid_repetitions(self) -> None:
        for value in ("0", "-1"):
            with self.assertRaises(SystemExit) as ctx:
                self._run(["m.mps", "--repetitions", value, "--dry-run"])
            self.assertNotEqual(ctx.exception.code, 0)

    def test_negative_warmups(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(["m.mps", "--warmups", "-1", "--dry-run"])

    def test_negative_cpu(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(["m.mps", "--cpu", "-1", "--dry-run"])

    def test_nonpositive_time_limit(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(["m.mps", "--time-limit", "0", "--dry-run"])

    def test_negative_threshold_and_floor(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(["m.mps", "--stable-threshold", "-0.1", "--dry-run"])
        with self.assertRaises(SystemExit):
            self._run(["m.mps", "--noise-floor-ms", "-1", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
