#!/usr/bin/env python3
"""Tests for scripts/claim_run.py -- the claim protocol's refusal logic.

Register with sor_add_python_test(test_claim_run) in tests/CMakeLists.txt.

Every assertion here corresponds to a way a wrong number actually reached a
report during this project: an unpinned single repetition, a host whose kernel
had dropped the TSC clocksource, a manifest that no longer matched the corpus,
and a suite with no reference answers being described as ready.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "compare.py").exists())


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


claim = _load("sor_claim", "scripts/claim_run.py")


HIGHS_CONFIG = {
    "output_flag": False, "log_to_console": False,
    "threads": 1, "parallel": "off", "time_limit": 60.0,
    "primal_feasibility_tolerance": 1e-7,
    "dual_feasibility_tolerance": 1e-7,
    "random_seed": 0, "solver": "choose", "presolve": "choose",
    "run_crossover": "on", "solve_relaxation": False,
    "small_matrix_value": None,
}


def args(**kw):
    base = dict(cpu=None, tol=1e-7, time_limit=60.0, warmups=1, repetitions=5,
                seed=0, allow_dirty=True, python=sys.executable,
                highs_source=None, skip_checker=False)
    base.update(kw)
    return argparse.Namespace(**base)


class ProtocolRefusalTests(unittest.TestCase):
    def refusals(self, **kw):
        exe = Path(sys.executable)
        _, r = claim.preflight(args(**kw), exe)
        return r

    def test_one_repetition_is_refused(self):
        # The single-repetition sweep is how a G2 wrong by 2x was published.
        self.assertTrue(any("1 warm-up + 5 repetitions" in x
                            for x in self.refusals(repetitions=1)))

    def test_zero_warmup_is_refused(self):
        self.assertTrue(any("1 warm-up + 5 repetitions" in x
                            for x in self.refusals(warmups=0)))

    def test_full_protocol_does_not_trip_the_repetition_rule(self):
        self.assertFalse(any("repetitions" in x for x in self.refusals()))

    def test_dirty_tree_is_refused_by_default(self):
        exe = Path(sys.executable)
        _, r = claim.preflight(args(allow_dirty=False), exe)
        state = claim.git_state()
        if state["dirty"]:
            self.assertTrue(any("dirty" in x for x in r))

    def test_environment_capture_has_the_fields_a_claim_needs(self):
        env, _ = claim.preflight(args(), Path(sys.executable))
        for key in ("clocksource", "governor", "mem_available_gib", "loadavg",
                    "kernel", "executable_sha256", "git", "compiler",
                    "cmake_flags", "temperature_c", "seed", "tolerance",
                    "repetitions", "warmups", "smt_active", "pswpin"):
            self.assertIn(key, env, key)
        self.assertIn("commit", env["git"])
        self.assertIn("dirty", env["git"])


class ManifestVerificationTests(unittest.TestCase):
    def write_manifest(self, d, entries, **over):
        doc = {"manifest_version": 1, "suite": "t", "tier": "T",
               "model_count": len(entries),
               "scoring_eligible_count": sum(1 for e in entries
                                             if e.get("scoring_eligible")),
               "models": entries}
        doc.update(over)
        p = Path(d) / "m.json"
        p.write_text(json.dumps(doc))
        return p

    def model(self, d, name, body=b"NAME T\nROWS\n N OBJ\nENDATA\n",
              eligible=True):
        p = Path(d) / name
        p.write_text(body.decode())
        rel = p.relative_to(ROOT) if str(p).startswith(str(ROOT)) else p
        return {"name": name, "path": str(rel), "sha256": claim.sha256_of(p),
                "scoring_eligible": eligible}, p

    def test_checksum_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            entry, path = self.model(d, "a.mps")
            entry["sha256"] = "0" * 64
            man = self.write_manifest(d, [entry])
            _, problems, _ = claim.verify_manifest(man)
            self.assertTrue(any("checksum changed" in p for p in problems))

    def test_missing_model_is_reported(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            entry, path = self.model(d, "a.mps")
            path.unlink()
            man = self.write_manifest(d, [entry])
            _, problems, _ = claim.verify_manifest(man)
            self.assertTrue(any("missing" in p for p in problems))

    def test_zero_scoring_eligible_blocks_a_claim(self):
        # An import-only manifest (Tier-2A today) must not be describable as
        # claim-ready: without reference answers nothing can be checked.
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            entry, _ = self.model(d, "a.mps", eligible=False)
            man = self.write_manifest(d, [entry])
            _, problems, _ = claim.verify_manifest(man)
            self.assertTrue(any("scoring-eligible" in p for p in problems))

    def test_model_count_disagreement_is_reported(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            entry, _ = self.model(d, "a.mps")
            man = self.write_manifest(d, [entry], model_count=7)
            _, problems, _ = claim.verify_manifest(man)
            self.assertTrue(any("model_count" in p for p in problems))

    def test_intact_manifest_verifies_clean(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            e1, _ = self.model(d, "a.mps")
            e2, _ = self.model(d, "b.mps", body=b"NAME U\nROWS\n N OBJ\nENDATA\n")
            man = self.write_manifest(d, [e1, e2])
            models, problems, doc = claim.verify_manifest(man)
            self.assertEqual(problems, [])
            self.assertEqual(len(models), 2)

    def test_committed_netlib_manifest_still_matches_the_corpus(self):
        man = ROOT / "benchmarks" / "manifests" / "netlib-v1.json"
        if not man.exists():
            self.skipTest("netlib manifest not frozen")
        models, problems, doc = claim.verify_manifest(man)
        self.assertEqual(problems, [], problems[:3])
        self.assertEqual(len(models), 93)




class FailClosedProtocolTests(unittest.TestCase):
    """Item 16: every fail-open path, with a test that would have caught it."""

    def test_output_is_not_destroyed_before_a_refusal(self):
        # An earlier version unlinked the output file BEFORE preflight, so a
        # refusal destroyed the previous run's evidence.
        src = (ROOT / "scripts" / "claim_run.py").read_text()
        i = src.index("def main()")
        body = src[i:]
        unlink_at = body.find(".unlink()")
        refuse_at = body.find("CLAIM RUN REFUSED")
        if unlink_at != -1:
            self.assertGreater(unlink_at, refuse_at,
                               "output is unlinked before the refusal decision")
        self.assertIn("refusing to overwrite evidence", body)

    def test_missing_source_highs_is_a_refusal(self):
        r = self.refusals(highs_source=None)
        self.assertTrue(any("source-built HiGHS" in x for x in r))

    def test_build_must_be_release_o3_ndebug_native(self):
        problems = claim.verify_build(Path(sys.executable), {"cmake_flags": {}})
        self.assertTrue(problems)
        good = {"cmake_flags": {"CMAKE_BUILD_TYPE": "Release",
                                "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
                                "SOR_NATIVE_ARCH": "ON"}}
        self.assertEqual(claim.verify_build(Path(sys.executable), good), [])
        for bad, needle in (
                ({"CMAKE_BUILD_TYPE": "Debug",
                  "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
                  "SOR_NATIVE_ARCH": "ON"}, "Release"),
                ({"CMAKE_BUILD_TYPE": "Release",
                  "CMAKE_CXX_FLAGS_RELEASE": "-O2 -DNDEBUG",
                  "SOR_NATIVE_ARCH": "ON"}, "-O3"),
                ({"CMAKE_BUILD_TYPE": "Release",
                  "CMAKE_CXX_FLAGS_RELEASE": "-O3",
                  "SOR_NATIVE_ARCH": "ON"}, "NDEBUG"),
                ({"CMAKE_BUILD_TYPE": "Release",
                  "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
                  "SOR_NATIVE_ARCH": "OFF"}, "march=native")):
            got = claim.verify_build(Path(sys.executable), {"cmake_flags": bad})
            self.assertTrue(any(needle in g for g in got), (bad, got))

    def test_competing_process_detection_is_not_only_loadavg(self):
        procs = claim.competing_processes(threshold=0.0)
        self.assertIsInstance(procs, list)
        if procs:
            self.assertIn("pcpu", procs[0])
            self.assertIn("comm", procs[0])

    def test_snapshot_delta_reports_swap_and_throttle_changes(self):
        before = {"pswpin": 10, "pswpout": 20, "throttle": {"c": 1},
                  "temperature_c": 50.0}
        after = {"pswpin": 15, "pswpout": 20, "throttle": {"c": 4},
                 "temperature_c": 70.0}
        d = claim.snapshot_delta(before, after)
        self.assertEqual(d["pswpin_delta"], 5)
        self.assertEqual(d["pswpout_delta"], 0)
        self.assertEqual(d["throttle_events"], {"c": 3})
        self.assertEqual(d["max_temperature_c"], 70.0)

    def test_throttle_counter_state_is_unknown_when_files_are_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(claim.throttle_counters(Path(d)))

    def test_throttle_counter_state_is_unknown_when_a_file_is_unreadable(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cpu0" / "thermal_throttle" / "core_throttle_count"
            path.parent.mkdir(parents=True)
            path.write_text("10\n")
            with mock.patch.object(Path, "read_text", side_effect=PermissionError):
                self.assertIsNone(claim.throttle_counters(Path(d)))

    def test_unchanged_valid_throttle_counters_are_known_zero_events(self):
        before = {"pswpin": 1, "pswpout": 2, "throttle": {"counter": 9}}
        after = {"pswpin": 1, "pswpout": 2, "throttle": {"counter": 9}}
        self.assertEqual(claim.snapshot_delta(before, after)["throttle_events"], {})

    def test_missing_or_changed_throttle_counter_set_is_unknown(self):
        before = {"pswpin": 1, "pswpout": 2, "throttle": {"a": 1}}
        after = {"pswpin": 1, "pswpout": 2, "throttle": {"b": 1}}
        self.assertIsNone(claim.snapshot_delta(before, after)["throttle_events"])

    def test_smt_sibling_above_five_percent_is_not_called_idle(self):
        with mock.patch.object(claim, "smt_siblings", return_value=[3]), \
                mock.patch.object(claim, "cpu_busy_percent", return_value=6.0):
            _, refusals = claim.preflight(args(cpu=2), Path(sys.executable))
        self.assertTrue(any("protocol maximum is 5.0%" in r for r in refusals))

    def valid_baseline(self, d: str, names=("a.mps",)) -> Path:
        root = Path(d)
        allow = root / "allow.json"
        allow.write_text(json.dumps({"_schema": {"version": 1}, "t": {}}))
        source_identity = {
            "kind": "sor-highs-source-api-runner", "runner_version": 1,
            "build_type": "Release", "optimization": "-O3",
            "ndebug": True, "native_arch": True,
            "compiler": "/usr/bin/c++", "highs_source_dirty": False,
            "highs_source_commit": "e" * 40, "highs_githash": "e" * 8,
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "models": [{"name": name, "scoring_eligible": True}
                       for name in names]}))
        preflight = {
            "record": "preflight", "claim_role": "preflight",
            "status": "PASS", "host_preflight_passed": True,
            "manifest": str(manifest),
            "manifest_sha256": claim.sha256_of(manifest),
            "dirty": False, "warmups": 1, "repetitions": 5,
            "affinity": 2, "taskset": "/usr/bin/taskset",
            "clocksource": "tsc", "governor": "performance",
            "mem_available_gib": 16.0, "competing_processes": [],
            "throttle_before": {"counter": 10},
            "tolerance": 1e-7, "executable_sha256": "a" * 64,
            "build_native_arch": "ON",
            "cmake_flags": {"CMAKE_BUILD_TYPE": "Release",
                            "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
                            "SOR_NATIVE_ARCH": "ON",
                            "CMAKE_CXX_COMPILER": "/usr/bin/c++"},
            "source_highs_sha256": "b" * 64,
            "source_highs_version": "1.15.1",
            "wheel_highs_version": "1.15.1",
            "source_highs": {
                "path": "/build/bin/sor_highs_source_runner",
                "sha256": "b" * 64, "version": "1.15.1",
                "identity": source_identity},
            "wheel_highs": {
                "version": "1.15.1", "distribution_version": "1.15.1",
                "module_sha256": "c" * 64, "core_sha256": "d" * 64},
            "candidate_solver": "sor:simplex",
            "allow_list_path": str(allow),
            "allow_list_sha256": claim.sha256_of(allow),
            "allow_list_schema_version": 1,
        }
        records = [preflight, {"record": "environment", "solvers": ["sor:simplex"]}]
        records += [{"record": "aggregate", "solver": "sor:simplex",
                     "instance": name, "median_s": 1.0,
                     "median_wall_s": 3.0} for name in names]
        for name in names:
            records += [
                {"record": "aggregate", "solver": "highs-source",
                 "instance": name, "status": "Optimal", "objective": 1.0,
                 "seconds": 1.0, "configuration": HIGHS_CONFIG,
                 "solver_version": "1.15.1", "build_identity": source_identity},
                {"record": "aggregate", "solver": "highs-wheel",
                 "instance": name, "status": "Optimal", "objective": 1.0,
                 "seconds": 1.0, "configuration": HIGHS_CONFIG,
                 "solver_version": "1.15.1"},
            ]
        records += [
            {"record": "postflight", "claim_role": "postflight",
             "status": "PASS", "session_problems": [],
             "host_after": {"mem_available_gib": 16.0,
                            "competing_processes": [],
                            "governor": "performance",
                            "throttle": {"counter": 10}},
             "delta": {"pswpin_delta": 0, "pswpout_delta": 0,
                       "throttle_events": {}}},
            {"record": "independent_check", "claim_role": "independent_check",
             "status": "PASS",
            "results": {name: {"checked": True,
                                 "checker_status": "VERIFIED",
                                 "residual_ok": True,
                                 "certificate_ok": True}
                         for name in names}},
            {"record": "gate", "claim_role": "gate", "status": "PASS",
             "candidate_solver": "sor:simplex"},
            {"record": "baseline_eligibility",
             "claim_role": "baseline_eligibility", "status": "PASS",
             "eligible": True},
            {"record": "gate_accept", "payload": {
                "baseline_version": "v-test", "reason": "test"}},
        ]
        p = root / "baseline.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        return p

    def test_validated_baseline_returns_separate_solver_and_wall_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.valid_baseline(d)
            data = claim.load_baselines(p, expected_instances={"a.mps"},
                                        manifest_sha256=json.loads(
                                            p.read_text().splitlines()[0])[
                                                "manifest_sha256"],
                                        tolerance=1e-7,
                                        source_highs_sha256="b" * 64,
                                        source_highs_version="1.15.1",
                                        wheel_highs_version="1.15.1")
            self.assertEqual(data.solver["a.mps"], 1.0)
            self.assertEqual(data.wall["a.mps"], 3.0)
            self.assertEqual(data.version, "v-test")

    def test_first_baseline_is_valid_without_an_older_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.valid_baseline(d)
            records = [json.loads(line) for line in p.read_text().splitlines()]
            records = [r for r in records if r.get("record") != "gate_accept"]
            p.write_text("".join(json.dumps(r) + "\n" for r in records))
            _, problems = claim.protocol.validate_baseline(
                p, expected_instances={"a.mps"}, require_accept=False)
            self.assertEqual(problems, [])

    def test_public_performance_fail_can_still_be_baseline_eligible(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.valid_baseline(d)
            records = [json.loads(line) for line in p.read_text().splitlines()]
            for record in records:
                if record.get("record") == "gate":
                    record["status"] = "FAIL"
                    record["failures"] = ["shifted SGM ratio 1.2 > 0.95"]
            p.write_text("".join(json.dumps(r) + "\n" for r in records))
            data = claim.load_baselines(p, expected_instances={"a.mps"})
            self.assertEqual(data.solver["a.mps"], 1.0)

    def test_correctness_invalid_run_is_not_baseline_eligible(self):
        source = claim.compare.Result(
            "highs-source", "a.mps", "Optimal", 1.0, seconds=1.0)
        candidate = claim.compare.Result(
            "sor:simplex", "a.mps", "Optimal", 1.0,
            proof="ProvedOptimalFP", seconds=0.5)
        outcome = claim.evaluate_baseline_eligibility(
            {"a.mps": {"sor:simplex": candidate, "highs-source": source,
                       "highs-wheel": source}},
            "sor:simplex", {"a.mps"},
            {"a.mps": (False, "residual rejected")},
            lane_problems=[], protocol_problems=[])
        self.assertEqual(outcome["status"], "FAIL")
        self.assertFalse(outcome["eligible"])

    def test_missing_baseline_file_is_rejected(self):
        with self.assertRaises(claim.BaselineError):
            claim.load_baselines(Path("/nonexistent.jsonl"))

    def test_explicitly_voided_baseline_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.valid_baseline(d)
            p.write_text(json.dumps({"record": "metadata",
                                     "gate_accept_voided": {"why": "bad"}})
                         + "\n" + p.read_text())
            with self.assertRaisesRegex(claim.BaselineError, "voided"):
                claim.load_baselines(p)

    def test_baseline_coverage_numeric_and_build_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.valid_baseline(d, ("a.mps", "unexpected.mps"))
            with self.assertRaisesRegex(claim.BaselineError, "unexpected"):
                claim.load_baselines(p, expected_instances={"a.mps"})
            with self.assertRaisesRegex(claim.BaselineError, "source HiGHS build"):
                claim.load_baselines(p, source_highs_sha256="c" * 64)

    def test_absent_allow_list_suite_is_rejected(self):
        snapshot, problems = claim.load_allow("no-such-suite-xyz", set())
        self.assertIsNone(snapshot)
        self.assertTrue(any("no known suite" in p for p in problems))

    def test_empty_allow_entry_is_rejected_and_snapshot_is_not_exposed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "allow.json"
            path.write_text(json.dumps({
                "_schema": {"version": 1}, "suite": {"a.mps": {}}}))
            snapshot, problems = claim.load_allow("suite", {"a.mps"}, path)
            self.assertIsNone(snapshot)
            self.assertTrue(any("nonempty reason" in p for p in problems))

    def test_valid_allow_snapshot_is_hashed_and_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "allow.json"
            path.write_text(json.dumps({
                "_schema": {"version": 1},
                "suite": {"a.mps": {"reason": "tracked",
                                      "max_work_ratio": 2.5}}}))
            snapshot, problems = claim.load_allow("suite", {"a.mps"}, path)
            self.assertEqual(problems, [])
            self.assertEqual(snapshot.sha256, claim.sha256_of(path))
            self.assertEqual(snapshot.schema_version, 1)

    def refusals(self, **kw):
        return claim.preflight(args(**kw), Path(sys.executable))[1]


class LaneAndMeasuredSolutionTests(unittest.TestCase):
    def make_highs(self, root: Path, solve_status="Optimal",
                   objective=1.25, *, build_type="Release",
                   native=True) -> Path:
        exe = root / "sor_highs_source_runner"
        identity = {
            "kind": "sor-highs-source-api-runner", "runner_version": 1,
            "build_type": build_type, "optimization": "-O3",
            "ndebug": True, "native_arch": native,
            "compiler": "/usr/bin/c++", "highs_source_dirty": False,
            "highs_source_commit": "a" * 40, "highs_githash": "a" * 8,
        }
        identity_payload = {"schema": "sor-highs-source-runner-v1",
                            "kind": "identity", "highs_version": "1.15.1",
                            "build_identity": identity}
        solve_payload = {
            "schema": "sor-highs-source-runner-v1", "kind": "solve_result",
            "read_status": "kOk", "run_status": "kOk",
            "options_applied": True, "model_status": solve_status,
            "objective": objective, "solve_seconds": 0.25,
            "simplex_iterations": 4, "highs_version": "1.15.1",
            "configuration": dict(HIGHS_CONFIG, time_limit=2.0, random_seed=7),
            "build_identity": identity,
        }
        exe.write_text(
            "#!/usr/bin/env python3\nimport json,sys\n"
            f"identity={identity_payload!r}\nsolve={solve_payload!r}\n"
            "print(json.dumps(identity if sys.argv[1:] == ['--identity'] else solve))\n")
        exe.chmod(0o755)
        return exe

    def test_source_highs_lane_executes_the_binary_and_records_command(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            exe = self.make_highs(root)
            model = root / "a.mps"; model.write_text("NAME A\nENDATA\n")
            result = claim.compare.run_highs_source(
                model, 2.0, "highs-source", exe, tol=1e-7, seed=7)
            self.assertEqual(result.status, "Optimal")
            self.assertEqual(result.objective, 1.25)
            self.assertEqual(result.seconds, 0.25)
            self.assertEqual(Path(result.command[0]), exe)
            self.assertIn("--model", result.command)

    def test_wheel_highs_lane_executes_highspy_worker(self):
        python = ROOT / "benchmarks" / ".venv-baseline" / "bin" / "python"
        model = ROOT / "examples" / "testlp.mps"
        if not python.exists() or not model.exists():
            self.skipTest("benchmark wheel environment/model unavailable")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "wheel.jsonl"
            proc = claim.subprocess.run(
                [str(python), str(ROOT / "scripts" / "compare.py"), str(model),
                 "--solvers", "highs-wheel", "--time-limit", "5",
                 "--repetitions", "1", "--jsonl", str(out),
                 "--allow-unchecked"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            aggregate = [json.loads(line) for line in out.read_text().splitlines()
                         if json.loads(line).get("record") == "aggregate"]
            self.assertEqual(len(aggregate), 1)
            self.assertEqual(aggregate[0]["solver"], "highs-wheel")
            self.assertEqual(aggregate[0]["status"].lower(), "optimal")
            self.assertIsNone(
                claim.protocol.highs_configuration_problem(
                    aggregate[0]["configuration"]))
            info, problems = claim.wheel_highs_info(str(python))
            self.assertEqual(problems, [])
            self.assertTrue(info["version"])
            self.assertTrue(info["core_sha256"])

    def test_source_probe_rejects_arbitrary_nonexecutables_and_bad_builds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            arbitrary = root / "blob"; arbitrary.write_text("not highs")
            _, problems = claim.source_highs_info(arbitrary)
            self.assertTrue(any("not executable" in p for p in problems))
            _, problems = claim.source_highs_info(root)
            self.assertTrue(any("not a regular file" in p for p in problems))

            build = root / "build"; (build / "bin").mkdir(parents=True)
            exe = self.make_highs(build / "bin", build_type="Debug", native=False)
            info, problems = claim.source_highs_info(
                exe, {"cmake_flags": {"CMAKE_CXX_COMPILER": "/usr/bin/c++"}})
            self.assertEqual(info["version"], "1.15.1")
            self.assertTrue(info["sha256"])
            self.assertTrue(any("not a native executable" in p for p in problems))
            self.assertTrue(any("not Release" in p for p in problems))
            self.assertTrue(any("-march=native" in p for p in problems))

    def test_source_and_wheel_disagreement_is_rejected(self):
        source = claim.compare.Result("highs-source", "a.mps", "Optimal",
                                      1.0, seconds=1.0,
                                      configuration=HIGHS_CONFIG)
        wheel = claim.compare.Result("highs-wheel", "a.mps", "Optimal",
                                     2.0, seconds=1.0,
                                     configuration=HIGHS_CONFIG)
        problems = claim.highs_lanes_agree(
            {"a.mps": {"highs-source": source, "highs-wheel": wheel}},
            {"a.mps"}, 1e-7, 1e-7)
        self.assertTrue(any("objective mismatch" in p for p in problems))

    def test_source_and_wheel_version_mismatch_is_a_preflight_refusal(self):
        source_info = {"path": "/tmp/highs", "sha256": "a" * 64,
                       "version": "1.14.0"}
        wheel_info = {"version": "1.15.1"}
        with mock.patch.object(claim, "source_highs_info",
                               return_value=(source_info, [])), \
                mock.patch.object(claim, "wheel_highs_info",
                                  return_value=(wheel_info, [])):
            _, problems = claim.preflight(
                args(highs_source=Path("/tmp/highs")), Path(sys.executable))
        self.assertTrue(any("does not match wheel" in p for p in problems))

    def test_checker_consumes_exact_measured_solution_without_resolving(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            solution = root / "measured.sol"; solution.write_text("measured")
            checker = root / "sor_check"
            checker.write_text("#!/bin/sh\necho VERIFIED\n")
            checker.chmod(0o755)
            measured = claim.compare.Result(
                "sor:hpr", "a.mps", "Optimal", 1.0,
                proof="ProvedKKT", seconds=0.2,
                command=["sor_solve", "a.mps", "--engine", "hpr",
                         "--relax-integrality", "--small-matrix-value", "1e-9",
                         "--solution-out", str(solution)],
                solution_file=str(solution),
                solution_sha256=claim.sha256_of(solution))
            checked = claim.independent_check(checker, root / "a.mps",
                                              measured, 1e-7)
            self.assertTrue(checked["checked"])
            self.assertEqual(checked["checker_status"], "VERIFIED")
            self.assertEqual(checked["measured_command"], measured.command)
            self.assertIn("--relax-integrality", checked["checker_command"])
            self.assertEqual(
                checked["checker_command"][
                    checked["checker_command"].index("--small-matrix-value") + 1],
                "1e-9")

    def test_sweep_targets_exact_noisy_pair_and_propagates_failure(self):
        ns = types.SimpleNamespace(
            python=sys.executable, claim_solvers="sor:simplex,highs-source,highs-wheel",
            tol=1e-7, time_limit=1.0, warmups=1, repetitions=5, seed=0,
            highs_source=Path("/tmp/highs"), cpu=2, method="dual",
            pricing="dse", basis_update="ft", dual_cost_perturbation=1.0,
            max_iter=99, relax_integrality=True, small_matrix_value=1e-9,
            sor_arg=["--hpr-full"])
        completed = types.SimpleNamespace(returncode=7)
        with mock.patch.object(claim.subprocess, "run", return_value=completed) as run:
            rc = claim.sweep(ns, [Path("only.mps")], Path("sor_solve"),
                             Path("rerun.jsonl"), solvers="highs-wheel")
        self.assertEqual(rc, 7)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--solvers") + 1], "highs-wheel")
        self.assertIn("only.mps", cmd)
        self.assertNotIn("sor:simplex,highs-source,highs-wheel", cmd)

    def test_rerun_provenance_marks_original_not_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "claim.jsonl"
            records = [
                {"record": "environment"},
                {"record": "aggregate", "instance": "a.mps",
                 "solver": "sor:simplex", "seconds": 9.0},
                {"record": "aggregate", "instance": "a.mps",
                 "solver": "highs-wheel", "seconds": 1.0},
            ]
            p.write_text("".join(json.dumps(r) + "\n" for r in records))
            claim.mark_originals_superseded(p, {("a.mps", "sor:simplex")})
            changed = [json.loads(x) for x in p.read_text().splitlines()]
            ours, wheel = changed[1], changed[2]
            self.assertTrue(ours["claim_superseded"])
            self.assertFalse(ours["claim_selected"])
            self.assertNotIn("claim_superseded", wheel)

    def test_development_override_forces_incomplete(self):
        gate = claim.compare.ClaimGateResult(status="PASS")
        claim.apply_protocol_outcome(
            gate, ["development override --allow-dirty was used"])
        self.assertEqual(gate.status, "INCOMPLETE")
        self.assertFalse(gate.passed)

    def test_unavailable_sgm_formats_safely(self):
        self.assertEqual(claim.fmt_optional(None), "n/a")
        self.assertEqual(claim.fmt_optional(float("nan")), "n/a")


if __name__ == "__main__":
    unittest.main(verbosity=1)
