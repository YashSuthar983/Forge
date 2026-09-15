#!/usr/bin/env python3
"""Shared, fail-closed validation for claim artifacts and pinned baselines."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


PROTOCOL_VERSION = 2
HIGHS_CONFIGURATION_FIELDS = {
    "output_flag", "log_to_console", "threads", "parallel", "time_limit",
    "primal_feasibility_tolerance", "dual_feasibility_tolerance",
    "random_seed", "solver", "presolve", "run_crossover",
    "solve_relaxation", "small_matrix_value",
}


def _finite_positive(value) -> bool:
    # JSON booleans are Python ints, but `true` is not a benchmark time.
    return (type(value) in (int, float) and math.isfinite(value) and
            value > 0.0)


def _sha256_text(value) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))


def highs_configuration_problem(config) -> str | None:
    if not isinstance(config, dict):
        return "configuration is not an object"
    missing = sorted(HIGHS_CONFIGURATION_FIELDS - set(config))
    if missing:
        return f"configuration is missing {missing}"
    if config.get("output_flag") is not False or \
            config.get("log_to_console") is not False:
        return "logging configuration is unknown or inconsistent"
    if (config.get("threads") != 1 or config.get("parallel") != "off" or
            config.get("solver") != "choose" or
            config.get("presolve") != "choose" or
            config.get("run_crossover") != "on"):
        return "fixed solver configuration is invalid"
    if not _finite_positive(config.get("time_limit")):
        return "time limit is not finite and positive"
    for name in ("primal_feasibility_tolerance",
                 "dual_feasibility_tolerance"):
        if not _finite_positive(config.get(name)):
            return f"{name} is not finite and positive"
    if type(config.get("random_seed")) is not int:
        return "random seed is invalid"
    if type(config.get("solve_relaxation")) is not bool:
        return "relaxation configuration is invalid"
    small = config.get("small_matrix_value")
    if small is not None and not _finite_positive(small):
        return "small-matrix threshold is invalid"
    return None


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def read_records(path: Path) -> tuple[list[dict], list[str]]:
    problems: list[str] = []
    records: list[dict] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        return [], [f"cannot read baseline: {e}"]
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {line_no}: invalid JSON: {e}")
            continue
        if not isinstance(rec, dict):
            problems.append(f"line {line_no}: record is not an object")
            continue
        records.append(rec)
    if not records:
        problems.append("baseline has no records")
    return records, problems


def _role(rec: dict) -> str | None:
    kind = rec.get("record")
    if kind == "metadata":
        return str(rec.get("claim_role") or "metadata")
    if kind in {"preflight", "postflight", "independent_check", "gate",
                "performance", "baseline_eligibility", "rerun",
                "gate_accept"}:
        return str(kind)
    if kind == "environment" and rec.get("claim_role"):
        return str(rec.get("claim_role"))
    return None


def _first_value(records: list[dict], *keys: str):
    value = None
    for rec in records:
        if rec.get("record") in {"environment", "metadata", "preflight",
                                 "postflight", "independent_check", "gate",
                                 "performance", "baseline_eligibility",
                                 "rerun", "gate_accept"}:
            for key in keys:
                if rec.get(key) is not None:
                    value = rec[key]
    return value


def _manifest_instances(records: list[dict], problems: list[str]) -> tuple[set[str] | None, str | None]:
    manifest_text = _first_value(records, "manifest")
    wanted_sha = _first_value(records, "manifest_sha256")
    if not manifest_text:
        problems.append("baseline does not name its frozen manifest")
        return None, wanted_sha
    path = Path(str(manifest_text))
    if not path.is_absolute():
        # Claim artifacts normally record an absolute path; retain support for
        # repository-relative manifests when validation is run from the repo.
        path = Path.cwd() / path
    if not path.is_file():
        problems.append(f"recorded manifest {path} is unavailable")
        return None, wanted_sha
    got_sha = sha256_file(path)
    if not wanted_sha or got_sha != wanted_sha:
        problems.append(
            f"manifest checksum mismatch: recorded {wanted_sha!r}, current {got_sha!r}")
    try:
        doc = json.loads(path.read_text())
        names = {str(e["name"]) for e in doc["models"]
                 if e.get("scoring_eligible")}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
        problems.append(f"recorded manifest cannot define coverage: {e}")
        return None, wanted_sha
    if not names:
        problems.append("recorded manifest has no scoring-eligible instances")
    return names, wanted_sha


@dataclass
class BaselineData:
    solver: dict[str, float] = field(default_factory=dict)
    wall: dict[str, float] = field(default_factory=dict)
    solver_label: str | None = None
    sha256: str | None = None
    version: str | int | None = None
    metadata: dict = field(default_factory=dict)


def validate_baseline(
        path: Path,
        *,
        expected_instances: set[str] | None = None,
        manifest_sha256: str | None = None,
        tolerance: float | None = None,
        source_highs_sha256: str | None = None,
        source_highs_version: str | None = None,
        wheel_highs_version: str | None = None,
        require_accept: bool = True) -> tuple[BaselineData, list[str]]:
    """Validate all protocol state before exposing any timing dictionaries."""
    data = BaselineData(sha256=sha256_file(path))
    records, problems = read_records(path)
    if not records:
        return data, problems

    protocol_records = [r for r in records if _role(r) is not None]
    merged: dict = {}
    for rec in records:
        if rec.get("record") in {"environment", "metadata", "preflight",
                                 "postflight", "independent_check", "gate",
                                 "performance", "baseline_eligibility",
                                 "rerun", "gate_accept"}:
            merged.update({k: v for k, v in rec.items() if v is not None})
    data.metadata = merged

    voided = [r for r in records if r.get("gate_accept_voided")]
    if voided:
        problems.append("baseline acceptance was explicitly voided (gate_accept_voided)")

    accepts: list[dict] = []
    for rec in records:
        if isinstance(rec.get("gate_accept"), dict):
            accepts.append(rec["gate_accept"])
        elif rec.get("record") == "gate_accept":
            accepts.append(rec.get("payload") if isinstance(rec.get("payload"), dict)
                           else rec)
    if require_accept and len(accepts) != 1:
        problems.append(f"baseline needs exactly one gate_accept record; found {len(accepts)}")
    if accepts:
        data.version = accepts[-1].get("baseline_version")
        if data.version is None:
            problems.append("gate_accept records no baseline_version")

    preflights = [r for r in records if _role(r) == "preflight"]
    postflights = [r for r in records if _role(r) == "postflight"]
    checks = [r for r in records if _role(r) == "independent_check"]
    gates = [r for r in records if _role(r) == "gate"]
    eligibilities = [r for r in records if _role(r) == "baseline_eligibility"]
    if len(preflights) != 1:
        problems.append(f"baseline needs exactly one preflight record; found {len(preflights)}")
    if len(postflights) != 1:
        problems.append(f"baseline needs exactly one postflight record; found {len(postflights)}")
    if len(checks) != 1:
        problems.append(f"baseline needs exactly one independent_check record; found {len(checks)}")
    if len(eligibilities) != 1:
        problems.append(
            "baseline needs exactly one baseline_eligibility record; "
            f"found {len(eligibilities)}")

    for rec in protocol_records:
        role = _role(rec)
        status = rec.get("status")
        if role in {"preflight", "postflight", "independent_check",
                    "baseline_eligibility", "rerun"}:
            if status != "PASS":
                problems.append(f"{role} protocol state is {status!r}, not 'PASS'")
    # A baseline may be slower than HiGHS.  `gate` and `performance` are
    # intentionally not required to PASS: performance is not evidence
    # integrity and cannot be a prerequisite for establishing the first SOR
    # baseline.
    if eligibilities and eligibilities[0].get("eligible") is not True:
        problems.append("baseline_eligibility does not explicitly mark the run eligible")

    if _first_value(records, "host_preflight_passed") is not True:
        problems.append("host preflight did not pass")
    if _first_value(records, "dirty") is not False:
        problems.append("baseline tree is dirty or dirty state is unknown")
    if _first_value(records, "warmups") != 1:
        problems.append(f"baseline needs exactly 1 warm-up; got {_first_value(records, 'warmups')!r}")
    if _first_value(records, "repetitions") != 5:
        problems.append(
            f"baseline needs exactly 5 repetitions; got {_first_value(records, 'repetitions')!r}")
    affinity = _first_value(records, "affinity", "cpu_affinity")
    if type(affinity) is not int or affinity < 0:
        problems.append("baseline CPU affinity is missing or invalid")
    if not _first_value(records, "taskset"):
        problems.append("baseline taskset availability is unknown")
    if _first_value(records, "clocksource") != "tsc":
        problems.append("baseline clocksource is unknown or not 'tsc'")
    if _first_value(records, "governor") != "performance":
        problems.append("baseline governor is unknown or not 'performance'")
    memory = _first_value(records, "mem_available_gib")
    if (type(memory) not in (int, float) or not math.isfinite(memory) or
            memory < 12.0):
        problems.append("baseline memory state is unknown or below 12 GiB")
    processes = _first_value(records, "competing_processes")
    if not isinstance(processes, list):
        problems.append("baseline process state is unknown")
    elif processes:
        problems.append("baseline preflight recorded competing processes")
    throttle_before = _first_value(records, "throttle_before")
    if not isinstance(throttle_before, dict) or not throttle_before:
        problems.append("baseline preflight throttle state is unknown")

    cmake = _first_value(records, "cmake_flags")
    native = _first_value(records, "build_native_arch", "native_arch")
    if not isinstance(cmake, dict):
        problems.append("baseline build configuration is unknown")
    else:
        if str(cmake.get("CMAKE_BUILD_TYPE", "")).lower() != "release":
            problems.append("baseline build is not Release")
        flags = str(cmake.get("CMAKE_CXX_FLAGS_RELEASE") or "")
        if "-O3" not in flags or "NDEBUG" not in flags:
            problems.append("baseline build does not record -O3 and NDEBUG")
    if str(native).upper() != "ON":
        problems.append("baseline build does not record -march=native")
    if not _first_value(records, "executable_sha256"):
        problems.append("baseline records no candidate executable SHA-256")

    recorded_manifest = _first_value(records, "manifest_sha256")
    if manifest_sha256 is not None and recorded_manifest != manifest_sha256:
        problems.append("baseline belongs to a different manifest")
    recorded_tol = _first_value(records, "tolerance", "tol")
    if tolerance is not None and (not isinstance(recorded_tol, (int, float)) or
                                  not math.isclose(float(recorded_tol), tolerance,
                                                   rel_tol=0.0, abs_tol=0.0)):
        problems.append(
            f"baseline tolerance {recorded_tol!r} does not match {tolerance!r}")

    for key, wanted, label in (
            ("source_highs_sha256", source_highs_sha256, "source HiGHS build"),
            ("source_highs_version", source_highs_version, "source HiGHS version"),
            ("wheel_highs_version", wheel_highs_version, "wheel HiGHS version")):
        got = _first_value(records, key)
        if got is None:
            problems.append(f"baseline records no {label}")
        elif wanted is not None and got != wanted:
            problems.append(f"baseline {label} {got!r} does not match {wanted!r}")
    sv = _first_value(records, "source_highs_version")
    wv = _first_value(records, "wheel_highs_version")
    if sv is not None and wv is not None and sv != wv:
        problems.append("baseline source and wheel HiGHS versions disagree")

    source_info = _first_value(records, "source_highs")
    if not isinstance(source_info, dict):
        problems.append("baseline source HiGHS configuration is unknown")
    else:
        if (source_info.get("sha256") !=
                _first_value(records, "source_highs_sha256")):
            problems.append("baseline source HiGHS hashes are inconsistent")
        if source_info.get("version") != sv:
            problems.append("baseline source HiGHS versions are inconsistent")
        identity = source_info.get("identity")
        if not isinstance(identity, dict) or identity.get("kind") != \
                "sor-highs-source-api-runner" or identity.get("runner_version") != 1:
            problems.append("baseline source runner identity is unknown")
        else:
            if (identity.get("build_type") != "Release" or
                    identity.get("optimization") != "-O3" or
                    identity.get("ndebug") is not True or
                    identity.get("native_arch") is not True):
                problems.append("baseline source runner build flags are invalid")
            if identity.get("highs_source_dirty") is not False:
                problems.append("baseline source HiGHS checkout state is not clean")
            commit = identity.get("highs_source_commit")
            githash = identity.get("highs_githash")
            if (not isinstance(commit, str) or
                    not re.fullmatch(r"[0-9a-fA-F]{40}", commit) or
                    not isinstance(githash, str) or not commit.startswith(githash)):
                problems.append("baseline source HiGHS pinned identity is invalid")
            sor_compiler = cmake.get("CMAKE_CXX_COMPILER") \
                if isinstance(cmake, dict) else None
            source_compiler = identity.get("compiler")
            if not sor_compiler or not source_compiler or \
                    Path(str(sor_compiler)).resolve() != \
                    Path(str(source_compiler)).resolve():
                problems.append("baseline source/SOR compiler builds do not match")

    wheel_info = _first_value(records, "wheel_highs")
    if not isinstance(wheel_info, dict):
        problems.append("baseline wheel HiGHS configuration is unknown")
    else:
        if wheel_info.get("version") != wv:
            problems.append("baseline wheel HiGHS versions are inconsistent")
        if (wheel_info.get("distribution_version") is None or
                str(wheel_info.get("distribution_version")).lstrip("v") !=
                str(wv).lstrip("v")):
            problems.append("baseline wheel distribution/runtime versions disagree")
        for key in ("module_sha256", "core_sha256"):
            if not _sha256_text(wheel_info.get(key)):
                problems.append(f"baseline wheel {key} is missing or invalid")

    if postflights:
        post = postflights[0]
        if post.get("session_problems"):
            problems.append("baseline postflight recorded session problems")
        delta = post.get("delta")
        if not isinstance(delta, dict):
            problems.append("baseline postflight delta is missing")
        else:
            for key in ("pswpin_delta", "pswpout_delta"):
                if delta.get(key) is None:
                    problems.append(f"baseline postflight {key} is unknown")
                elif delta.get(key) != 0:
                    problems.append(f"baseline postflight {key} is nonzero")
            if delta.get("throttle_events") is None:
                problems.append("baseline postflight throttle state is unknown")
            elif delta.get("throttle_events"):
                problems.append("baseline postflight recorded throttling")
        host_after = post.get("host_after")
        if not isinstance(host_after, dict):
            problems.append("baseline postflight host state is missing")
        else:
            post_mem = host_after.get("mem_available_gib")
            if (type(post_mem) not in (int, float) or
                    not math.isfinite(post_mem) or post_mem < 12.0):
                problems.append(
                    "baseline postflight memory state is unknown or below 12 GiB")
            if not isinstance(host_after.get("competing_processes"), list):
                problems.append("baseline postflight process state is unknown")
            if host_after.get("governor") != "performance":
                problems.append(
                    "baseline postflight governor is unknown or not 'performance'")
            if not isinstance(host_after.get("throttle"), dict) or \
                    not host_after.get("throttle"):
                problems.append("baseline postflight throttle state is unknown")
    if checks:
        results = checks[0].get("results")
        if not isinstance(results, dict) or not results:
            problems.append("baseline independent checker results are missing")
        elif (any(not isinstance(v, dict) for v in results.values()) or
              any(not v.get("checked") or v.get("checker_status") != "VERIFIED" or
                  v.get("residual_ok") is not True or
                  v.get("certificate_ok") is not True
                  for v in results.values() if isinstance(v, dict))):
            problems.append("baseline contains unchecked or rejected measured solutions")
    if _first_value(records, "development_overrides"):
        problems.append("baseline was produced with development overrides")

    allow_path = _first_value(records, "allow_list_path")
    allow_sha = _first_value(records, "allow_list_sha256")
    allow_version = _first_value(records, "allow_list_schema_version")
    if not allow_path or not _sha256_text(allow_sha) or allow_version != 1:
        problems.append("baseline allow-list provenance is missing or invalid")
    else:
        allow_file = Path(str(allow_path))
        if sha256_file(allow_file) != allow_sha:
            problems.append("baseline allow-list snapshot checksum does not match")

    inferred_expected, _ = _manifest_instances(records, problems)
    expected = expected_instances if expected_instances is not None else inferred_expected
    if expected_instances is not None and inferred_expected is not None and inferred_expected != expected_instances:
        problems.append("baseline manifest scoring set differs from the requested suite")
    if checks and expected is not None and isinstance(checks[0].get("results"), dict):
        checked_instances = set(checks[0]["results"])
        if checked_instances != expected:
            problems.append(
                "baseline checker coverage differs from the scoring-eligible suite")

    active = [r for r in records if r.get("record") == "aggregate"
              and not r.get("claim_superseded") and r.get("claim_selected") is not False]
    if expected is not None:
        for lane in ("highs-source", "highs-wheel"):
            lane_rows = [r for r in active if r.get("solver") == lane]
            grouped: dict[str, list[dict]] = {}
            for rec in lane_rows:
                grouped.setdefault(str(rec.get("instance")), []).append(rec)
            if set(grouped) != expected:
                problems.append(f"baseline {lane} coverage differs from the declared suite")
            for inst, rows in grouped.items():
                if len(rows) != 1:
                    problems.append(f"baseline has {len(rows)} selected {lane} rows for {inst}")
                    continue
                rec = rows[0]
                config_problem = highs_configuration_problem(
                    rec.get("configuration"))
                if (str(rec.get("status", "")).lower() != "optimal" or
                        rec.get("error") or
                        config_problem is not None or
                        not isinstance(rec.get("objective"), (int, float)) or
                        not math.isfinite(rec["objective"]) or
                        not isinstance(rec.get("seconds"), (int, float)) or
                        not math.isfinite(rec["seconds"]) or rec["seconds"] <= 0):
                    detail = f" ({config_problem})" if config_problem else ""
                    problems.append(
                        f"baseline {lane} row for {inst} is invalid{detail}")
                if lane == "highs-source":
                    identity = rec.get("build_identity")
                    if (rec.get("solver_version") != sv or
                            not isinstance(identity, dict) or
                            identity.get("kind") != "sor-highs-source-api-runner" or
                            not isinstance(source_info, dict) or
                            not isinstance(source_info.get("identity"), dict) or
                            any(identity.get(key) != source_info["identity"].get(key)
                                for key in ("runner_version", "compiler",
                                            "highs_source_commit",
                                            "highs_githash"))):
                        problems.append(
                            f"baseline source runner provenance for {inst} is invalid")
                else:
                    if rec.get("solver_version") != wv:
                        problems.append(
                            f"baseline wheel provenance for {inst} is invalid")
        source_rows = {str(r.get("instance")): r for r in active
                       if r.get("solver") == "highs-source"}
        wheel_rows = {str(r.get("instance")): r for r in active
                      if r.get("solver") == "highs-wheel"}
        for inst in sorted(expected & set(source_rows) & set(wheel_rows)):
            source = source_rows[inst]
            wheel = wheel_rows[inst]
            if source.get("configuration") != wheel.get("configuration"):
                problems.append(f"baseline source/wheel configuration mismatch for {inst}")
            if str(source.get("status", "")).lower() != \
                    str(wheel.get("status", "")).lower():
                problems.append(f"baseline source/wheel status mismatch for {inst}")
            so, wo = source.get("objective"), wheel.get("objective")
            if (isinstance(so, (int, float)) and isinstance(wo, (int, float)) and
                    math.isfinite(so) and math.isfinite(wo) and
                    abs(so - wo) > max(1e-7, 1e-7 * (1.0 + abs(so)))):
                problems.append(f"baseline source/wheel objective mismatch for {inst}")
    sor_labels = {str(r.get("solver")) for r in active
                  if str(r.get("solver", "")).lower().startswith("sor")}
    requested_label = _first_value(records, "candidate_solver")
    if requested_label is not None:
        extras = sorted(sor_labels - {str(requested_label)})
        if extras:
            problems.append(f"baseline has unexpected SOR solver lane(s): {extras}")
        sor_labels = {s for s in sor_labels if s == requested_label}
    if len(sor_labels) != 1:
        problems.append(f"baseline needs exactly one candidate solver lane; found {sorted(sor_labels)}")
    else:
        data.solver_label = next(iter(sor_labels))
        selected = [r for r in active if r.get("solver") == data.solver_label]
        by_instance: dict[str, list[dict]] = {}
        for rec in selected:
            by_instance.setdefault(str(rec.get("instance")), []).append(rec)
        if expected is not None:
            missing = sorted(expected - set(by_instance))
            unexpected = sorted(set(by_instance) - expected)
            if missing:
                problems.append(f"baseline misses {len(missing)} expected instance(s): {missing[:5]}")
            if unexpected:
                problems.append(f"baseline has unexpected instance(s): {unexpected[:5]}")
        for inst, rows in by_instance.items():
            if len(rows) != 1:
                problems.append(f"baseline has {len(rows)} selected aggregates for {inst}")
                continue
            rec = rows[0]
            solve = rec.get("median_s") if rec.get("median_s") is not None else rec.get("seconds")
            wall = (rec.get("median_wall_s") if rec.get("median_wall_s") is not None
                    else rec.get("wall_s"))
            if not _finite_positive(solve):
                problems.append(f"{inst}: baseline solver time is not finite and positive")
            else:
                data.solver[inst] = float(solve)
            if not _finite_positive(wall):
                problems.append(f"{inst}: baseline wall time is not finite and positive")
            else:
                data.wall[inst] = float(wall)

    return data, problems
