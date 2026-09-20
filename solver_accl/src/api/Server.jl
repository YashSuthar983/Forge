module SorApiServer

# Line protocol over stdin/stdout so a C++ (or any) caller can drive every
# engine in this library without embedding the Julia runtime.
#
# GRANULARITY IS THE WHOLE DESIGN. The earlier sidecar in the companion C++
# repo exposed per-operation primitives -- spmv, dot, project_box -- called
# once per vector per iteration, and measured 240x SLOWER than the plain CPU
# loop with 52% of runtime in JSON serialisation. Nothing here is finer than
# "solve this entire model", or "solve these K models", so the transport cost
# is paid once per solve rather than once per iteration. Do not add a verb
# that runs inside an iteration loop.
#
# Verbs:
#   PING
#   CAPABILITIES                                   -- the engine roster + this box's GPU
#   RECOMMEND <n> <m> <nnz_a> <nnz_q> <iters> <K>  -- cost-model advice, no solve
#   SOLVE <path> <engine> <max_iter> <tol>
#   SOLVE_BATCH <engine> <max_iter> <tol> <path...>
#   QUIT
#
# One JSON object per request on stdout, flushed. Logging goes to stderr so
# stdout carries protocol traffic only.

using ..SorCore: to_string, SolveResult
using ..SorCertify: finalize_result
using ..SorModel: LpProblem
using ..SorSimplex: solve_simplex, simplex_evidence, SimplexOptions, SimplexDiagnostics
using ..SorPdhg: PdhgOptions, PdhgDiagnostics
using ..SorHpr: HprOptions, HprDiagnostics
using ..SorSearch: BabOptions, BabDiagnostics, milp_evidence, solve_milp as solve_mip
using ..SorQpActiveSet: QpProblemG, QpOptionsG, QpDiagnosticsG, solve_qp_activeset
using ..SorQpIpm: QpOptionsIpm, QpDiagnosticsIpm, solve_qp_ipm
using ..SorQpAuto: solve_qp_auto
using ..SorSearchQp: MiqpOptions, MiqpDiagnostics, solve_miqp, miqp_evidence, miqp_raw_result
using ..QpHprCore: QpHprOptions, QpHprDiagnostics, solve_qp_hpr
using ..FirstOrderCore: solve_lp_firstorder
using ..BatchedFirstOrderCore: solve_lp_firstorder_batched
using ..SorCostModel: estimate_work, recommend_backend, explain_recommendation, GPU_BACKEND
using ..SorCLI
using ..SorApi
using ..SorApiConvert: to_one_sided, recover_solution
using ..SorApiGpu: detect_gpu, gpu_info, GpuProbe

export serve, handle_line, PROTOCOL_VERSION

const PROTOCOL_VERSION = 2

# ---------------------------------------------------------------- JSON out --
# Deliberately hand-rolled rather than taking a JSON dependency: the payloads
# here are flat dictionaries of strings, numbers and booleans, and the
# dependency ledger for a "from mathematical foundations" solver is worth
# keeping short.
jstr(s) = "\"" * replace(String(s), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""

function jval(v)
    v === nothing              && return "null"
    v isa Bool                 && return v ? "true" : "false"
    v isa AbstractString       && return jstr(v)
    v isa Symbol               && return jstr(String(v))
    if v isa AbstractFloat
        (isnan(v) || isinf(v)) && return "null"
        return string(v)
    end
    v isa Integer              && return string(v)
    v isa AbstractVector       && return "[" * join(map(jval, v), ",") * "]"
    v isa AbstractDict         && return jobj(v)
    return jstr(string(v))
end

jobj(d::AbstractDict) =
    "{" * join([jstr(String(k)) * ":" * jval(v) for (k, v) in sort(collect(d), by = first)], ",") * "}"

emit(io::IO, d::AbstractDict) = (println(io, jobj(d)); flush(io))

err(msg) = Dict{String,Any}("ok" => false, "error" => msg)

# Set an option field only when it exists, so this layer does not break every
# time an engine's option struct gains or renames a knob.
function set_if!(opts, field::Symbol, value)
    hasfield(typeof(opts), field) && setfield!(opts, field, value)
    return opts
end

# ------------------------------------------------------------ result shape --
function result_dict(r::SolveResult, engine::String, kind::String, elapsed_ms::Float64)
    d = Dict{String,Any}(
        "ok"            => true,
        "engine"        => engine,
        "engine_kind"   => kind,
        "status"        => String(to_string(r.status)),
        "proof_level"   => String(to_string(r.proof)),
        "objective"     => r.objective,
        "dual_bound"    => r.dual_bound,
        "iterations"    => Int(r.iterations),
        "wall_ms"       => elapsed_ms,
        "max_primal_violation" => r.max_primal_violation,
        "max_dual_violation"   => r.max_dual_violation,
        "gap_rel"       => r.gap_rel,
        "termination"   => r.termination_reason,
    )
    isempty(r.downgrade_reason) || (d["downgrade_reason"] = r.downgrade_reason)
    return d
end

# ------------------------------------------------------------------ solves --
function solve_lp_cpu(engine::String, lp::LpProblem, max_iter::Int, tol::Float64)
    if engine == "simplex"
        o = SimplexOptions(); set_if!(o, :max_iterations, UInt64(max_iter))
        set_if!(o, :primal_feas_tol, tol); set_if!(o, :dual_feas_tol, tol)
        d = SimplexDiagnostics()
        raw = solve_simplex(lp, o, d)
        return finalize_result(raw, simplex_evidence(d, o))
    elseif engine == "pdhg"
        o = PdhgOptions(); set_if!(o, :max_iterations, UInt64(max_iter))
        set_if!(o, :primal_tol, tol); set_if!(o, :dual_tol, tol)
        d = PdhgDiagnostics()
        raw, ev = SorCLI.call_solve_pdhg(lp, o, "cpu", d)
        return finalize_result(raw, ev)
    elseif engine == "hpr"
        o = HprOptions(); set_if!(o, :max_iterations, UInt64(max_iter))
        set_if!(o, :primal_tol, tol); set_if!(o, :dual_tol, tol)
        d = HprDiagnostics()
        raw, ev = SorCLI.call_solve_hpr(lp, o, "cpu", d)
        return finalize_result(raw, ev)
    end
    error("not an LP CPU engine: $engine")
end

# The KA LP engines take the legacy one-sided model, so a real MPS has to be
# transformed first. `to_one_sided` grows the row and column counts; the
# reported shape below is the TRANSFORMED one, on purpose -- quoting a GPU
# time against the original shape would misstate what was solved.
function solve_lp_gpu(lp::LpProblem, max_iter::Int, tol::Float64, probe::GpuProbe)
    conv, map = to_one_sided(lp)
    t0 = time_ns()
    r = solve_lp_firstorder(conv; backend = probe.backend, max_iter = max_iter, tol = tol)
    ms = (time_ns() - t0) / 1e6
    x = recover_solution(map, r.x)
    return Dict{String,Any}(
        "ok"            => true,
        "engine"        => "pdhg-gpu",
        "engine_kind"   => "gpu",
        "status"        => r.converged ? "Feasible" : "Interrupted",
        "proof_level"   => r.converged ? "FeasibleOnly" : "None",
        "objective"     => r.objective + map.obj_offset,
        "iterations"    => r.iterations,
        "residual"      => r.residual,
        "wall_ms"       => ms,
        "converted_rows" => map.m_conv,
        "converted_cols" => map.n_conv,
        "note"          => "solved on the one-sided transform of the model; shape differs from the original",
    ), x
end

function solve_qp(engine::String, path::String, max_iter::Int, tol::Float64, probe::GpuProbe)
    lp, Qmat, int_mask, msg = SorCLI._load_qp_model(path, "")
    isempty(msg) || return err(msg)

    if engine == "miqp"
        o = MiqpOptions(); d = MiqpDiagnostics()
        t0 = time_ns()
        res = solve_miqp(QpProblemG(lp, Qmat), int_mask, o, d)
        ms = (time_ns() - t0) / 1e6
        raw = miqp_raw_result(res, d)
        r = finalize_result(raw, miqp_evidence(d, o))
        return result_dict(r, engine, "cpu", ms)
    end

    qpg = QpProblemG(lp, Qmat)
    if engine == "qp-hpr-gpu"
        o = QpHprOptions(); set_if!(o, :max_iterations, UInt64(max_iter))
        d = QpHprDiagnostics()
        t0 = time_ns()
        raw = solve_qp_hpr(lp, Qmat, o, d; backend = probe.backend)
        ms = (time_ns() - t0) / 1e6
        out = Dict{String,Any}("ok" => true, "engine" => engine, "engine_kind" => "gpu",
                               "status" => String(to_string(raw.proposed_status)),
                               "proof_level" => String(to_string(raw.proposed_level)),
                               "objective" => raw.objective, "iterations" => Int(raw.iterations),
                               "wall_ms" => ms)
        return out
    end

    t0 = time_ns()
    raw = if engine == "qp-activeset"
        solve_qp_activeset(qpg, QpOptionsG(), QpDiagnosticsG())
    elseif engine == "qp-ipm"
        solve_qp_ipm(qpg, QpOptionsIpm(), QpDiagnosticsIpm())
    else
        solve_qp_auto(qpg, QpOptionsG(), QpOptionsIpm(), QpDiagnosticsG())
    end
    ms = (time_ns() - t0) / 1e6
    return Dict{String,Any}("ok" => true, "engine" => engine, "engine_kind" => "cpu",
                            "status" => String(to_string(raw.proposed_status)),
                            "proof_level" => String(to_string(raw.proposed_level)),
                            "objective" => raw.objective, "iterations" => Int(raw.iterations),
                            "wall_ms" => ms)
end

# ------------------------------------------------------------------ verbs ---
function do_capabilities(probe::GpuProbe)
    rows = Any[]
    for e in SorApi.ENGINES
        push!(rows, Dict{String,Any}(
            "name"          => e.name,
            "kind"          => SorApi.kind_string(e.kind),
            "class"         => SorApi.class_string(e.class),
            "batched"       => e.batched,
            "proof_ceiling" => String(to_string(e.proof_ceiling)),
            "model"         => String(e.model),
            # A GPU engine is listed always, but usable only when a device
            # was found; the caller must be able to tell those apart.
            "usable"        => (e.kind == SorApi.CPU_ENGINE) || probe.available,
            "notes"         => e.notes,
        ))
    end
    d = Dict{String,Any}("ok" => true, "protocol" => PROTOCOL_VERSION,
                         "julia" => string(VERSION), "engines" => rows)
    merge!(d, gpu_info(probe))
    return d
end

function do_recommend(parts::Vector{SubString{String}})
    length(parts) >= 7 || return err("RECOMMEND needs <n> <m> <nnz_a> <nnz_q> <iters> <K>")
    n, m, nnz_a, nnz_q, iters, K = (parse(Int, parts[i]) for i in 2:7)
    w = estimate_work(n = n, m = m, nnz_a = nnz_a, nnz_q = nnz_q,
                      est_iterations = iters, batch_size = K)
    choice = recommend_backend(w)
    return Dict{String,Any}("ok" => true,
                            "backend" => choice == GPU_BACKEND ? "gpu" : "cpu",
                            "total_flops" => w.total_flops,
                            "reason" => explain_recommendation(w))
end

function do_solve(parts::Vector{SubString{String}}, probe::GpuProbe)
    length(parts) >= 3 || return err("SOLVE needs <path> <engine> [max_iter] [tol]")
    path   = String(parts[2])
    engine = lowercase(String(parts[3]))
    max_iter = length(parts) >= 4 ? parse(Int, parts[4]) : 100_000
    tol      = length(parts) >= 5 ? parse(Float64, parts[5]) : 1e-8

    isfile(path) || return err("no such file: $path")
    spec = SorApi.find_engine(engine)
    spec === nothing && return err("unknown engine '$engine'; see CAPABILITIES")
    if spec.kind == SorApi.GPU_ENGINE && !probe.available
        return err("engine '$engine' needs a GPU; none detected ($(probe.detail))")
    end
    spec.batched && return err("engine '$engine' is batched; use SOLVE_BATCH")

    try
        if spec.class == SorApi.CLASS_QP || spec.class == SorApi.CLASS_MIQP
            return solve_qp(engine, path, max_iter, tol, probe)
        end

        lp, rep = SorCLI.read_mps_file_auto(path)
        lp === nothing && return err("could not read $path")

        if engine == "pdhg-gpu"
            d, _ = solve_lp_gpu(lp, max_iter, tol, probe)
            return d
        elseif engine == "milp"
            o = BabOptions(); set_if!(o, :max_nodes, max_iter)
            dg = BabDiagnostics()
            t0 = time_ns()
            raw = solve_mip(lp, o, dg)
            ms = (time_ns() - t0) / 1e6
            return result_dict(finalize_result(raw, milp_evidence(dg, o)), engine, "cpu", ms)
        else
            t0 = time_ns()
            r = solve_lp_cpu(engine, lp, max_iter, tol)
            ms = (time_ns() - t0) / 1e6
            return result_dict(r, engine, "cpu", ms)
        end
    catch e
        return err("$engine failed: " * sprint(showerror, e))
    end
end

function do_solve_batch(parts::Vector{SubString{String}}, probe::GpuProbe)
    length(parts) >= 5 || return err("SOLVE_BATCH needs <engine> <max_iter> <tol> <path...>")
    engine = lowercase(String(parts[2]))
    max_iter = parse(Int, parts[3])
    tol      = parse(Float64, parts[4])
    paths    = String.(parts[5:end])

    spec = SorApi.find_engine(engine)
    spec === nothing && return err("unknown engine '$engine'; see CAPABILITIES")
    spec.batched || return err("engine '$engine' is not batched; use SOLVE")
    probe.available || return err("batched engines need a GPU; none detected ($(probe.detail))")

    for p in paths
        isfile(p) || return err("no such file: $p")
    end

    try
        if engine == "pdhg-gpu-batched"
            convs = Vector{Any}(undef, length(paths))
            maps  = Vector{Any}(undef, length(paths))
            for (i, p) in enumerate(paths)
                lp, _ = SorCLI.read_mps_file_auto(p)
                lp === nothing && return err("could not read $p")
                convs[i], maps[i] = to_one_sided(lp)
            end
            t0 = time_ns()
            r = solve_lp_firstorder_batched(convs; backend = probe.backend,
                                            max_iter = max_iter, tol = tol)
            ms = (time_ns() - t0) / 1e6
            return Dict{String,Any}("ok" => true, "engine" => engine, "engine_kind" => "gpu",
                                    "batch_size" => length(paths),
                                    "objective" => [r.objective[k] + maps[k].obj_offset
                                                    for k in eachindex(paths)],
                                    "converged" => collect(r.converged),
                                    "residual"  => collect(r.residual),
                                    "iterations" => r.iterations,
                                    "wall_ms" => ms)
        end
        return err("SOLVE_BATCH does not yet route '$engine'; qp-hpr-gpu-batched takes " *
                   "K bound-perturbations of one model, which this file-list protocol " *
                   "cannot express -- call it in-process for now")
    catch e
        return err("$engine failed: " * sprint(showerror, e))
    end
end

# ------------------------------------------------------------------ driver --
"""
    handle_line(line, probe) -> Dict

Pure request/response, split out from the IO loop so the protocol is testable
without spawning a process.
"""
function handle_line(line::AbstractString, probe::GpuProbe)
    parts = split(strip(line))
    isempty(parts) && return err("empty request")
    verb = uppercase(String(parts[1]))

    verb == "PING"         && return Dict{String,Any}("ok" => true, "pong" => true,
                                                      "julia" => string(VERSION),
                                                      "protocol" => PROTOCOL_VERSION)
    verb == "CAPABILITIES" && return do_capabilities(probe)
    verb == "RECOMMEND"    && return do_recommend(parts)
    verb == "SOLVE"        && return do_solve(parts, probe)
    verb == "SOLVE_BATCH"  && return do_solve_batch(parts, probe)
    verb == "QUIT"         && return Dict{String,Any}("ok" => true, "bye" => true)
    return err("unknown verb '$verb'")
end

"""
    serve(; in=stdin, out=stdout)

Read requests until QUIT or EOF. The GPU probe runs once at startup, not per
request -- probing costs seconds the first time a vendor package loads.
"""
function serve(; in::IO = stdin, out::IO = stdout)
    probe = detect_gpu()
    println(stderr, "SORACCL_READY vendor=$(probe.vendor) device=$(probe.device_name)")
    flush(stderr)
    for line in eachline(in)
        isempty(strip(line)) && continue
        d = handle_line(line, probe)
        emit(out, d)
        uppercase(String(first(split(strip(line))))) == "QUIT" && break
    end
    return nothing
end

end # module
