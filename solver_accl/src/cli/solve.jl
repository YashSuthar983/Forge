module SorCLI

# CLI port of sor/cli/sor_solve.cpp. Loaded by bin/sor_solve (not SovereignSolver.jl).

using Printf
using QPSReader
using ..SorCore: Status, ProofLevel, RawResult, SolveResult, ProofEvidence
using ..SorCore: Optimal, Feasible, Interrupted, to_string, human_line, kNaN
using ..SorModel: LpProblem, kInf, n_integer
import ..SorModel: n_rows, n_cols, nnz
using ..SorSparse: from_triplets
using ..SorCertify: finalize_result
using ..SorIO: read_mps_file_auto as io_read_mps, MpsReadReport, write_solution as io_write_solution
using ..SorPdhg: solve_pdhg as pdhg_solve, pdhg_evidence as pdhg_ev
using ..SorPdhg: PdhgOptions as PortPdhgOptions, PdhgDiagnostics as PortPdhgDiagnostics
using ..SorHpr: solve_hpr as hpr_solve, hpr_evidence as hpr_ev
using ..SorHpr: HprOptions as PortHprOptions, HprDiagnostics as PortHprDiagnostics
using ..SorBackend: make_cpu_backend, make_cpu_lp_device
using ..SorQp: solve_qp_diag, QpProblem, QpOptions, QpDiagnostics, qp_evidence
using ..SorQpActiveSet: QpProblemG, QpOptionsG, QpDiagnosticsG
using ..SorQpActiveSet: qp_activeset_evidence
using ..SorQpIpm: QpOptionsIpm
using ..SorQpAuto: solve_qp_auto
using ..SorSearch: BabOptions, BabDiagnostics, solve_milp, milp_evidence
using ..SorSearch: SimplexOptions, SimplexDiagnostics, SimplexMethod
using ..SorSearch: SimplexAuto, SimplexPrimal, SimplexDual
using ..SorSearch: solve_simplex!, simplex_evidence
using ..SorSearch: PdhgOptions, PdhgDiagnostics, HprOptions, HprDiagnostics
using ..SorSearch: TransferStats
using ..SorSearchQp: MiqpOptions, MiqpDiagnostics, solve_miqp, miqp_evidence, miqp_raw_result

export sor_solve_main

function _find_mod(name::Symbol)
    for m in (Main,)
        isdefined(m, name) && return getfield(m, name)
    end
    for (_, mod) in Base.loaded_modules
        nameof(mod) === name && return mod
        isdefined(mod, name) && return getfield(mod, name)
    end
    return nothing
end

function usage()
    print(stderr,
        "usage: sor_solve MODEL.mps [options]\n",
        "  --engine NAME    auto | simplex (default) | pdhg | hpr | milp | qp | miqp\n",
        "                   'auto' sniffs the file for a QUADOBJ/quadratic section\n",
        "                   and integer/binary columns, then picks the narrowest\n",
        "                   correct engine (simplex/milp/qp/miqp) itself -- the\n",
        "                   recommended way to point this at an arbitrary .mps/.qps\n",
        "                   file without knowing its problem class up front.\n",
        "  --q-diag LIST    comma-separated diagonal of Q (if not using .qps)\n",
        "  --backend NAME   cpu (default) | vulkan | julia_gpu\n",
        "  --method NAME    auto | primal | dual   (simplex and MILP node LPs)\n",
        "  --max-iter N     iteration / node limit\n",
        "  --tol T          feasibility tolerance\n",
        "  --time-limit S   wall-clock limit in seconds\n",
        "  --no-scaling     skip Ruiz equilibration\n",
        "  --no-presolve    skip presolve (simplex/milp)\n",
        "  --verbose        iteration / node log\n",
        "  --hpr-vanilla | --hpr-full\n",
        "  --solution-out PATH   write a plain-text solution file for sor_check\n")
end

function write_f64_tok(v::Float64)
    isnan(v) && return "nan"
    v == Inf && return "inf"
    v == -Inf && return "-inf"
    return @sprintf("%.17g", v)
end

function write_solution_out(path::AbstractString, r::SolveResult)
    isempty(path) && return
    try
        open(path, "w") do out
            println(out, "status ", to_string(r.status))
            println(out, "proof ", to_string(r.proof))
            println(out, "objective ", write_f64_tok(r.objective))
            for (tag, vec) in (("x", r.x), ("y", r.y), ("ray", r.ray))
                print(out, tag, ' ', length(vec))
                for v in vec
                    print(out, ' ', write_f64_tok(v))
                end
                println(out)
            end
        end
    catch
        @printf(stderr, "warning: could not open '%s' for --solution-out\n", path)
    end
end

function print_result(r::SolveResult)
    @printf("\nstatus:            %s\n", to_string(r.status))
    @printf("proof_level:       %s\n", to_string(r.proof))
    @printf("                   %s\n", human_line(r.status, r.proof))
    !isempty(r.downgrade_reason) && @printf("downgrade:         %s\n", r.downgrade_reason)
    @printf("objective:         %.10e\n", r.objective)
end

function exit_code_for(s::Status)
    s === Optimal || s === Feasible ? Cint(0) :
    s === Interrupted ? Cint(4) : Cint(5)
end

function print_transfer(s::TransferStats)
    @printf("  kernel           %10.3f\n", s.kernel_ms)
    @printf("  host->device     %10.3f  (%llu bytes)\n", s.h2d_ms, s.h2d_bytes)
    @printf("  device->host     %10.3f  (%llu bytes)\n", s.d2h_ms, s.d2h_bytes)
    @printf("  ipc overhead     %10.3f\n", s.ipc_ms)
    @printf("  kernel calls     %llu\n", s.calls)
end

function _int_mask(qps)
    n = qps.nvar
    mask = fill(false, n)
    vt = qps.vartypes
    length(vt) == n || return mask
    for j in 1:n
        t = vt[j]
        mask[j] = t == QPSReader.VTYPE_Binary ||
                  t == QPSReader.VTYPE_Integer ||
                  t == QPSReader.VTYPE_SemiInteger
        if t == QPSReader.VTYPE_Binary
            # keep file bounds; binary typically [0,1]
        end
    end
    return mask
end

function qps_to_lp(qps, path::AbstractString)
    m, n = qps.ncon, qps.nvar
    A = from_triplets(m, n, qps.arows, qps.acols, qps.avals)
    lo = Float64.(qps.lvar)
    hi = Float64.(qps.uvar)
    for j in 1:n
        !isfinite(hi[j]) && (hi[j] = kInf)
        !isfinite(lo[j]) && (lo[j] = -kInf)
    end
    rlo = Float64.(qps.lcon)
    rhi = Float64.(qps.ucon)
    for i in 1:m
        !isfinite(rhi[i]) && (rhi[i] = kInf)
        !isfinite(rlo[i]) && (rlo[i] = -kInf)
    end
    name = isempty(qps.name) ? path : String(qps.name)
    return LpProblem(name, A, Float64.(qps.c), Float64(qps.c0),
                     qps.objsense == :max, rlo, rhi, lo, hi,
                     _int_mask(qps),
                     String.(qps.connames), String.(qps.varnames)),
           count(_int_mask(qps))
end

# Shared QPS/QUADOBJ loading for both --engine qp and --engine miqp --
# returns ("", ...) as the error message on success, a human-readable
# message otherwise, so each caller can print+exit with its own preferred
# formatting rather than this function picking one for both.
function _load_qp_model(path::AbstractString, q_diag_arg::AbstractString)
    qps = read_qps_any(path)
    lp, _ = qps_to_lp(qps, path)
    n = Int(n_cols(lp.A))
    int_mask = _int_mask(qps)

    Qmat = zeros(n, n)
    has_q = false
    for t in eachindex(qps.qrows)
        i, j, v = Int(qps.qrows[t]), Int(qps.qcols[t]), Float64(qps.qvals[t])
        Qmat[i, j] += v
        i != j && (Qmat[j, i] += v)
        has_q = true
    end
    if !has_q && !isempty(q_diag_arg)
        vals = parse.(Float64, split(q_diag_arg, ","))
        if length(vals) != n
            return lp, Qmat, int_mask,
                   "--q-diag has $(length(vals)) values, model has $n columns"
        end
        for j in 1:n
            Qmat[j, j] = vals[j]
        end
        has_q = true
    end
    has_q || return lp, Qmat, int_mask,
                    "no quadratic terms -- need a QUADOBJ section in the file or --q-diag"
    return lp, Qmat, int_mask, ""
end

function read_mps_file_auto(path::AbstractString; fixed::Union{Nothing,Bool}=nothing)
    try
        report = MpsReadReport()
        lp = io_read_mps(path, report)
        return (problem=lp, n_integer=Int(report.n_integer), warnings=report.warnings)
    catch
    end
    warnings = String[]
    if fixed === true
        qps = readqps(path; mpsformat=:fixed)
        lp, nint = qps_to_lp(qps, path)
        return (problem=lp, n_integer=nint, warnings=warnings)
    elseif fixed === false
        qps = readqps(path; mpsformat=:free)
        lp, nint = qps_to_lp(qps, path)
        return (problem=lp, n_integer=nint, warnings=warnings)
    end
    local qps
    try
        qps = readqps(path; mpsformat=:free)
    catch
        qps = readqps(path; mpsformat=:fixed)
    end
    lp, nint = qps_to_lp(qps, path)
    return (problem=lp, n_integer=nint, warnings=warnings)
end

function read_qps_any(path::AbstractString)
    try
        return readqps(path; mpsformat=:free)
    catch
        return readqps(path; mpsformat=:fixed)
    end
end

# --engine auto: read the file once through QPSReader (which parses QUADOBJ
# and BOUNDS/integer markers uniformly whether or not either is present --
# an ordinary LP just comes back with qrows/vartypes empty) purely to
# classify it, then hand off to the exact same, already-tested branch a
# user would have picked by hand. If QPSReader itself can't parse the file
# at all, fall back to "simplex" -- the existing default -- rather than
# fail auto-detection outright; `read_mps_file_auto`'s own native-reader
# path (tried first for every non-QP engine) gets a chance to read it from
# there, exactly as it already does today when no --engine is given.
function sniff_engine(path::AbstractString)
    local qps
    try
        qps = read_qps_any(path)
    catch
        return "simplex"
    end
    has_q = !isempty(qps.qrows)
    n_int = count(_int_mask(qps))
    has_q && n_int > 0 && return "miqp"
    has_q && return "qp"
    n_int > 0 && return "milp"
    return "simplex"
end

function call_solve_pdhg(problem, pdhg_opts, backend_name, diag)
    po = PortPdhgOptions()
    po.max_iterations = pdhg_opts.max_iterations
    po.check_every = pdhg_opts.check_every
    po.primal_tol = pdhg_opts.primal_tol
    po.dual_tol = pdhg_opts.dual_tol
    po.ruiz_iterations = pdhg_opts.ruiz_iterations
    po.verbose = pdhg_opts.verbose
    pd = PortPdhgDiagnostics()
    raw = pdhg_solve(problem, po, make_cpu_backend(), pd)
    diag.iterations = pd.iterations
    diag.primal_residual = pd.primal_residual
    diag.dual_residual = pd.dual_residual
    diag.gap_rel = pd.gap_rel
    diag.dual_bound_finite = pd.dual_bound_finite
    diag.total_ms = pd.total_ms
    return raw, pdhg_ev(pd, po)
end

function call_solve_hpr(problem, hpr_opts, backend_name, diag)
    ho = PortHprOptions()
    ho.max_iterations = hpr_opts.max_iterations
    ho.check_every = hpr_opts.check_every
    ho.primal_tol = hpr_opts.primal_tol
    ho.dual_tol = hpr_opts.dual_tol
    ho.gap_tol = hpr_opts.gap_tol
    ho.ruiz_iterations = hpr_opts.ruiz_iterations
    ho.time_limit_s = hpr_opts.time_limit_s
    ho.use_primal_weight = hpr_opts.use_primal_weight
    ho.use_restart = hpr_opts.use_restart
    ho.use_halpern = hpr_opts.use_halpern
    ho.verbose = hpr_opts.verbose
    hd = PortHprDiagnostics()
    raw = hpr_solve(problem, ho, make_cpu_lp_device(), hd)
    diag.iterations = hd.iterations
    diag.primal_residual = hd.primal_residual
    diag.dual_residual = hd.dual_residual
    diag.gap_rel = hd.gap_rel
    diag.dual_bound_finite = hd.dual_bound_finite
    diag.total_ms = hd.total_ms
    diag.termination_reason = hd.termination_reason
    return raw, hpr_ev(hd, ho)
end

function need_value(args, i, what)
    i + 1 > length(args) && error("$what needs a value")
    return args[i + 1], i + 1
end

function sor_solve_main(ARGS::Vector{String})::Cint
    if length(ARGS) < 1
        usage()
        return Cint(2)
    end

    path = ""
    backend_name = "cpu"
    engine_name = "simplex"
    q_diag_arg = ""
    solution_out = ""
    pdhg_opts = PdhgOptions()
    hpr_opts = HprOptions()
    sx_opts = SimplexOptions()
    mps_fixed = nothing
    tol_given = false
    tol = 0.0

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--backend"
            backend_name, i = need_value(ARGS, i, "--backend")
        elseif a == "--engine"
            engine_name, i = need_value(ARGS, i, "--engine")
        elseif a == "--q-diag"
            q_diag_arg, i = need_value(ARGS, i, "--q-diag")
        elseif a == "--max-iter"
            v, i = need_value(ARGS, i, "--max-iter")
            n = parse(UInt64, v)
            pdhg_opts.max_iterations = n
            hpr_opts.max_iterations = n
            sx_opts.max_iterations = n
        elseif a == "--tol"
            v, i = need_value(ARGS, i, "--tol")
            tol = parse(Float64, v)
            tol_given = true
        elseif a == "--time-limit"
            v, i = need_value(ARGS, i, "--time-limit")
            t = parse(Float64, v)
            sx_opts.time_limit_s = t
            hpr_opts.time_limit_s = t
        elseif a == "--method"
            mth, i = need_value(ARGS, i, "--method")
            if mth == "auto"
                sx_opts.method = SimplexAuto
            elseif mth == "primal"
                sx_opts.method = SimplexPrimal
            elseif mth == "dual"
                sx_opts.method = SimplexDual
            else
                @printf(stderr, "error: unknown method '%s'\n", mth)
                return Cint(2)
            end
        elseif a == "--no-scaling"
            sx_opts.ruiz_iterations = 0
            pdhg_opts.ruiz_iterations = 0
            hpr_opts.ruiz_iterations = 0
        elseif a == "--no-presolve"
            sx_opts.presolve = false
        elseif a == "--fixed-mps"
            mps_fixed = true
        elseif a == "--free-mps"
            mps_fixed = false
        elseif a == "--verbose"
            pdhg_opts.verbose = true
            hpr_opts.verbose = true
            sx_opts.verbose = true
        elseif a == "--hpr-vanilla"
            hpr_opts.use_primal_weight = false
            hpr_opts.use_restart = false
            hpr_opts.use_halpern = false
        elseif a == "--hpr-full"
            hpr_opts.use_primal_weight = true
            hpr_opts.use_restart = true
            hpr_opts.use_halpern = true
        elseif a == "--solution-out"
            solution_out, i = need_value(ARGS, i, "--solution-out")
        elseif a == "-h" || a == "--help"
            usage()
            return Cint(0)
        elseif !isempty(a) && a[1] == '-'
            @printf(stderr, "error: unknown option '%s'\n", a)
            return Cint(2)
        else
            path = a
        end
        i += 1
    end

    if isempty(path)
        usage()
        return Cint(2)
    end
    if engine_name != "auto" && engine_name != "pdhg" && engine_name != "simplex" &&
       engine_name != "hpr" && engine_name != "milp" && engine_name != "qp" &&
       engine_name != "miqp"
        @printf(stderr,
                "error: engine '%s' not implemented (have auto|simplex|pdhg|hpr|milp|qp|miqp)\n",
                engine_name)
        return Cint(3)
    end
    if tol_given
        pdhg_opts.primal_tol = pdhg_opts.dual_tol = tol
        hpr_opts.primal_tol = hpr_opts.dual_tol = hpr_opts.gap_tol = tol
        sx_opts.primal_feas_tol = sx_opts.dual_feas_tol = tol
    end

    if engine_name == "auto"
        engine_name = sniff_engine(path)
        @printf("auto-detected engine: %s\n", engine_name)
    end

    try
        if engine_name == "qp"
            lp, Qmat, int_mask_qp, errmsg = _load_qp_model(path, q_diag_arg)
            if !isempty(errmsg)
                @printf(stderr, "error: %s\n", errmsg)
                return Cint(2)
            end
            n = Int(n_cols(lp.A))
            n_int_qp = count(int_mask_qp)
            n_int_qp > 0 && println(stderr,
                "warning: integer/binary columns present; --engine qp solves the QP " *
                "relaxation (use --engine miqp for the integer problem)")

            @printf("model:             %s\n", isempty(lp.name) ? path : lp.name)
            @printf("rows x cols:       %d x %d   nnz %d\n",
                    Int(n_rows(lp.A)), n, Int(nnz(lp.A)))
            @printf("engine:            qp\n")

            # SorQp's diagonal dual method is only correct for diagonal Q and
            # equality-only rows -- dispatch to it when the file's actual
            # shape fits that fast path, and to the general active-set
            # engine (SorQpActiveSet) otherwise. Mirrors SorSimplex's
            # Auto-falls-back-to-a-narrower-fast-path pattern, not a new one.
            diagonal_only = true
            for i in 1:n, j in 1:n
                i != j && Qmat[i, j] != 0.0 && (diagonal_only = false)
            end
            m_rows = Int(n_rows(lp.A))
            all_eq_rows = all(isfinite(lp.row_lo[i]) && lp.row_lo[i] == lp.row_hi[i]
                              for i in 1:m_rows)

            local raw, ev, total_ms
            if diagonal_only && all_eq_rows
                qp = QpProblem(lp, [Qmat[j, j] for j in 1:n])
                qd = QpDiagnostics()
                raw = solve_qp_diag(qp, QpOptions(), qd)
                ev = qp_evidence(qd, QpOptions())
                total_ms = qd.total_ms
            else
                qpg = QpProblemG(lp, Qmat)
                qdg = QpDiagnosticsG()
                raw = solve_qp_auto(qpg, QpOptionsG(), QpOptionsIpm(), qdg)
                ev = qp_activeset_evidence(qdg, QpOptionsG())
                total_ms = qdg.total_ms
            end
            r = finalize_result(raw, ev)
            print_result(r)
            write_solution_out(solution_out, r)
            @printf("max primal viol:   %.3e\n", r.max_primal_violation)
            @printf("dual residual:     %.3e\n", r.max_dual_violation)
            @printf("iterations:        %llu\n", r.iterations)
            println("\ntiming (ms)")
            @printf("  total            %10.3f\n", total_ms)
            return exit_code_for(r.status)
        end

        if engine_name == "miqp"
            lp, Qmat, int_mask, errmsg = _load_qp_model(path, q_diag_arg)
            if !isempty(errmsg)
                @printf(stderr, "error: %s\n", errmsg)
                return Cint(2)
            end
            n = Int(n_cols(lp.A))
            n_int_miqp = count(int_mask)
            n_int_miqp == 0 &&
                println(stderr, "warning: no integer/binary columns; miqp reduces to qp")

            @printf("model:             %s\n", isempty(lp.name) ? path : lp.name)
            @printf("rows x cols:       %d x %d   nnz %d\n",
                    Int(n_rows(lp.A)), n, Int(nnz(lp.A)))
            n_int_miqp > 0 && @printf("integer columns:   %d\n", n_int_miqp)
            @printf("engine:            miqp\n")

            qpg = QpProblemG(lp, Qmat)
            mopts = MiqpOptions()
            sx_opts.time_limit_s > 0.0 && (mopts.time_limit_s = sx_opts.time_limit_s)
            sx_opts.max_iterations != 0 && (mopts.max_nodes = sx_opts.max_iterations)
            tol_given && (mopts.int_tol = max(tol, 1e-9))
            mdiag = MiqpDiagnostics()
            mresult = solve_miqp(qpg, int_mask, mopts, mdiag)
            raw = miqp_raw_result(mresult)
            ev = miqp_evidence(mresult, mopts)
            r = finalize_result(raw, ev)
            print_result(r)
            write_solution_out(solution_out, r)
            if isfinite(mresult.dual_bound)
                @printf("dual bound:        %.10e\n", mresult.dual_bound)
                @printf("mip gap:           %.3e\n", mresult.gap)
            end
            @printf("nodes:             %llu\n", mdiag.nodes)
            @printf("qp solves:         %llu\n", mdiag.qp_solves)
            @printf("gpu wavefronts:    %llu  (batched qp solves %llu, fallbacks %llu)\n",
                    mdiag.gpu_wavefronts, mdiag.gpu_batch_qp_solves, mdiag.gpu_batch_fallbacks)
            @printf("termination:       %s\n", r.termination_reason)
            println("\ntiming (ms)")
            @printf("  total            %10.3f\n", mdiag.total_ms)
            return exit_code_for(r.status)
        end

        loaded = read_mps_file_auto(path; fixed=mps_fixed)
        problem = loaded.problem
        n_int = loaded.n_integer
        for w in loaded.warnings
            @printf(stderr, "warning: %s\n", w)
        end

        @printf("model:             %s\n",
                isempty(problem.name) ? path : problem.name)
        @printf("rows x cols:       %d x %d   nnz %d\n",
                Int(n_rows(problem)),
                Int(n_cols(problem)),
                Int(nnz(problem)))
        n_int > 0 && @printf("integer columns:   %d\n", n_int)
        @printf("engine:            %s\n", engine_name)

        if engine_name == "milp"
            n_int == 0 && println(stderr, "warning: no integer columns; milp reduces to LP")
            bab = BabOptions()
            bab.lp = sx_opts
            bab.verbose = sx_opts.verbose
            bab.lp.max_iterations = 0
            sx_opts.max_iterations != 0 && (bab.max_nodes = sx_opts.max_iterations)
            sx_opts.time_limit_s > 0.0 && (bab.time_limit_s = sx_opts.time_limit_s)
            if tol_given
                bab.primal_feas_tol = tol
                bab.int_tol = max(tol, 1e-9)
            end
            diag = BabDiagnostics()
            raw = solve_milp(problem, bab, diag)
            ev = milp_evidence(diag, bab)
            r = finalize_result(raw, ev)
            print_result(r)
            write_solution_out(solution_out, r)
            if isfinite(diag.dual_bound)
                @printf("dual bound:        %.10e\n", diag.dual_bound)
                @printf("mip gap:           %.3e\n", diag.gap_rel)
            end
            @printf("nodes:             %llu\n", diag.nodes)
            @printf("lp solves:         %llu\n", diag.lp_solves)
            @printf("lp fallbacks:      %llu\n", diag.lp_fallbacks)
            @printf("integer row roundings: %llu\n", diag.integer_row_roundings)
            @printf("binary cover cuts: %llu\n", diag.binary_cover_cuts)
            @printf("strong branch LPs: %llu  (pseudocost updates %llu)\n",
                    diag.strong_branch_solves, diag.pseudocost_updates)
            @printf("integer feas:      %llu  (heuristic hits %llu)\n",
                    diag.integer_feasible, diag.heuristic_hits)
            @printf("LP repair:         %llu attempts, %llu hits\n",
                    diag.lp_repair_attempts, diag.lp_repair_hits)
            @printf("feasibility pump:  %llu attempts, %llu hits\n",
                    diag.feasibility_pump_attempts, diag.feasibility_pump_hits)
            @printf("integer dive:      %llu attempts, %llu LPs, %llu hits\n",
                    diag.integer_dive_attempts, diag.integer_dive_lp_solves,
                    diag.integer_dive_hits)
            @printf("RENS:              %llu attempts, %llu LPs, %llu hits\n",
                    diag.rens_attempts, diag.rens_lp_solves, diag.rens_hits)
            @printf("integer neighborhood: %llu attempts, %llu trials, %llu hits\n",
                    diag.integer_neighborhood_attempts,
                    diag.integer_neighborhood_trials, diag.integer_neighborhood_hits)
            @printf("termination:       %s\n", r.termination_reason)
            println("\ntiming (ms)")
            @printf("  total            %10.3f\n", diag.total_ms)
            return exit_code_for(r.status)
        end

        if engine_name == "simplex"
            n_int > 0 && println("NOTE:              solving the LP RELAXATION ",
                                 "(use --engine milp for branch-and-bound)")
            diag = SimplexDiagnostics()
            raw = solve_simplex!(problem, sx_opts, diag, nothing)
            ev = simplex_evidence(diag, sx_opts)
            r = finalize_result(raw, ev)
            print_result(r)
            write_solution_out(solution_out, r)
            if diag.dual_bound_finite
                @printf("dual bound:        %.10e\n", r.dual_bound)
                @printf("rel gap:           %.3e\n", r.gap_rel)
            end
            @printf("max primal viol:   %.3e\n", r.max_primal_violation)
            @printf("dual residual:     %.3e\n", r.max_dual_violation)
            @printf("iterations:        %llu  (phase1 %llu, phase2 %llu)\n",
                    diag.iterations, diag.phase1_iterations, diag.phase2_iterations)
            @printf("termination:       %s\n", r.termination_reason)
            println("\ntiming (ms)")
            @printf("  total            %10.3f\n", diag.total_ms)
            @printf("  scaling          %10.3f\n", diag.scaling_ms)
            @printf("  factorization    %10.3f\n", diag.factor_ms)
            @printf("  simplex loop     %10.3f\n", diag.loop_ms)
            return exit_code_for(r.status)
        end

        if engine_name == "hpr"
            if backend_name != "cpu"
                @printf(stderr, "warning: LpDevice '%s' unavailable; using cpu\n", backend_name)
            end
            println("backend:           cpu (accelerated=no)")
            diag = HprDiagnostics()
            raw, ev = call_solve_hpr(problem, hpr_opts, backend_name, diag)
            r = finalize_result(raw, ev)
            print_result(r)
            write_solution_out(solution_out, r)
            @printf("max row violation: %.3e\n", r.max_primal_violation)
            @printf("dual residual:     %.3e\n", r.max_dual_violation)
            @printf("iterations:        %llu\n", r.iterations)
            @printf("termination:       %s\n", r.termination_reason)
            println("\ntiming (ms)")
            @printf("  total            %10.3f\n", diag.total_ms)
            print_transfer(diag.device_stats)
            return exit_code_for(r.status)
        end

        # pdhg
        if backend_name != "cpu"
            @printf(stderr, "warning: backend '%s' unavailable; using cpu\n", backend_name)
        end
        println("backend:           cpu (accelerated=no)")
        diag = PdhgDiagnostics()
        raw, ev = call_solve_pdhg(problem, pdhg_opts, backend_name, diag)
        r = finalize_result(raw, ev)
        print_result(r)
        write_solution_out(solution_out, r)
        @printf("max row violation: %.3e\n", r.max_primal_violation)
        @printf("dual residual:     %.3e\n", r.max_dual_violation)
        @printf("iterations:        %llu\n", r.iterations)
        @printf("termination:       %s\n", r.termination_reason)
        println("\ntiming (ms)")
        @printf("  total            %10.3f\n", diag.total_ms)
        print_transfer(diag.kernel_stats)
        return exit_code_for(r.status)
    catch e
        @printf(stderr, "error: %s\n", sprint(showerror, e))
        return Cint(1)
    end
end

end # module SorCLI
