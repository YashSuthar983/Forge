module SorQpAuto

# Evidence-based dispatcher across this repo's three QP engines --
# SorQp's diagonal-Q/equality-only fast path, SorQpActiveSet (general,
# proven robust), and SorQpIpm (general, faster when it works but known
# to diverge on rank-deficient Q -- docs/AGENDA.md Agenda 11). Mirrors
# SorSimplex's own Auto mode (try Dual, fall back to Primal on anything
# but success) rather than inventing a new pattern -- same idea here:
# try the faster engine when the problem's shape makes that safe, always
# fall back to the slower-but-more-robust one otherwise, so this
# dispatcher is never less correct than always calling the safe engine
# directly, only sometimes faster.
#
# The routing thresholds below are not guessed -- they come directly from
# the Maros-Meszaros real-library sweep (docs/AGENDA.md Agenda 11,
# bench/run_qp_full_bench.jl, 2026-09-06). Two signals cleanly separated
# every IPM failure from every IPM success in that sweep:
#   - q/n (inequality-row count over variable count): every DUALC*
#     instance IPM failed on has q/n roughly 24-63 (few variables, many
#     constraints -- the "dual" shape the DUALC* name describes); every
#     instance IPM solved has q/n well under 1 (QPCBLEND, the largest,
#     is 0.89).
#   - fraction of Q's diagonal that's near-zero relative to its max: 29
#     of QAFIRO's 32 diagonal entries are exactly zero (fraction ~0.91)
#     and IPM diverges on it even though q/n there (51/32 ~ 1.6) doesn't
#     look extreme by itself -- rank-deficiency is a separate, additional
#     signal from q/n, not a substitute for it.
# Neither threshold is tight against the data (5x and 30% respectively,
# against gaps that were actually 1.6x-vs-24x and 0-vs-91%), deliberately
# leaving margin rather than fitting the exact boundary of one sweep.

using ..SorCore: RawResult, Status
using ..SorCore: Optimal
using ..SorModel: LpProblem
using ..SorSparse: n_cols
using ..SorQpActiveSet: QpProblemG, QpOptionsG, QpDiagnosticsG
using ..SorQpActiveSet: solve_qp_activeset_auto, _build_constraints
using ..SorQpIpm: QpOptionsIpm, QpDiagnosticsIpm, solve_qp_ipm

export solve_qp_auto

const Q_OVER_N_IPM_LIMIT = 5.0
const RANK_DEFICIENT_FRACTION_LIMIT = 0.3

"""
    solve_qp_auto(problem::QpProblemG, opts_as::QpOptionsG, opts_ip::QpOptionsIpm,
                  diag::QpDiagnosticsG) -> RawResult

Picks between `SorQpIpm` (faster when it works) and
`SorQpActiveSet`'s ridge-escalation ladder (slower, more robust) based on
two cheap, pre-solve signals measured to correlate with `SorQpIpm`'s
known failure mode. Always falls back to the active-set engine if IPM is
skipped or doesn't reach `Optimal`, so this function is never less
correct than calling `solve_qp_activeset_auto` directly -- only
sometimes faster. `diag` is filled in from whichever engine actually
ran; a second, discarded `QpDiagnosticsIpm` is used internally when IPM
is attempted and abandoned.
"""
function solve_qp_auto(problem::QpProblemG, opts_as::QpOptionsG, opts_ip::QpOptionsIpm,
                       diag::QpDiagnosticsG)
    lp = problem.linear
    n = Int(n_cols(lp.A))
    Q = problem.Q

    maxdiag = 0.0
    @inbounds for j in 1:n
        maxdiag = max(maxdiag, abs(Q[j, j]))
    end
    near_zero = 0
    if maxdiag > 0.0
        thresh = 1e-8 * maxdiag
        @inbounds for j in 1:n
            abs(Q[j, j]) < thresh && (near_zero += 1)
        end
    end
    rank_deficient_frac = n > 0 ? near_zero / n : 0.0

    _, beq, _, bineq = _build_constraints(lp, n)
    q = length(bineq)
    q_over_n = n > 0 ? q / n : 0.0

    try_ipm = rank_deficient_frac <= RANK_DEFICIENT_FRACTION_LIMIT &&
              q_over_n <= Q_OVER_N_IPM_LIMIT

    if try_ipm
        ip_diag = QpDiagnosticsIpm()
        raw = solve_qp_ipm(problem, opts_ip, ip_diag)
        if raw.proposed_status == Optimal
            diag.iterations = ip_diag.iterations
            diag.primal_residual = ip_diag.primal_residual
            diag.stationarity = ip_diag.dual_residual
            diag.objective = ip_diag.objective
            diag.total_ms = ip_diag.total_ms
            diag.termination_reason = "ipm: " * ip_diag.termination_reason
            return raw
        end
    end

    raw = solve_qp_activeset_auto(problem, opts_as, diag)
    diag.termination_reason = (try_ipm ? "ipm failed, fell back to active-set: " : "active-set (ipm skipped): ") *
                              diag.termination_reason
    return raw
end

end # module
