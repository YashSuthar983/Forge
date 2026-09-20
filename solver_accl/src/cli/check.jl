module SorCheck

# Independent checker (SIH26119 checklist item 30). Port of sor/cli/sor_check.cpp.
# Re-verifies a claim from `sor_solve --solution-out` against the original model
# using only row/bound violations and farkas_violation. No solver internals.

using Printf
using ..SorCore: Optimal, Feasible, Infeasible, to_string
using ..SorModel: max_row_violation, max_bound_violation, objective
import ..SorModel: n_rows, n_cols, nnz
using ..SorIO: read_mps_file_auto, MpsReadReport, read_solution
using ..SorFarkas: farkas_violation

export sor_check_main

function usage()
    print(stderr,
        "usage: sor_check MODEL.mps SOLUTION.sol [--tol T]\n",
        "  SOLUTION.sol is written by `sor_solve ... --solution-out FILE`.\n",
        "  Exit 0: the claim independently verifies. Exit 1: it does not.\n")
end

function fail(what, residual, tol)
    @printf("FAIL  %-28s residual=%.3e  tol=%.3e\n", what, residual, tol)
    return false
end

function pass(what, residual, tol)
    @printf("pass  %-28s residual=%.3e  tol=%.3e\n", what, residual, tol)
    return true
end

function sor_check_main(args)::Cint
    if length(args) < 2
        usage()
        return Cint(2)
    end
    model_path = ""
    solution_path = ""
    tol = 1e-7
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--tol"
            i + 1 > length(args) && (usage(); return Cint(2))
            i += 1
            tol = parse(Float64, args[i])
        elseif a == "-h" || a == "--help"
            usage()
            return Cint(0)
        elseif isempty(model_path)
            model_path = a
        elseif isempty(solution_path)
            solution_path = a
        else
            usage()
            return Cint(2)
        end
        i += 1
    end
    if isempty(model_path) || isempty(solution_path)
        usage()
        return Cint(2)
    end

    try
        rep = MpsReadReport()
        lp = read_mps_file_auto(model_path, rep)
        for w in rep.warnings
            println(stderr, "warning: ", w)
        end
        isfile(solution_path) || begin
            @printf(stderr, "error: cannot open '%s'\n", solution_path)
            return Cint(2)
        end
        sol = open(read_solution, solution_path)

        @printf("model:     %s  (%d rows x %d cols, nnz %lld)\n",
                model_path, Int(n_rows(lp)), Int(n_cols(lp)), Int(nnz(lp)))
        @printf("claimed:   status=%s proof=%s objective=%.10e\n",
                to_string(sol.status), to_string(sol.proof), sol.objective)

        ok = true
        if sol.status == Optimal || sol.status == Feasible
            if length(sol.x) != Int(n_cols(lp))
                @printf("FAIL  x has %d entries, model has %d columns\n",
                        length(sol.x), Int(n_cols(lp)))
                ok = false
            else
                row_viol = max_row_violation(lp, sol.x)
                bound_viol = max_bound_violation(lp, sol.x)
                ok &= (row_viol <= tol) ? pass("row bounds", row_viol, tol) :
                                          fail("row bounds", row_viol, tol)
                ok &= (bound_viol <= tol) ? pass("column bounds", bound_viol, tol) :
                                            fail("column bounds", bound_viol, tol)
                true_obj = objective(lp, sol.x)
                obj_err = abs(true_obj - sol.objective) / (1.0 + abs(true_obj))
                ok &= (obj_err <= tol) ? pass("objective (recomputed)", obj_err, tol) :
                                         fail("objective (recomputed)", obj_err, tol)
            end
        elseif sol.status == Infeasible
            if isempty(sol.ray)
                println("no certificate: status=Infeasible but no ray was recorded ",
                        "(honest gap, not a failure -- nothing to verify)")
            else
                v = farkas_violation(lp, sol.ray)
                if !isfinite(v)
                    ok = fail("farkas certificate", v, tol)
                else
                    ok &= (v <= tol) ? pass("farkas certificate", v, tol) :
                                       fail("farkas certificate", v, tol)
                end
            end
        else
            @printf("no independent check implemented for status=%s (honest gap, not a failure)\n",
                    to_string(sol.status))
        end
        println(ok ? "VERIFIED" : "REJECTED")
        return ok ? Cint(0) : Cint(1)
    catch e
        @printf(stderr, "error: %s\n", sprint(showerror, e))
        return Cint(2)
    end
end

end # module
