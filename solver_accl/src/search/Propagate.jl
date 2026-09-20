module SorPropagate

# Port of sor/sor_search/src/propagate.cpp (Achterberg thesis 2007, Ch.10.4).
# 1-based indices. Tightens col_lo/col_hi in place.

using ..SorModel: LpProblem, kInf
import ..SorModel: n_rows, n_cols

export PropagateResult, propagate_bounds

Base.@kwdef mutable struct PropagateResult
    feasible::Bool = true
    tightened::UInt64 = 0
    rounds::Int = 0
end

function propagate_bounds(lp::LpProblem, col_lo::Vector{Float64}, col_hi::Vector{Float64};
                          tol::Float64=1e-9, max_rounds::Int=10)
    res = PropagateResult()
    m = Int(n_rows(lp))
    n = Int(n_cols(lp))
    length(col_lo) == n && length(col_hi) == n || return res

    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    av = lp.A.vals

    for rcount in 0:(max_rounds - 1)
        res.rounds = rcount
        changed_this_round = false

        for i in 1:m
            row_lo = lp.row_lo[i]
            row_hi = lp.row_hi[i]
            isfinite(row_lo) || isfinite(row_hi) || continue

            beg = Int(rp[i])
            endk = Int(rp[i + 1]) - 1

            for kk in beg:endk
                j = Int(ci[kk])
                aj = av[kk]
                aj == 0.0 && continue

                rmin = 0.0
                rmax = 0.0
                rmin_finite = true
                rmax_finite = true
                for t in beg:endk
                    t == kk && continue
                    q = Int(ci[t])
                    a = av[t]
                    a == 0.0 && continue
                    lo = col_lo[q]
                    hi = col_hi[q]
                    if a > 0.0
                        if rmin_finite
                            if isfinite(lo); rmin += a * lo; else; rmin_finite = false; end
                        end
                        if rmax_finite
                            if isfinite(hi); rmax += a * hi; else; rmax_finite = false; end
                        end
                    else
                        if rmin_finite
                            if isfinite(hi); rmin += a * hi; else; rmin_finite = false; end
                        end
                        if rmax_finite
                            if isfinite(lo); rmax += a * lo; else; rmax_finite = false; end
                        end
                    end
                end

                new_lo = -kInf
                new_hi = kInf
                if aj > 0.0
                    isfinite(row_hi) && rmin_finite && (new_hi = (row_hi - rmin) / aj)
                    isfinite(row_lo) && rmax_finite && (new_lo = (row_lo - rmax) / aj)
                else
                    isfinite(row_hi) && rmin_finite && (new_lo = (row_hi - rmin) / aj)
                    isfinite(row_lo) && rmax_finite && (new_hi = (row_lo - rmax) / aj)
                end

                if !isempty(lp.is_integer) && lp.is_integer[j]
                    new_lo > -kInf && (new_lo = ceil(new_lo - tol))
                    new_hi < kInf && (new_hi = floor(new_hi + tol))
                end

                if new_lo > col_lo[j] + tol
                    col_lo[j] = new_lo
                    res.tightened += UInt64(1)
                    changed_this_round = true
                end
                if new_hi < col_hi[j] - tol
                    col_hi[j] = new_hi
                    res.tightened += UInt64(1)
                    changed_this_round = true
                end
                if col_lo[j] > col_hi[j] + tol
                    res.feasible = false
                    res.rounds = rcount + 1
                    return res
                end
            end
        end

        res.rounds = rcount + 1
        changed_this_round || break
    end
    return res
end

end # module
