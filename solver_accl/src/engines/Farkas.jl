module SorFarkas

# Port of Farkas infeasibility certificate check (Chvátal 1983, Ch.8). 1-based CSR.

using ..SorCore: kPosInf
using ..SorSparse: n_rows, n_cols
using ..SorModel: LpProblem

export farkas_violation

function farkas_violation(lp::LpProblem, y::Vector{Float64})
    m = Int(n_rows(lp.A))
    n = Int(n_cols(lp.A))
    length(y) != m && return kPosInf

    d = zeros(n)
    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    av = lp.A.vals
    @inbounds for i in 1:m
        yi = y[i]
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            d[Int(ci[k])] += yi * av[k]
        end
    end

    kZeroTol = 1e-11
    L = 0.0
    @inbounds for j in 1:n
        dj = d[j]
        if dj > kZeroTol
            isfinite(lp.col_lo[j]) || return kPosInf
            L += dj * lp.col_lo[j]
        elseif dj < -kZeroTol
            isfinite(lp.col_hi[j]) || return kPosInf
            L += dj * lp.col_hi[j]
        end
    end

    U = 0.0
    @inbounds for i in 1:m
        yi = y[i]
        if yi > kZeroTol
            isfinite(lp.row_hi[i]) || return kPosInf
            U += yi * lp.row_hi[i]
        elseif yi < -kZeroTol
            isfinite(lp.row_lo[i]) || return kPosInf
            U += yi * lp.row_lo[i]
        end
    end

    return max(0.0, U - L)
end

end # module
