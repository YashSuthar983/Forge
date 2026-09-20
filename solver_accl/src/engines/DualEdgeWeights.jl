# Exact dual steepest-edge weights: w_i = ||B^{-T} e_i||_2^2.
# Included into SorSimplex. Unused by the dual stub; kept for a later dual port.

function rebuild_dual_edge_weights!(m::Int, btran!, weights::Vector{Float64})
    m < 0 && return false
    resize!(weights, m)
    fill!(weights, 1.0)
    rhs = zeros(Float64, m)
    for i in 1:m
        fill!(rhs, 0.0)
        rhs[i] = 1.0
        btran!(rhs)
        norm2 = 0.0
        for v in rhs
            isfinite(v) || return false
            norm2 += v * v
            isfinite(norm2) || return false
        end
        weights[i] = max(norm2, 1e-300)
    end
    return true
end
