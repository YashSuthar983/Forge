# Glue over SparseLU for the C++ BasisFactor surface used by revised simplex.
# Included into SorSimplex (which already `using ..SparseLU`). Product-form
# updates only; Forrest–Tomlin / work-since-factor triggers are no-ops until
# SparseLU grows those fields.

lu_dimension(bf::SparseBasisFactor) = bf.m
lu_n_updates(bf::SparseBasisFactor) = length(bf.eta_p)
lu_eta_nnz(bf::SparseBasisFactor) = length(bf.eta_val)
lu_is_valid(bf::SparseBasisFactor) = bf.valid
lu_stats(bf::SparseBasisFactor) = bf.stats
lu_work_since_factor(::SparseBasisFactor) = 0
lu_current_bump_width(::SparseBasisFactor) = 0

function lu_factorize!(bf::SparseBasisFactor, m::Int,
                       col_ptr::AbstractVector{<:Integer},
                       row_idx::AbstractVector{<:Integer},
                       vals::Vector{Float64}, opts::LuOptions;
                       singular_slots=nothing, vacant_rows=nothing)
    # Copy: SparseLU's factorize! reuses/rewrites index buffers; the simplex
    # driver keeps live CSC scratch (bcp/bri/bvals) across refactorizations.
    cp = Vector{Int}(col_ptr)
    ri = Vector{Int}(row_idx)
    vv = copy(vals)
    return factorize!(bf, m, cp, ri, vv, opts;
                      singular_slots=singular_slots, vacant_rows=vacant_rows)
end

lu_ftran!(bf::SparseBasisFactor, b::Vector{Float64}) = ftran!(bf, b)
lu_btran!(bf::SparseBasisFactor, d::Vector{Float64}) = btran!(bf, d)

function lu_update!(bf::SparseBasisFactor, p::Int, alpha::Vector{Float64},
                    min_pivot::Float64=1e-11)
    return update!(bf, p, alpha, min_pivot)
end

function lu_needs_refactor(bf::SparseBasisFactor, update_limit::Int,
                           eta_nnz_ratio::Float64,
                           bump_width_max::Int=0,
                           work_ratio_max::Float64=0.0)
    needs_refactor(bf, update_limit, eta_nnz_ratio) && return true
    if bump_width_max > 0 && lu_current_bump_width(bf) > bump_width_max
        return true
    end
    if work_ratio_max > 0.0 && bf.stats.factor_nnz > 0
        if lu_work_since_factor(bf) > work_ratio_max * bf.stats.factor_nnz
            return true
        end
    end
    return false
end
