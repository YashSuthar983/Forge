module SorCuts

# Port of sor/sor_search/src/cuts.cpp - Gomory mixed-integer separator.
# 1-based tableau indices: structural columns 1:n, slack of row i is n+i.

using ..SorCore: Index, kNaN
using ..SorModel: LpProblem, kInf
import ..SorModel: n_rows, n_cols, nnz
using ..SorSparse: from_triplets, to_csc
using ..SparseLU: SparseBasisFactor, LuOptions, factorize!, btran!

export CutOptions, CutDiagnostics, CutRow, separate_gomory_mi, apply_cuts
export NonbasicStatus, SimplexBasis

@enum NonbasicStatus::UInt8 begin
    Basic = 0
    AtLower
    AtUpper
    AtZeroFree
end

mutable struct SimplexBasis
    n_struct::Index
    basic::Vector{Index}                 # length m: basic variable index in 1:n+m
    status::Vector{NonbasicStatus}       # length n+m
end
SimplexBasis() = SimplexBasis(Index(0), Index[], NonbasicStatus[])

Base.@kwdef mutable struct CutOptions
    max_rounds::Int = 20
    min_progress_rel::Float64 = 1e-4
    dynamism_max::Float64 = 1e6
    violation_min::Float64 = 1e-4
    frac_min::Float64 = 1e-4
    max_cuts_per_round::Int = 200
end

Base.@kwdef mutable struct CutDiagnostics
    rounds::Int = 0
    candidates_considered::UInt64 = 0
    gmi_cuts_added::UInt64 = 0
    rejected_dynamism::UInt64 = 0
    rejected_violation::UInt64 = 0
    rejected_free_nonbasic::UInt64 = 0
    root_bound_before::Float64 = kNaN
    root_bound_after::Float64 = kNaN
end

mutable struct CutRow
    cols::Vector{Index}
    vals::Vector{Float64}
    row_lo::Float64
    row_hi::Float64
    name::String
end
CutRow() = CutRow(Index[], Float64[], -kInf, kInf, "")

const kZeroTol = 1e-11

function separate_gomory_mi(lp::LpProblem, x::Vector{Float64},
                            basis::SimplexBasis, opts::CutOptions,
                            diag::CutDiagnostics)
    m = Int(n_rows(lp))
    ns = Int(n_cols(lp))
    nt = ns + m
    if Int(basis.n_struct) != ns || length(basis.basic) != m ||
       length(basis.status) != nt || length(x) != ns
        return CutRow[]
    end

    Acsc = to_csc(lp.A)
    acp = Acsc.pattern.col_ptr
    ari = Acsc.pattern.row_idx
    acv = Acsc.vals

    bcol_ptr = zeros(Int, m + 1)
    bcol_ptr[1] = 1
    brow_idx = Int[]
    bvals = Float64[]
    sizehint!(brow_idx, Int(nnz(lp)))
    sizehint!(bvals, Int(nnz(lp)))
    for slot in 1:m
        bj = Int(basis.basic[slot])
        if 1 <= bj <= ns
            for k in Int(acp[bj]):(Int(acp[bj + 1]) - 1)
                push!(brow_idx, Int(ari[k]))
                push!(bvals, acv[k])
            end
        elseif ns < bj <= nt
            push!(brow_idx, bj - ns)
            push!(bvals, -1.0)
        else
            return CutRow[]
        end
        bcol_ptr[slot + 1] = length(brow_idx) + 1
    end

    factor = SparseBasisFactor()
    lu_opts = LuOptions()
    factored = factorize!(factor, m, bcol_ptr, brow_idx, bvals, lu_opts)
    factored && factor.valid || return CutRow[]

    is_basic = falses(nt)
    for slot in 1:m
        is_basic[Int(basis.basic[slot])] = true
    end

    lo = Vector{Float64}(undef, nt)
    hi = Vector{Float64}(undef, nt)
    for j in 1:ns
        lo[j] = lp.col_lo[j]
        hi[j] = lp.col_hi[j]
    end
    for i in 1:m
        lo[ns + i] = lp.row_lo[i]
        hi[ns + i] = lp.row_hi[i]
    end

    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    av = lp.A.vals

    cuts = CutRow[]
    for slot in 1:m
        length(cuts) >= opts.max_cuts_per_round && break
        bj = Int(basis.basic[slot])
        (1 <= bj <= ns) || continue
        (isempty(lp.is_integer) || !lp.is_integer[bj]) && continue

        beta = x[bj]
        f0 = beta - floor(beta)
        (f0 < opts.frac_min || f0 > 1.0 - opts.frac_min) && continue
        diag.candidates_considered += UInt64(1)

        y = zeros(Float64, m)
        y[slot] = 1.0
        btran!(factor, y)

        tab = zeros(Float64, nt)
        for i in 1:m
            for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
                tab[Int(ci[k])] += y[i] * av[k]
            end
        end
        for i in 1:m
            tab[ns + i] = -y[i]
        end

        struct_coef = zeros(Float64, ns)
        rhs = 1.0
        reject_free = false
        for j in 1:nt
            is_basic[j] && continue
            alpha = tab[j]
            abs(alpha) <= kZeroTol && continue
            st = basis.status[j]
            if st == AtZeroFree
                reject_free = true
                diag.rejected_free_nonbasic += UInt64(1)
                break
            end
            at_lower = (st == AtLower)
            gamma = at_lower ? alpha : -alpha
            is_int_var = (j <= ns) && !isempty(lp.is_integer) && lp.is_integer[j]

            coeff = if is_int_var
                fj = gamma - floor(gamma)
                fj <= f0 ? (fj / f0) : ((1.0 - fj) / (1.0 - f0))
            else
                gamma >= 0.0 ? (gamma / f0) : (-gamma / (1.0 - f0))
            end
            coeff <= kZeroTol && continue

            localc = at_lower ? coeff : -coeff
            if at_lower
                rhs += coeff * lo[j]
            else
                rhs -= coeff * hi[j]
            end

            if j <= ns
                struct_coef[j] += localc
            else
                srow = j - ns
                for k in Int(rp[srow]):(Int(rp[srow + 1]) - 1)
                    struct_coef[Int(ci[k])] += localc * av[k]
                end
            end
        end
        reject_free && continue

        cols = Index[]
        vals_out = Float64[]
        max_abs = 0.0
        min_abs = Inf
        for k in 1:ns
            v = struct_coef[k]
            abs(v) <= kZeroTol && continue
            push!(cols, Index(k))
            push!(vals_out, v)
            max_abs = max(max_abs, abs(v))
            min_abs = min(min_abs, abs(v))
        end
        isempty(cols) && continue
        if max_abs / max(min_abs, 1e-300) > opts.dynamism_max
            diag.rejected_dynamism += UInt64(1)
            continue
        end

        activity = 0.0
        for q in eachindex(cols)
            activity += vals_out[q] * x[Int(cols[q])]
        end
        violation = rhs - activity
        if violation < opts.violation_min
            diag.rejected_violation += UInt64(1)
            continue
        end

        name = "GMI_" * string(Int(diag.gmi_cuts_added) + length(cuts))
        push!(cuts, CutRow(cols, vals_out, rhs, kInf, name))
        diag.gmi_cuts_added += UInt64(1)
    end
    return cuts
end

function apply_cuts(lp::LpProblem, cuts::Vector{CutRow})
    isempty(cuts) && return lp
    m = Int(n_rows(lp))
    n = Int(n_cols(lp))
    rp = lp.A.pattern.row_ptr
    ci = lp.A.pattern.col_idx
    av = lp.A.vals

    rows = Index[]
    cols = Index[]
    vals = Float64[]
    sizehint!(rows, Int(nnz(lp)) + 8 * length(cuts))
    sizehint!(cols, length(rows))
    sizehint!(vals, length(rows))
    for i in 1:m
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            push!(rows, Index(i))
            push!(cols, ci[k])
            push!(vals, av[k])
        end
    end

    row_lo = copy(lp.row_lo)
    row_hi = copy(lp.row_hi)
    row_names = copy(lp.row_names)
    length(row_names) == m || resize!(row_names, m)

    for q in eachindex(cuts)
        r = Index(m + q)
        for t in eachindex(cuts[q].cols)
            push!(rows, r)
            push!(cols, cuts[q].cols[t])
            push!(vals, cuts[q].vals[t])
        end
        push!(row_lo, cuts[q].row_lo)
        push!(row_hi, cuts[q].row_hi)
        nm = isempty(cuts[q].name) ? ("CUT_" * string(q)) : cuts[q].name
        push!(row_names, nm)
    end

    A = from_triplets(m + length(cuts), n, rows, cols, vals)
    return LpProblem(lp.name, A, copy(lp.c), lp.obj_offset, lp.maximize,
                     row_lo, row_hi, copy(lp.col_lo), copy(lp.col_hi),
                     copy(lp.is_integer), row_names, copy(lp.col_names))
end

end # module
