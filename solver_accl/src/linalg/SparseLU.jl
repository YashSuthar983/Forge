
module SparseLU

using SparseArrays

export LuOptions, LuStats, SparseBasisFactor, factorize!, ftran!, btran!, update!, needs_refactor

Base.@kwdef struct LuOptions
    markowitz_threshold::Float64 = 0.1
    pivot_tol::Float64 = 1e-11
    max_search_cols::Int = 8
end

mutable struct LuStats
    dimension::Int
    input_nnz::Int
    factor_nnz::Int
    triangular_pivots::Int
    nucleus_pivots::Int
    singular_count::Int
    largest_multiplier::Float64
end
LuStats() = LuStats(0, 0, 0, 0, 0, 0, 0.0)

mutable struct SparseBasisFactor
    m::Int
    valid::Bool

    piv_row::Vector{Int}
    piv_slot::Vector{Int}
    piv_val::Vector{Float64}
    rpos::Vector{Int}
    cpos::Vector{Int}

    u_off::Vector{Int}
    u_len::Vector{Int}
    u_idx::Vector{Int}
    u_val::Vector{Float64}

    l_start::Vector{Int}
    l_idx::Vector{Int}
    l_val::Vector{Float64}

    u_col_start::Vector{Int}
    u_col_row::Vector{Int}
    l_col_start::Vector{Int}
    l_col_row::Vector{Int}

    eta_p::Vector{Int}
    eta_start::Vector{Int}
    eta_idx::Vector{Int}
    eta_val::Vector{Float64}
    eta_pivot::Vector{Float64}

    u_live_nnz::Int
    u_alloc_nnz::Int

    work::Vector{Float64}
    mark::Vector{Bool}
    dfs_stack::Vector{Int}
    reach::Vector{Int}
    order::Vector{Int}
    seed::Vector{Int}
    dense_below::Int
    stats::LuStats

    function SparseBasisFactor()
        new(0, false, 
            Int[], Int[], Float64[], Int[], Int[],
            Int[], Int[], Int[], Float64[],
            Int[1], Int[], Float64[],
            Int[], Int[], Int[], Int[],
            Int[], Int[1], Int[], Float64[], Float64[],
            0, 0,
            Float64[], Bool[], Int[], Int[], Int[], Int[], 0, LuStats())
    end
end

# Helper struct for Elimination
mutable struct Elim
    m::Int
    row_cols::Vector{Vector{Int}}
    row_vals::Vector{Vector{Float64}}
    col_rows::Vector{Vector{Int}}
    col_cnt::Vector{Int}
    row_live::Vector{Bool}
    col_live::Vector{Bool}

    q_col1::Vector{Int}
    q_row1::Vector{Int}
    bucket::Vector{Vector{Int}}

    tmp_cols::Vector{Int}
    pr_cols::Vector{Int}
    tmp_vals::Vector{Float64}
    pr_vals::Vector{Float64}
end

function Elim(m::Int)
    row_cols = [Int[] for _ in 1:m]
    row_vals = [Float64[] for _ in 1:m]
    col_rows = [Int[] for _ in 1:m]
    bucket = [Int[] for _ in 1:(m+2)]
    Elim(m, row_cols, row_vals, col_rows, zeros(Int, m), trues(m), trues(m),
         Int[], Int[], bucket, Int[], Int[], Float64[], Float64[])
end

function note_col!(e::Elim, j::Int)
    k = e.col_cnt[j]
    if k == 1
        push!(e.q_col1, j)
    end
    if k >= 0 && k < length(e.bucket)
        push!(e.bucket[k+1], j) # +1 for 1-based indexing of bucket size
    end
end

function note_row!(e::Elim, i::Int)
    if length(e.row_cols[i]) == 1
        push!(e.q_row1, i)
    end
end

function find_val(e::Elim, i::Int, j::Int)
    rc = e.row_cols[i]
    idx = searchsortedfirst(rc, j)
    if idx <= length(rc) && rc[idx] == j
        return e.row_vals[i][idx]
    end
    return nothing
end

function compact_col!(e::Elim, j::Int)
    cr = e.col_rows[j]
    sort!(cr)
    unique!(cr)
    w = 1
    amax = 0.0
    for k in 1:length(cr)
        i = cr[k]
        if !e.row_live[i]
            continue
        end
        v = find_val(e, i, j)
        if v === nothing
            continue
        end
        cr[w] = i
        w += 1
        amax = max(amax, abs(v))
    end
    resize!(cr, w - 1)
    e.col_cnt[j] = w - 1
    return amax
end

function factorize!(bf::SparseBasisFactor, m::Int, col_ptr::Vector{Int}, row_idx::Vector{Int}, vals::Vector{Float64}, opts::LuOptions; singular_slots=nothing, vacant_rows=nothing)
    bf.m = m
    bf.valid = false
    bf.stats = LuStats()
    bf.stats.dimension = m
    bf.stats.input_nnz = length(row_idx)

    empty!(bf.piv_row); empty!(bf.piv_slot); empty!(bf.piv_val)
    empty!(bf.u_off); empty!(bf.u_len); empty!(bf.u_idx); empty!(bf.u_val)
    bf.l_start = [1]; empty!(bf.l_idx); empty!(bf.l_val)
    empty!(bf.u_col_start); empty!(bf.u_col_row)
    empty!(bf.l_col_start); empty!(bf.l_col_row)
    empty!(bf.eta_p); bf.eta_start = [1]
    empty!(bf.eta_idx); empty!(bf.eta_val); empty!(bf.eta_pivot)
    
    bf.work = zeros(Float64, m)
    empty!(bf.mark); empty!(bf.dfs_stack); empty!(bf.reach); empty!(bf.order); empty!(bf.seed)
    bf.dense_below = 0
    bf.u_live_nnz = 0
    bf.u_alloc_nnz = 0
    
    if singular_slots !== nothing empty!(singular_slots) end
    if vacant_rows !== nothing empty!(vacant_rows) end

    if m == 0
        bf.valid = true
        return true
    end

    e = Elim(m)
    for j in 1:m
        for k in col_ptr[j]:(col_ptr[j+1]-1)
            i = row_idx[k]
            v = vals[k]
            if v == 0.0 continue end
            push!(e.row_cols[i], j)
            push!(e.row_vals[i], v)
            push!(e.col_rows[j], i)
            e.col_cnt[j] += 1
        end
    end

    for j in 1:m note_col!(e, j) end
    for i in 1:m note_row!(e, i) end

    function eliminate(r::Int, c::Int, piv::Float64)
        empty!(e.pr_cols); empty!(e.pr_vals)
        rc = e.row_cols[r]
        rv = e.row_vals[r]
        for k in 1:length(rc)
            if rc[k] == c continue end
            push!(e.pr_cols, rc[k])
            push!(e.pr_vals, rv[k])
        end

        push!(bf.piv_row, r)
        push!(bf.piv_slot, c)
        push!(bf.piv_val, piv)

        for k in 1:length(e.pr_cols)
            push!(bf.u_idx, e.pr_cols[k])
            push!(bf.u_val, e.pr_vals[k])
        end
        push!(bf.u_off, bf.u_alloc_nnz) # 0-based offset internally might be tricky, let's keep it 1-based logic but wait, u_alloc_nnz starts at 0? Let's use 1-based indexing.
        push!(bf.u_len, length(e.pr_cols))
        bf.u_alloc_nnz += length(e.pr_cols) # wait, u_off should be 1-based. Let's fix this below.
        
        targets = copy(e.col_rows[c])
        for i in targets
            if i == r || !e.row_live[i] continue end
            aic = find_val(e, i, c)
            if aic === nothing continue end
            mult = aic / piv
            bf.stats.largest_multiplier = max(bf.stats.largest_multiplier, abs(mult))
            push!(bf.l_idx, i)
            push!(bf.l_val, mult)

            empty!(e.tmp_cols); empty!(e.tmp_vals)
            ric = e.row_cols[i]
            riv = e.row_vals[i]
            a = 1; b = 1
            while a <= length(ric) || b <= length(e.pr_cols)
                if a <= length(ric) && ric[a] == c
                    a += 1; continue
                end
                ja = a <= length(ric) ? ric[a] : typemax(Int)
                jb = b <= length(e.pr_cols) ? e.pr_cols[b] : typemax(Int)
                
                if ja < jb
                    push!(e.tmp_cols, ja); push!(e.tmp_vals, riv[a]); a += 1
                elseif jb < ja
                    v = -mult * e.pr_vals[b]
                    if v != 0.0
                        push!(e.tmp_cols, jb); push!(e.tmp_vals, v)
                        e.col_cnt[jb] += 1
                        push!(e.col_rows[jb], i)
                        note_col!(e, jb)
                    end
                    b += 1
                else
                    v = riv[a] - mult * e.pr_vals[b]
                    if v != 0.0
                        push!(e.tmp_cols, ja); push!(e.tmp_vals, v)
                    else
                        e.col_cnt[ja] -= 1
                        note_col!(e, ja)
                    end
                    a += 1; b += 1
                end
            end
            e.col_cnt[c] -= 1
            e.row_cols[i] = copy(e.tmp_cols)
            e.row_vals[i] = copy(e.tmp_vals)
            note_row!(e, i)
        end
        push!(bf.l_start, length(bf.l_idx) + 1) # 1-based start

        for j in e.row_cols[r]
            e.col_cnt[j] -= 1
            if j != c note_col!(e, j) end
        end
        e.row_live[r] = false
        e.col_live[c] = false
        empty!(e.row_cols[r])
        empty!(e.row_vals[r])
        empty!(e.col_rows[c])
        e.col_cnt[c] = 0
    end

    bf.u_alloc_nnz = 1 # 1-based
    
    function peel()
        moved = false
        while true
            did = false
            while !isempty(e.q_col1)
                c = pop!(e.q_col1)
                if !e.col_live[c] continue end
                amax = compact_col!(e, c)
                if e.col_cnt[c] != 1
                    note_col!(e, c); continue
                end
                if !(amax > opts.pivot_tol) continue end
                r = e.col_rows[c][1]
                v = find_val(e, r, c)
                if v === nothing continue end
                eliminate(r, c, v)
                bf.stats.triangular_pivots += 1
                did = true; moved = true
            end
            while !isempty(e.q_row1)
                r = pop!(e.q_row1)
                if !e.row_live[r] || length(e.row_cols[r]) != 1 continue end
                c = e.row_cols[r][1]
                v = e.row_vals[r][1]
                if !(abs(v) > opts.pivot_tol) continue end
                eliminate(r, c, v)
                bf.stats.triangular_pivots += 1
                did = true; moved = true
            end
            if !did break end
        end
        return moved
    end

    function markowitz()
        best_cost = typemax(Int)
        best_mag = 0.0
        out_r = -1; out_c = -1; out_v = 0.0
        examined = 0
        stop = false

        for k in 2:length(e.bucket)
            if stop break end
            bk = e.bucket[k]
            w = 1; t = 1
            while t <= length(bk)
                c = bk[t]
                if !e.col_live[c] || e.col_cnt[c] != (k - 1)
                    t += 1; continue
                end
                bk[w] = c
                w += 1
                amax = compact_col!(e, c)
                if e.col_cnt[c] != (k - 1)
                    w -= 1; note_col!(e, c); t += 1; continue
                end
                if !(amax > opts.pivot_tol)
                    t += 1; continue
                end
                floor_mag = opts.markowitz_threshold * amax
                for i in e.col_rows[c]
                    v_ptr = find_val(e, i, c)
                    if v_ptr === nothing continue end
                    v = v_ptr
                    mag = abs(v)
                    if !(mag >= floor_mag) || !(mag > opts.pivot_tol) continue end
                    rc = length(e.row_cols[i])
                    cost = (rc - 1) * (k - 2)
                    if cost < best_cost || (cost == best_cost && mag > best_mag)
                        best_cost = cost
                        best_mag = mag
                        out_r = i; out_c = c; out_v = v
                    end
                end
                examined += 1
                if best_cost == 0 || (examined >= opts.max_search_cols && out_c != -1)
                    stop = true
                    t += 1
                    break
                end
                t += 1
            end
            for u in t:length(bk)
                bk[w] = bk[u]
                w += 1
            end
            resize!(bk, w - 1)
        end
        return out_r, out_c, out_v
    end

    peel()
    while true
        r, c, v = markowitz()
        if c == -1 break end
        eliminate(r, c, v)
        bf.stats.nucleus_pivots += 1
        peel()
    end

    n_piv = length(bf.piv_row)
    if n_piv < m
        free_rows = Int[]
        free_cols = Int[]
        for i in 1:m if e.row_live[i] push!(free_rows, i) end end
        for j in 1:m if e.col_live[j] push!(free_cols, j) end end
        bf.stats.singular_count = length(free_cols)
        if singular_slots !== nothing append!(singular_slots, free_cols) end
        if vacant_rows !== nothing append!(vacant_rows, free_rows) end

        n_sing = min(length(free_rows), length(free_cols))
        for t in 1:n_sing
            push!(bf.piv_row, free_rows[t])
            push!(bf.piv_slot, free_cols[t])
            push!(bf.piv_val, 1.0)
            push!(bf.u_off, bf.u_alloc_nnz)
            push!(bf.u_len, 0)
            push!(bf.l_start, length(bf.l_idx) + 1)
        end
    end

    bf.rpos = fill(-1, m)
    bf.cpos = fill(-1, m)
    for k in 1:length(bf.piv_row)
        bf.rpos[bf.piv_row[k]] = k
        bf.cpos[bf.piv_slot[k]] = k
    end
    for k in 1:length(bf.u_idx) bf.u_idx[k] = bf.cpos[bf.u_idx[k]] end
    for k in 1:length(bf.l_idx) bf.l_idx[k] = bf.rpos[bf.l_idx[k]] end

    build_col_patterns!(bf)
    bf.u_live_nnz = bf.u_alloc_nnz
    bf.mark = fill(false, length(bf.piv_val))
    empty!(bf.dfs_stack); empty!(bf.reach); empty!(bf.order)
    bf.dense_below = length(bf.piv_val) ÷ 2

    bf.stats.factor_nnz = length(bf.u_idx) + length(bf.l_idx) + length(bf.piv_val)
    bf.valid = (n_piv == m)
    return bf.valid
end

function build_col_patterns!(bf::SparseBasisFactor)
    n = length(bf.piv_val)
    # 0-based prefix counts, then +1 - same conversion as SorSparse.to_csc.
    bf.u_col_start = zeros(Int, n + 1)
    bf.u_col_row = fill(0, length(bf.u_idx))
    bf.l_col_start = zeros(Int, n + 1)
    bf.l_col_row = fill(0, length(bf.l_idx))

    for t in 1:length(bf.u_idx)
        bf.u_col_start[bf.u_idx[t] + 1] += 1
    end
    for t in 1:length(bf.l_idx)
        bf.l_col_start[bf.l_idx[t] + 1] += 1
    end
    for j in 1:n
        bf.u_col_start[j + 1] += bf.u_col_start[j]
        bf.l_col_start[j + 1] += bf.l_col_start[j]
    end
    bf.u_col_start .+= 1
    bf.l_col_start .+= 1

    u_cursor = copy(bf.u_col_start[1:end-1])
    l_cursor = copy(bf.l_col_start[1:end-1])
    
    for k in 1:n
        beg = bf.u_off[k]; end_ = beg + bf.u_len[k]
        for t in beg:(end_-1)
            idx = bf.u_idx[t]
            bf.u_col_row[u_cursor[idx]] = k
            u_cursor[idx] += 1
        end
        beg = bf.l_start[k]; end_ = bf.l_start[k + 1]
        for t in beg:(end_-1)
            idx = bf.l_idx[t]
            bf.l_col_row[l_cursor[idx]] = k
            l_cursor[idx] += 1
        end
    end
end

function solve_lower(bf::SparseBasisFactor, v::Vector{Float64})
    n = length(bf.piv_val)
    for k in 1:n
        zk = v[k]
        if zk == 0.0 continue end
        beg = bf.l_start[k]; end_ = bf.l_start[k + 1]
        for t in beg:(end_-1)
            v[bf.l_idx[t]] -= bf.l_val[t] * zk
        end
    end
end

function solve_upper(bf::SparseBasisFactor, v::Vector{Float64})
    n = length(bf.piv_val)
    for k in n:-1:1
        s = v[k]
        beg = bf.u_off[k]; end_ = beg + bf.u_len[k]
        for t in beg:(end_-1)
            s -= bf.u_val[t] * v[bf.u_idx[t]]
        end
        v[k] = s / bf.piv_val[k]
    end
end

function solve_upper_t(bf::SparseBasisFactor, v::Vector{Float64})
    n = length(bf.piv_val)
    for k in 1:n
        zk = v[k] / bf.piv_val[k]
        v[k] = zk
        if zk == 0.0 continue end
        beg = bf.u_off[k]; end_ = beg + bf.u_len[k]
        for t in beg:(end_-1)
            v[bf.u_idx[t]] -= bf.u_val[t] * zk
        end
    end
end

function solve_lower_t(bf::SparseBasisFactor, v::Vector{Float64})
    n = length(bf.piv_val)
    for k in n:-1:1
        s = v[k]
        beg = bf.l_start[k]; end_ = bf.l_start[k + 1]
        for t in beg:(end_-1)
            s -= bf.l_val[t] * v[bf.l_idx[t]]
        end
        v[k] = s
    end
end

function reach_dfs!(bf::SparseBasisFactor, seed::Vector{Int}, adj_func::Function)
    for j in seed
        if !bf.mark[j]
            bf.mark[j] = true
            push!(bf.dfs_stack, j)
        end
    end
    empty!(bf.reach)
    while !isempty(bf.dfs_stack)
        k = pop!(bf.dfs_stack)
        adj_func(k)
        push!(bf.reach, k)
        if length(bf.reach) + length(bf.dfs_stack) > bf.dense_below
            for j in bf.reach bf.mark[j] = false end
            while !isempty(bf.dfs_stack)
                bf.mark[pop!(bf.dfs_stack)] = false
            end
            for j in seed bf.mark[j] = false end
            empty!(bf.reach)
            return false
        end
    end
    return true
end

function sparse_lower(bf::SparseBasisFactor, seed::Vector{Int}, v::Vector{Float64})
    success = reach_dfs!(bf, seed, k -> begin
        beg = bf.l_start[k]; end_ = bf.l_start[k + 1]
        for t in beg:(end_-1)
            i = bf.l_idx[t]
            if !bf.mark[i]
                bf.mark[i] = true
                push!(bf.dfs_stack, i)
            end
        end
    end)
    if !success return false end
    
    bf.order = copy(bf.reach)
    sort!(bf.order)
    for k in bf.order
        zk = v[k]
        if zk == 0.0 continue end
        beg = bf.l_start[k]; end_ = bf.l_start[k + 1]
        for t in beg:(end_-1)
            i = bf.l_idx[t]
            if bf.mark[i]
                v[i] -= bf.l_val[t] * zk
            end
        end
    end
    return true
end

function sparse_upper(bf::SparseBasisFactor, seed::Vector{Int}, v::Vector{Float64})
    success = reach_dfs!(bf, seed, k -> begin
        beg = bf.u_col_start[k]; end_ = bf.u_col_start[k + 1]
        for t in beg:(end_-1)
            i = bf.u_col_row[t]
            if !bf.mark[i]
                bf.mark[i] = true
                push!(bf.dfs_stack, i)
            end
        end
    end)
    if !success return false end
    
    bf.order = copy(bf.reach)
    sort!(bf.order)
    for i in length(bf.order):-1:1
        k = bf.order[i]
        s = v[k]
        beg = bf.u_off[k]; end_ = beg + bf.u_len[k]
        for t in beg:(end_-1)
            j = bf.u_idx[t]
            if bf.mark[j]
                s -= bf.u_val[t] * v[j]
            end
        end
        v[k] = s / bf.piv_val[k]
    end
    return true
end

function sparse_upper_t(bf::SparseBasisFactor, seed::Vector{Int}, v::Vector{Float64})
    success = reach_dfs!(bf, seed, k -> begin
        beg = bf.u_off[k]; end_ = beg + bf.u_len[k]
        for t in beg:(end_-1)
            i = bf.u_idx[t]
            if !bf.mark[i]
                bf.mark[i] = true
                push!(bf.dfs_stack, i)
            end
        end
    end)
    if !success return false end

    bf.order = copy(bf.reach)
    sort!(bf.order)
    for k in bf.order
        zk = v[k] / bf.piv_val[k]
        v[k] = zk
        if zk == 0.0 continue end
        beg = bf.u_off[k]; end_ = beg + bf.u_len[k]
        for t in beg:(end_-1)
            j = bf.u_idx[t]
            if bf.mark[j]
                v[j] -= bf.u_val[t] * zk
            end
        end
    end
    return true
end

function sparse_lower_t(bf::SparseBasisFactor, seed::Vector{Int}, v::Vector{Float64})
    success = reach_dfs!(bf, seed, k -> begin
        beg = bf.l_col_start[k]; end_ = bf.l_col_start[k + 1]
        for t in beg:(end_-1)
            i = bf.l_col_row[t]
            if !bf.mark[i]
                bf.mark[i] = true
                push!(bf.dfs_stack, i)
            end
        end
    end)
    if !success return false end

    bf.order = copy(bf.reach)
    sort!(bf.order)
    for i in length(bf.order):-1:1
        k = bf.order[i]
        s = v[k]
        beg = bf.l_start[k]; end_ = bf.l_start[k + 1]
        for t in beg:(end_-1)
            j = bf.l_idx[t]
            if bf.mark[j]
                s -= bf.l_val[t] * v[j]
            end
        end
        v[k] = s
    end
    return true
end

function ftran!(bf::SparseBasisFactor, b::Vector{Float64})
    if bf.m == 0 return end
    n = length(bf.piv_val)
    for k in 1:n bf.work[k] = b[bf.piv_row[k]] end

    sp = false
    if n >= 64
        nz = 0
        for k in 1:n if bf.work[k] != 0.0 nz += 1 end end
        if 4 * nz <= n
            empty!(bf.seed)
            for k in 1:n if bf.work[k] != 0.0 push!(bf.seed, k) end end
            if sparse_lower(bf, bf.seed, bf.work)
                sp = true
                for k in bf.reach bf.mark[k] = false end
            end
        end
    end
    if !sp solve_lower(bf, bf.work) end

    sp = false
    if n >= 64
        nz = 0
        for k in 1:n if bf.work[k] != 0.0 nz += 1 end end
        if 4 * nz <= n
            empty!(bf.seed)
            for k in 1:n if bf.work[k] != 0.0 push!(bf.seed, k) end end
            if sparse_upper(bf, bf.seed, bf.work)
                sp = true # marks stay set for the scatter
            end
        end
    end
    if sp
        for k in 1:n b[bf.piv_slot[k]] = bf.mark[k] ? bf.work[k] : 0.0 end
        for k in bf.reach bf.mark[k] = false end
    else
        solve_upper(bf, bf.work)
        for k in 1:n b[bf.piv_slot[k]] = bf.work[k] end
    end

    for t in 1:length(bf.eta_p)
        p = bf.eta_p[t]
        if b[p] == 0.0 continue end
        pv = b[p] / bf.eta_pivot[t]
        beg = bf.eta_start[t]; end_ = bf.eta_start[t + 1]
        for k in beg:(end_-1)
            i = bf.eta_idx[k]
            if i != p b[i] -= bf.eta_val[k] * pv end
        end
        b[p] = pv
    end
end

function btran!(bf::SparseBasisFactor, d::Vector{Float64})
    if bf.m == 0 return end

    for t in length(bf.eta_p):-1:1
        p = bf.eta_p[t]
        s = d[p]
        beg = bf.eta_start[t]; end_ = bf.eta_start[t + 1]
        for k in beg:(end_-1)
            i = bf.eta_idx[k]
            if i != p s -= bf.eta_val[k] * d[i] end
        end
        d[p] = s / bf.eta_pivot[t]
    end

    n = length(bf.piv_val)
    for k in 1:n bf.work[k] = d[bf.piv_slot[k]] end

    sp = false
    if n >= 64
        nz = 0
        for k in 1:n if bf.work[k] != 0.0 nz += 1 end end
        if 4 * nz <= n
            empty!(bf.seed)
            for k in 1:n if bf.work[k] != 0.0 push!(bf.seed, k) end end
            if sparse_upper_t(bf, bf.seed, bf.work)
                sp = true
                for k in bf.reach bf.mark[k] = false end
            end
        end
    end
    if !sp solve_upper_t(bf, bf.work) end

    sp = false
    if n >= 64
        nz = 0
        for k in 1:n if bf.work[k] != 0.0 nz += 1 end end
        if 4 * nz <= n
            empty!(bf.seed)
            for k in 1:n if bf.work[k] != 0.0 push!(bf.seed, k) end end
            if sparse_lower_t(bf, bf.seed, bf.work)
                sp = true
            end
        end
    end
    if sp
        for k in 1:n d[bf.piv_row[k]] = bf.mark[k] ? bf.work[k] : 0.0 end
        for k in bf.reach bf.mark[k] = false end
    else
        solve_lower_t(bf, bf.work)
        for k in 1:n d[bf.piv_row[k]] = bf.work[k] end
    end
end

function update!(bf::SparseBasisFactor, p::Int, alpha::Vector{Float64}, min_pivot::Float64 = 1e-11)
    ap = p <= length(alpha) ? alpha[p] : 0.0
    if !(abs(ap) > min_pivot) return false end
    push!(bf.eta_p, p)
    push!(bf.eta_pivot, ap)
    for i in 1:bf.m
        if alpha[i] != 0.0 && i != p
            push!(bf.eta_idx, i)
            push!(bf.eta_val, alpha[i])
        end
    end
    push!(bf.eta_start, length(bf.eta_idx) + 1)
    return true
end

function needs_refactor(bf::SparseBasisFactor, update_limit::Int, eta_nnz_ratio::Float64)
    if update_limit > 0 && length(bf.eta_p) >= update_limit
        return true
    end
    if eta_nnz_ratio > 0.0 && bf.stats.factor_nnz > 0
        limit = eta_nnz_ratio * bf.stats.factor_nnz
        if length(bf.eta_val) > limit return true end
    end
    return false
end

end # module
