# Bounded-variable dual revised simplex (augmented [A|-I]; Harris; Devex/DSE;
# EXPAND; BFRT). Included into SorSimplex. Port of dual_simplex.cpp (1-based).
# Vulkan/CUDA backends and Forrest–Tomlin updates are not ported.

function solve_dual_simplex(problem::LpProblem, opts::SimplexOptions,
                            diag::SimplexDiagnostics,
                            out_basis::Union{Nothing,SimplexBasis}=nothing,
                            warm::Union{Nothing,SimplexBasis}=nothing)
    t_all = time_ns()
    time_detail = opts.verbose
    t_part = UInt64(0)
    use_devex = opts.pricing !== Dantzig

    pmin = copy_lp(problem)
    sense = problem.maximize ? -1.0 : 1.0
    if pmin.maximize
        pmin.c .*= -1.0
        pmin.maximize = false
    end

    p = copy_lp(pmin)
    t_scale = time_ns()
    scaling = ruiz_scale!(p, opts.ruiz_iterations)
    diag.scaling_ms = ms_since(t_scale)

    m = Int(p.A.pattern.n_rows)
    ns = Int(p.A.pattern.n_cols)
    nt = ns + m
    use_exact_dse = (opts.pricing === DSE && m <= 64)

    ac = to_csc(p.A)
    acp = ac.pattern.col_ptr
    ari = ac.pattern.row_idx
    acv = ac.vals

    function for_col(j::Int, fn)
        if j <= ns
            for k in Int(acp[j]):(Int(acp[j + 1]) - 1)
                fn(Int(ari[k]), acv[k])
            end
        else
            fn(j - ns, -1.0)
        end
        return nothing
    end

    lo = Vector{Float64}(undef, nt)
    hi = Vector{Float64}(undef, nt)
    cost = zeros(Float64, nt)
    for j in 1:ns
        lo[j] = p.col_lo[j]
        hi[j] = p.col_hi[j]
        cost[j] = p.c[j]
    end
    for i in 1:m
        lo[ns + i] = p.row_lo[i]
        hi[ns + i] = p.row_hi[i]
    end

    colnorm2 = ones(Float64, nt)
    for j in 1:nt
        s = 1.0
        for_col(j, (i, v) -> (s += v * v; nothing))
        colnorm2[j] = s
    end

    dtol = fill(opts.dual_feas_tol, nt)
    for j in 1:ns
        dtol[j] = max(opts.dual_feas_tol * scaling.col_scale[j], 1e-12)
    end
    for i in 1:m
        dtol[ns + i] = max(opts.dual_feas_tol / scaling.row_scale[i], 1e-12)
    end

    ptol = fill(opts.primal_feas_tol, nt)
    for j in 1:ns
        ptol[j] = max(opts.primal_feas_tol / scaling.col_scale[j], 1e-12)
    end
    for i in 1:m
        ptol[ns + i] = max(opts.primal_feas_tol * scaling.row_scale[i], 1e-12)
    end

    basis = zeros(Int, m)
    slot_of = fill(0, nt)
    st = fill(AtLower, nt)
    value = zeros(Float64, nt)
    xB = zeros(Float64, m)

    function park!(j::Int)
        l = lo[j]; u = hi[j]; cj = cost[j]
        if l == u && l > -kInf
            st[j] = AtLower
            value[j] = l
            return
        end
        if l > -kInf && u < kInf
            if cj >= 0.0
                st[j] = AtLower; value[j] = l
            else
                st[j] = AtUpper; value[j] = u
            end
        elseif l > -kInf
            st[j] = AtLower; value[j] = l
        elseif u < kInf
            st[j] = AtUpper; value[j] = u
        else
            st[j] = AtZeroFree; value[j] = 0.0
        end
        return nothing
    end

    for j in 1:ns
        park!(j)
    end
    for i in 1:m
        basis[i] = ns + i
        slot_of[ns + i] = i
        st[ns + i] = Basic
    end

    warm_ok = warm !== nothing && length(warm.basic) == m && length(warm.status) == nt
    if warm_ok
        seen = fill(false, nt)
        sane = true
        for s in 1:m
            v = warm.basic[s]
            if v < 1 || v > nt || seen[v]
                sane = false
                break
            end
            seen[v] = true
        end
        if sane
            copyto!(basis, warm.basic)
            copyto!(st, warm.status)
            fill!(slot_of, 0)
            for s in 1:m
                slot_of[basis[s]] = s
            end
            for j in 1:nt
                if st[j] === Basic && slot_of[j] < 1
                    park!(j)
                elseif st[j] !== Basic && slot_of[j] >= 1
                    st[j] = Basic
                end
                if st[j] !== Basic
                    if st[j] === AtLower
                        value[j] = (lo[j] > -kInf) ? lo[j] : 0.0
                    elseif st[j] === AtUpper
                        value[j] = (hi[j] < kInf) ? hi[j] : 0.0
                    else
                        value[j] = 0.0
                    end
                end
            end
            diag.warm_starts += 1
        end
    end

    nonbasic = Int[]
    nonbasic_pos = fill(0, nt)
    sizehint!(nonbasic, nt)
    for j in 1:nt
        st[j] === Basic && continue
        nonbasic_pos[j] = length(nonbasic) + 1
        push!(nonbasic, j)
    end
    function add_nonbasic!(j::Int)
        (j < 1 || j > nt || nonbasic_pos[j] >= 1) && return
        nonbasic_pos[j] = length(nonbasic) + 1
        push!(nonbasic, j)
        return nothing
    end
    function remove_nonbasic!(j::Int)
        (j < 1 || j > nt) && return
        pos = nonbasic_pos[j]
        pos < 1 && return
        last = nonbasic[end]
        nonbasic[pos] = last
        nonbasic_pos[last] = pos
        pop!(nonbasic)
        nonbasic_pos[j] = 0
        return nothing
    end

    factor = SparseBasisFactor()
    function do_ftran!(v::Vector{Float64})
        diag.solve_calls += 1
        lu_ftran!(factor, v)
    end
    function do_btran!(v::Vector{Float64})
        diag.solve_calls += 1
        lu_btran!(factor, v)
    end
    lu_opts = LuOptions(pivot_tol=min(opts.pivot_tol, 1e-11))

    bcp = Int[1]
    bri = Int[]
    bvals = Float64[]
    bad_slots = Int[]
    vacant_rows = Int[]
    rhs = zeros(Float64, m)
    y = zeros(Float64, m)
    alpha = zeros(Float64, m)
    cB = zeros(Float64, m)
    rho = zeros(Float64, m)
    col_w = ones(Float64, nt)
    row_w = ones(Float64, m)
    cand_j = Int[]
    cand_aj = Float64[]
    cand_d = Float64[]
    bfrt_workspace = DualBfrtWorkspace()

    redcost = zeros(Float64, nt)
    prow = zeros(Float64, nt)
    prow_used = fill(false, nt)
    prow_idx = Int[]
    flip_touched = fill(false, nt)
    flipped_list = Int[]
    d_valid = false

    arp = p.A.pattern.row_ptr
    aci = p.A.pattern.col_idx
    avl = p.A.vals

    function build_pivotal_row!()
        for j in prow_idx
            prow[j] = 0.0
            prow_used[j] = false
        end
        empty!(prow_idx)
        for i in 1:m
            r = rho[i]
            r == 0.0 && continue
            for k in Int(arp[i]):(Int(arp[i + 1]) - 1)
                j = Int(aci[k])
                if !prow_used[j]
                    prow_used[j] = true
                    push!(prow_idx, j)
                end
                prow[j] += r * avl[k]
            end
            jl = ns + i
            if !prow_used[jl]
                prow_used[jl] = true
                push!(prow_idx, jl)
            end
            prow[jl] -= r
        end
        return nothing
    end

    function reset_weights!()
        for j in 1:nt
            col_w[j] = max(1.0, colnorm2[j])
        end
        fill!(row_w, 1.0)
        if use_exact_dse && !rebuild_dual_edge_weights!(m, do_btran!, row_w)
            fill!(row_w, 1.0)
        end
        return nothing
    end

    function repair_weights!()
        bad = false
        max_col = 0.0
        max_row = 0.0
        for w in col_w
            if !isfinite(w) || w <= 0.0
                bad = true
                break
            end
            max_col = max(max_col, w)
        end
        if !bad
            for w in row_w
                if !isfinite(w) || w <= 0.0
                    bad = true
                    break
                end
                max_row = max(max_row, w)
            end
        end
        if bad
            reset_weights!()
            return
        end
        max_col > 1e100 && (col_w ./= max_col)
        max_row > 1e100 && (row_w ./= max_row)
        return nothing
    end

    function build_basis_matrix!()
        empty!(bri); empty!(bvals)
        resize!(bcp, 1); bcp[1] = 1
        for s in 1:m
            for_col(basis[s], (i, v) -> (push!(bri, i); push!(bvals, v); nothing))
            push!(bcp, length(bri) + 1)
        end
        return nothing
    end

    function recompute_xB!()
        fill!(rhs, 0.0)
        for j in nonbasic
            st[j] === Basic && continue
            vj = value[j]
            vj == 0.0 && continue
            for_col(j, (i, v) -> (rhs[i] -= v * vj; nothing))
        end
        do_ftran!(rhs)
        copyto!(xB, rhs)
        return nothing
    end

    function apply_flip_shift!(flipped::Vector{Int})
        isempty(flipped) && return
        fill!(rhs, 0.0)
        for j in flipped
            delta = st[j] === AtUpper ? hi[j] - lo[j] : lo[j] - hi[j]
            delta == 0.0 && continue
            for_col(j, (i, v) -> (rhs[i] -= v * delta; nothing))
        end
        do_ftran!(rhs)
        @inbounds for i in 1:m
            xB[i] += rhs[i]
        end
        return nothing
    end

    function primal_infeasibility()
        s = 0.0
        for i in 1:m
            v = basis[i]
            if xB[i] < lo[v] - ptol[v]
                s += lo[v] - xB[i]
            elseif xB[i] > hi[v] + ptol[v]
                s += xB[i] - hi[v]
            end
        end
        return s
    end

    function exact_reduced_cost(j::Int)
        d = cost[j]
        for_col(j, (i, v) -> (d -= v * y[i]; nothing))
        return d
    end
    function rebuild_redcost!()
        for j in 1:nt
            redcost[j] = exact_reduced_cost(j)
        end
        d_valid = true
        diag.dual_rebuilds += 1
        return nothing
    end
    reduced_cost(j::Int) = redcost[j]

    function dual_infeasibility()
        s = 0.0
        for j in nonbasic
            st[j] === Basic && continue
            lo[j] == hi[j] && continue
            d = reduced_cost(j)
            sj = st[j]
            if sj === AtLower
                d < -dtol[j] && (s += -d)
            elseif sj === AtUpper
                d > dtol[j] && (s += d)
            elseif sj === AtZeroFree
                abs(d) > dtol[j] && (s += abs(d))
            end
        end
        return s
    end

    function recompute_pi!()
        for i in 1:m
            cB[i] = cost[basis[i]]
        end
        copyto!(y, cB)
        do_btran!(y)
        rebuild_redcost!()
        return nothing
    end

    phase = 1
    min_ptol = opts.primal_feas_tol
    for t in ptol
        min_ptol = min(min_ptol, t)
    end
    expand_cap = 0.5 * min(min_ptol, opts.dual_feas_tol)
    expand_start = min(opts.expand_delta, expand_cap)
    expand_active = opts.use_expand
    expand_eps = expand_start

    function do_factorize!()
        t0 = time_ns()
        repairs_before = diag.basis_repairs
        build_basis_matrix!()
        ok = lu_factorize!(factor, m, bcp, bri, bvals, lu_opts;
                           singular_slots=bad_slots, vacant_rows=vacant_rows)
        if !ok
            n = min(length(bad_slots), length(vacant_rows))
            for t in 1:n
                slot = bad_slots[t]
                newv = ns + vacant_rows[t]
                oldv = basis[slot]
                slot_of[oldv] = 0
                park!(oldv)
                add_nonbasic!(oldv)
                remove_nonbasic!(newv)
                basis[slot] = newv
                slot_of[newv] = slot
                st[newv] = Basic
                diag.basis_repairs += 1
            end
            build_basis_matrix!()
            lu_factorize!(factor, m, bcp, bri, bvals, lu_opts;
                          singular_slots=bad_slots, vacant_rows=vacant_rows)
        end
        diag.refactorizations += 1
        diag.factor_ms += ms_since(t0)
        recompute_xB!()
        recompute_pi!()
        if diag.refactorizations == 1 || diag.basis_repairs != repairs_before
            reset_weights!()
        end
        expand_eps = expand_start
        if phase == 2 && dual_infeasibility() > 0.0
            phase = 1
            diag.phase_restarts += 1
            diag.phase_restarts > 32 && (expand_active = false)
        end
        return nothing
    end

    do_factorize!()
    dual_infeasibility() <= 0.0 && (phase = 2)

    function block_t(i::Int, delta::Float64, slack::Float64)
        v = basis[i]
        l = lo[v]; u = hi[v]; x = xB[i]
        tol_v = ptol[v]
        below = (l > -kInf) && (x < l - tol_v)
        above = (u <  kInf) && (x > u + tol_v)
        if delta > 0.0
            below && return (l + slack - x) / delta
            above && return Inf
            u < kInf && return (u + slack - x) / delta
            return Inf
        end
        above && return (u - slack - x) / delta
        below && return Inf
        l > -kInf && return (l - slack - x) / delta
        return Inf
    end

    function apply_pivot!(q::Int, qdir::Int, t::Float64, leave::Int)
        xp_before = leave < 1 ? 0.0 : xB[leave]
        @inbounds for i in 1:m
            xB[i] -= Float64(qdir) * t * alpha[i]
        end
        if leave < 1
            if st[q] === AtLower
                st[q] = AtUpper; value[q] = hi[q]
            else
                st[q] = AtLower; value[q] = lo[q]
            end
            diag.bound_flips += 1
            return true
        end
        vl = basis[leave]
        remove_nonbasic!(q)
        add_nonbasic!(vl)
        lv = lo[vl]; uv = hi[vl]
        delta_p = -Float64(qdir) * alpha[leave]
        tol_v = ptol[vl]
        p_below = (lv > -kInf) && (xp_before < lv - tol_v)
        p_above = (uv <  kInf) && (xp_before > uv + tol_v)
        vl_st = if delta_p > 0.0
            p_below ? AtLower : AtUpper
        else
            p_above ? AtUpper : AtLower
        end
        lv == uv && (vl_st = AtLower)
        q_from = st[q] === AtLower ? lo[q] : (st[q] === AtUpper ? hi[q] : 0.0)
        st[vl] = vl_st
        value[vl] = vl_st === AtLower ? lv : uv
        slot_of[vl] = 0
        basis[leave] = q
        slot_of[q] = leave
        st[q] = Basic
        xB[leave] = q_from + Float64(qdir) * t
        if use_devex
            ap = alpha[leave]
            ap2 = max(ap * ap, 1e-30)
            wq = col_w[q]
            leaving_col_w = wq / ap2
            col_w[vl] = isfinite(leaving_col_w) ? max(1.0, leaving_col_w) : 1.0
            wr = row_w[leave]
            for i in 1:m
                i == leave && continue
                r = alpha[i] / ap
                candidate = r * r * wr
                isfinite(candidate) && (row_w[i] = max(row_w[i], candidate))
            end
            leaving_w = wr / ap2
            row_w[leave] = isfinite(leaving_w) ? max(1.0, leaving_w) : 1.0
        end
        return false
    end

    since_refactor = 0
    function maybe_update_factor!(leave::Int)
        leave < 1 && return
        ap = leave <= length(alpha) ? abs(alpha[leave]) : 0.0
        mult = ap > 0.0 ? 1.0 / ap : Inf
        diag.largest_update_multiplier = max(diag.largest_update_multiplier, mult)
        unstable = opts.refactor_multiplier_limit > 0.0 &&
                   mult > opts.refactor_multiplier_limit
        eta_full = lu_needs_refactor(factor, opts.refactor_interval,
                                     opts.refactor_eta_ratio,
                                     opts.bump_width_max,
                                     opts.refactor_work_ratio)
        updated = lu_update!(factor, leave, alpha, opts.pivot_tol)
        since_refactor += 1
        if unstable || !updated || eta_full || since_refactor >= opts.refactor_interval
            do_factorize!()
            since_refactor = 0
        end
        return nothing
    end

    iter = UInt64(0)
    max_iter = opts.max_iterations != 0 ? opts.max_iterations :
        UInt64(max(10000, 20 * (m + nt)))
    status = NotSolved
    reason = ""
    farkas_ray = Float64[]
    farkas_ray_violation = kPosInf

    best_merit = Inf
    flat_checks = 0
    merit_phase = -1
    merit_at_start = primal_infeasibility()
    merit_low = merit_at_start
    diag.merit_start = merit_at_start
    diag.merit_best = merit_at_start
    kMeritEvery = 64
    kFlatLimit = 32

    t_loop = time_ns()
    while true
        if iter >= max_iter
            status = Interrupted
            reason = "iteration limit ($max_iter)"
            break
        end
        if diag.basis_repairs > opts.max_basis_repairs
            status = NumericalFailure
            reason = "basis went singular $(diag.basis_repairs) times; refusing to continue on a degraded factorization"
            break
        end
        if opts.time_limit_s > 0.0 && (iter % 64) == 0 &&
           (time_ns() - t_all) / 1e9 > opts.time_limit_s
            status = Interrupted
            reason = "time limit ($(opts.time_limit_s)s)"
            break
        end
        (iter & 127) == 0 && repair_weights!()
        if opts.stall_abort && iter > 0 && (iter % kMeritEvery) == 0
            if phase != merit_phase
                merit_phase = phase
                best_merit = Inf
                flat_checks = 0
            end
            merit = phase == 2 ? primal_infeasibility() : dual_infeasibility()
            if phase == 2 && merit < merit_low
                merit_low = merit
                diag.merit_best = merit
            end
            if merit < best_merit * (1.0 - 1e-9)
                best_merit = merit
                flat_checks = 0
            elseif (flat_checks += 1) >= kFlatLimit
                status = Interrupted
                diag.stalled = true
                reason = "dual stalled: phase $phase merit flat for $(kFlatLimit * kMeritEvery) iterations"
                break
            end
        end

        t_step = 0.0
        q = 0
        qdir = 0
        leave = 0
        d_enter = 0.0
        theta_dual = 0.0
        used_bfrt = false

        diag.pricing_calls += 1
        if phase == 1
            flipped = false
            time_detail && (t_part = time_ns())
            best = 0.0
            for j in nonbasic
                lo[j] == hi[j] && continue
                d = reduced_cost(j)
                boxed = lo[j] > -kInf && hi[j] < kInf
                if boxed
                    if st[j] === AtLower && d < -dtol[j]
                        st[j] = AtUpper
                        value[j] = hi[j]
                        diag.bound_flips += 1
                        flip_touched[j] = true
                        flipped = true
                        continue
                    end
                    if st[j] === AtUpper && d > dtol[j]
                        st[j] = AtLower
                        value[j] = lo[j]
                        diag.bound_flips += 1
                        flip_touched[j] = true
                        flipped = true
                        continue
                    end
                    continue
                end
                dir = 0
                viol = 0.0
                sj = st[j]
                if sj === AtLower
                    if d < -dtol[j]
                        dir = 1; viol = -d
                    end
                elseif sj === AtUpper
                    if d > dtol[j]
                        dir = -1; viol = d
                    end
                elseif sj === AtZeroFree
                    if abs(d) > dtol[j]
                        dir = d < 0.0 ? 1 : -1
                        viol = abs(d)
                    end
                end
                dir == 0 && continue
                flipped && continue
                den = (use_devex && isfinite(col_w[j])) ? col_w[j] : 1.0
                score = viol * viol / max(den, 1e-30)
                if score > best
                    best = score; q = j; qdir = dir; d_enter = d
                end
            end
            time_detail && (diag.price_ms += ms_since(t_part))
            if flipped
                empty!(flipped_list)
                for j in nonbasic
                    if flip_touched[j]
                        push!(flipped_list, j)
                        flip_touched[j] = false
                    end
                end
                apply_flip_shift!(flipped_list)
                iter += 1
                diag.phase1_iterations += 1
                continue
            end

            if q < 1
                if since_refactor > 0
                    do_factorize!()
                    since_refactor = 0
                    continue
                end
                if dual_infeasibility() > 0.0
                    status = NumericalFailure
                    reason = "phase 1 pricing/full-scan disagreement on dual infeasibility"
                    break
                end
                phase = 2
                continue
            end

            fill!(alpha, 0.0)
            for_col(q, (i, v) -> (alpha[i] += v; nothing))
            time_detail && (t_part = time_ns())
            do_ftran!(alpha)
            time_detail && (diag.solve_ms += ms_since(t_part))

            p1_slack = opts.harris_slack + (expand_active ? expand_eps : 0.0)
            t_bound = Inf
            if st[q] === AtLower && hi[q] < kInf
                t_bound = hi[q] - lo[q]
            elseif st[q] === AtUpper && lo[q] > -kInf
                t_bound = hi[q] - lo[q]
            end

            t_max = t_bound
            for i in 1:m
                a = alpha[i]
                abs(a) <= opts.pivot_tol && continue
                tb = block_t(i, -Float64(qdir) * a, p1_slack)
                tb < t_max && (t_max = tb)
            end
            t_max < 0.0 && (t_max = 0.0)

            best_piv = 0.0
            t_step = 0.0
            leave = 0
            for i in 1:m
                a = alpha[i]
                mag = abs(a)
                mag <= opts.pivot_tol && continue
                te = block_t(i, -Float64(qdir) * a, 0.0)
                (!(te < Inf) || te > t_max) && continue
                if mag > best_piv
                    best_piv = mag
                    leave = i
                    t_step = max(0.0, te)
                end
            end

            if leave < 1 && !(t_bound < Inf)
                if since_refactor > 0
                    do_factorize!()
                    since_refactor = 0
                    continue
                end
                if primal_infeasibility() <= opts.primal_feas_tol
                    status = Unbounded
                    reason = "dual phase 1: unblocked improving column at a primal-feasible basis"
                    break
                end
                status = NumericalFailure
                reason = "dual phase 1 stalled: unblocked improving column at a primal-infeasible basis"
                break
            end
            t_step = leave < 1 ? t_bound : t_step

            if leave >= 1
                fill!(rho, 0.0)
                rho[leave] = 1.0
                do_btran!(rho)
                build_pivotal_row!()
                if use_devex
                    ap = alpha[leave]
                    ap2 = max(ap * ap, 1e-30)
                    wq = col_w[q]
                    for j in prow_idx
                        st[j] === Basic && continue
                        aj = prow[j]
                        col_w[j] = max(1.0, col_w[j], (aj * aj / ap2) * wq)
                    end
                end
            end
        else
            time_detail && (t_part = time_ns())

            function row_viol(i::Int)
                v = basis[i]
                if xB[i] < lo[v] - ptol[v]
                    return (true, lo[v] - xB[i], true)
                end
                if xB[i] > hi[v] + ptol[v]
                    return (true, xB[i] - hi[v], false)
                end
                return (false, 0.0, true)
            end

            best = 0.0
            leave_to_lower = true
            function consider_row!(i::Int)
                hit, viol, to_lo = row_viol(i)
                hit || return
                den = (use_devex && isfinite(row_w[i])) ? row_w[i] : 1.0
                score = viol * viol / max(den, 1e-30)
                if score > best
                    best = score
                    leave = i
                    leave_to_lower = to_lo
                end
                return nothing
            end
            for i in 1:m
                consider_row!(i)
            end
            time_detail && (diag.price_ms += ms_since(t_part))

            if leave < 1
                if since_refactor > 0
                    do_factorize!()
                    since_refactor = 0
                    continue
                end
                if dual_infeasibility() > 0.0
                    phase = 1
                    diag.phase_restarts += 1
                    diag.phase_restarts > 32 && (expand_active = false)
                    continue
                end
                status = Optimal
                reason = "no primal-infeasible basic variable"
                break
            end

            fill!(rho, 0.0)
            rho[leave] = 1.0
            time_detail && (t_part = time_ns())
            do_btran!(rho)
            time_detail && (diag.solve_ms += ms_since(t_part))

            dual_slack = opts.harris_slack + (expand_active ? expand_eps : 0.0)
            srow = leave_to_lower ? 1.0 : -1.0

            time_detail && (t_part = time_ns())
            build_pivotal_row!()
            empty!(cand_j); empty!(cand_aj); empty!(cand_d)
            for j in prow_idx
                st[j] === Basic && continue
                lo[j] == hi[j] && continue
                aj = prow[j]
                sa = srow * aj
                elig = false
                sj = st[j]
                if sj === AtLower
                    elig = sa < -opts.pivot_tol
                elseif sj === AtUpper
                    elig = sa > opts.pivot_tol
                elseif sj === AtZeroFree
                    elig = abs(sa) > opts.pivot_tol
                end
                elig || continue
                push!(cand_j, j)
                push!(cand_aj, aj)
                push!(cand_d, redcost[j])
            end
            time_detail && (diag.price_ms += ms_since(t_part))

            vl = basis[leave]
            target = leave_to_lower ? lo[vl] : hi[vl]
            delta_primal = xB[leave] - target
            harris_theta = dual_harris_theta(cand_j, cand_aj, cand_d, st, srow, dual_slack)

            choice = dual_legacy_ratio(cand_j, cand_aj, cand_d, st, srow,
                                       harris_theta, opts.pivot_tol)
            bfrt = dual_bfrt_choose(cand_j, cand_aj, cand_d, st, lo, hi,
                                    delta_primal, srow, harris_theta,
                                    opts.pivot_tol, opts.dual_feas_tol,
                                    dual_slack, bfrt_workspace)
            if bfrt.ok && bfrt.pivot >= 1 && bfrt.pivot_dir != 0 &&
               bfrt.theta_dual != 0.0 && !isempty(bfrt.flips)
                choice = bfrt
            end
            if !choice.ok || choice.pivot < 1
                choice = dual_legacy_ratio(cand_j, cand_aj, cand_d, st, srow,
                                           harris_theta, opts.pivot_tol)
            end

            if !choice.ok || choice.pivot < 1
                if since_refactor > 0
                    do_factorize!()
                    since_refactor = 0
                    continue
                end
                status = Infeasible
                reason = "primal-infeasible basic with no dual-feasible entering column"
                farkas_ray = zeros(Float64, m)
                for i in 1:m
                    farkas_ray[i] = srow * rho[i] * scaling.row_scale[i]
                end
                farkas_ray_violation = farkas_violation(pmin, farkas_ray)
                break
            end

            q = choice.pivot
            qdir = choice.pivot_dir
            d_enter = choice.d_enter
            used_bfrt = choice.theta_dual != 0.0
            theta_dual = choice.theta_dual

            if used_bfrt && !isempty(choice.flips)
                for fj in choice.flips
                    flip_nonbasic!(st, value, fj, lo[fj], hi[fj])
                    diag.bound_flips += 1
                end
                apply_flip_shift!(choice.flips)
            end

            fill!(alpha, 0.0)
            for_col(q, (i, v) -> (alpha[i] += v; nothing))
            time_detail && (t_part = time_ns())
            do_ftran!(alpha)
            time_detail && (diag.solve_ms += ms_since(t_part))

            if used_bfrt
                for j in prow_idx
                    st[j] === Basic && continue
                    redcost[j] -= theta_dual * prow[j]
                end
                for i in 1:m
                    y[i] += theta_dual * rho[i]
                end
                d_enter = redcost[q]
            end

            ap = alpha[leave]
            if abs(ap) <= opts.pivot_tol
                if since_refactor > 0
                    do_factorize!()
                    since_refactor = 0
                    continue
                end
                status = NumericalFailure
                reason = "dual pivot element vanished after FTRAN"
                break
            end

            denom = -Float64(qdir) * ap
            t_step = abs(denom) <= opts.pivot_tol ? 0.0 : (target - xB[leave]) / denom
            t_step < 0.0 && (t_step = 0.0)
        end

        leave_var = leave >= 1 ? basis[leave] : 0
        was_flip = apply_pivot!(q, qdir, t_step, leave)
        use_exact_dse && !was_flip && reset_weights!()
        if !was_flip
            arq = prow[q]
            apiv = alpha[leave]
            row_ok = abs(arq) > opts.pivot_tol &&
                     abs(arq - apiv) <= 1e-6 * (1.0 + abs(apiv))
            if used_bfrt && row_ok && leave_var >= 1
                redcost[q] = 0.0
                redcost[leave_var] = -theta_dual
            elseif row_ok && d_valid && leave_var >= 1
                theta = d_enter / arq
                for i in 1:m
                    y[i] += theta * rho[i]
                end
                for j in prow_idx
                    st[j] === Basic && continue
                    redcost[j] -= theta * prow[j]
                end
                redcost[q] = 0.0
                redcost[leave_var] = -theta
            else
                recompute_pi!()
                diag.dual_resyncs += 1
            end
            maybe_update_factor!(leave)
        end

        if t_step <= 1e-12
            diag.degenerate_steps += 1
            if expand_active
                expand_eps = min(expand_cap, expand_eps * max(opts.expand_factor, 1.0))
                diag.expand_steps += 1
            end
        end

        iter += 1
        if phase == 1
            diag.phase1_iterations += 1
        else
            diag.phase2_iterations += 1
        end

        if opts.verbose && (iter % 500) == 0
            recompute_pi!()
            obj = 0.0
            for i in 1:m
                obj += cost[basis[i]] * xB[i]
            end
            for j in 1:nt
                st[j] !== Basic && (obj += cost[j] * value[j])
            end
            println("  iter $iter  phase $phase  dual-infeas $(dual_infeasibility())  prim-infeas $(primal_infeasibility())  obj $(sense * obj + problem.obj_offset)")
        end
    end
    diag.loop_ms = ms_since(t_loop)

    diag.iterations = iter
    diag.final_phase = phase
    diag.status = status
    diag.basis_dimension = m
    diag.factor_nnz = factor.stats.factor_nnz
    diag.largest_multiplier = factor.stats.largest_multiplier

    x = zeros(Float64, ns)
    for j in 1:ns
        x[j] = st[j] === Basic ? xB[slot_of[j]] : value[j]
        x[j] *= scaling.col_scale[j]
    end

    for i in 1:m
        cB[i] = cost[basis[i]]
    end
    copyto!(y, cB)
    do_btran!(y)
    yout = Vector{Float64}(undef, m)
    for i in 1:m
        yout[i] = y[i] * scaling.row_scale[i]
    end

    diag.primal_residual = max(max_row_violation(pmin, x), max_bound_violation(pmin, x))

    aty = zeros(Float64, ns)
    ax = zeros(Float64, m)
    rp = pmin.A.pattern.row_ptr
    ci = pmin.A.pattern.col_idx
    for i in 1:m
        s = 0.0
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            j = Int(ci[k])
            v = pmin.A.vals[k]
            aty[j] += v * yout[i]
            s += v * x[j]
        end
        ax[i] = s
    end

    dres = 0.0
    at_tol = max(kAtBound, opts.primal_feas_tol)
    function accum_dual!(v, l, u, d)
        at_lo = (l > -kInf) && (v <= l + at_tol * (1.0 + abs(l)))
        at_hi = (u <  kInf) && (v >= u - at_tol * (1.0 + abs(u)))
        at_lo && at_hi && return
        if at_lo
            dres = max(dres, max(0.0, -d))
        elseif at_hi
            dres = max(dres, max(0.0, d))
        else
            dres = max(dres, abs(d))
        end
    end
    for j in 1:ns
        accum_dual!(x[j], pmin.col_lo[j], pmin.col_hi[j], pmin.c[j] - aty[j])
    end
    for i in 1:m
        accum_dual!(ax[i], pmin.row_lo[i], pmin.row_hi[i], yout[i])
    end
    diag.dual_residual = dres

    obj_min = 0.0
    for j in 1:ns
        obj_min += pmin.c[j] * x[j]
    end
    diag.primal_objective = sense * obj_min + problem.obj_offset

    finite = true
    dval = 0.0
    for j in 1:ns
        d = pmin.c[j] - aty[j]
        b = d >= 0.0 ? pmin.col_lo[j] : pmin.col_hi[j]
        if isinf(b)
            if abs(d) > opts.dual_feas_tol
                finite = false
                break
            end
            continue
        end
        dval += mul_zero_safe(d, b)
    end
    if finite
        for i in 1:m
            yi = yout[i]
            b = yi >= 0.0 ? pmin.row_lo[i] : pmin.row_hi[i]
            if isinf(b)
                if abs(yi) > opts.dual_feas_tol
                    finite = false
                    break
                end
                continue
            end
            dval += mul_zero_safe(yi, b)
        end
    end
    diag.dual_bound_finite = finite && isfinite(dval)
    diag.dual_objective = diag.dual_bound_finite ? sense * dval + problem.obj_offset : NaN
    diag.gap_rel = diag.dual_bound_finite ?
        abs(diag.primal_objective - diag.dual_objective) / (1.0 + abs(diag.primal_objective)) :
        Inf

    if out_basis !== nothing
        out_basis.n_struct = ns
        out_basis.basic = copy(basis)
        out_basis.status = copy(st)
    end

    raw = RawResult()
    raw.x = x
    raw.y = sense .* yout
    if status === Infeasible && !isempty(farkas_ray)
        raw.ray = farkas_ray
    end
    raw.objective = diag.primal_objective
    raw.dual_bound = diag.dual_objective
    raw.iterations = iter
    raw.engine = "simplex_dual"
    raw.backend = "cpu"
    raw.proposed_status = status
    raw.termination_reason = reason
    diag.ray_violation = farkas_ray_violation
    if status === Optimal
        raw.proposed_level = ProvedOptimalFP
    elseif status === Infeasible || status === Unbounded
        raw.proposed_level = BoundOnly
    elseif status === Interrupted
        raw.proposed_level = diag.primal_residual <= opts.primal_feas_tol ? FeasibleOnly : None
    else
        raw.proposed_level = None
    end
    diag.total_ms = ms_since(t_all)
    return raw
end
