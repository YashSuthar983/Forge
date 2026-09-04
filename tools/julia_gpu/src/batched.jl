# Batched kernels: ONE shared sparsity pattern, N right-hand sides.
#
# This is the capability the whole prototype exists to explore
# (sor/docs/architecture.md §3 — batched LpDevice seam). The LPs a
# solver actually spends its time on -- strong-branching children, diving
# iterations, LNS relaxations, decomposition subproblems, scenario instances --
# differ from one another only in their bound vectors, so they share a pattern
# exactly. One kernel launch covers the whole batch.
#
# Layout: X is n_items * n_cols, contiguous, item-major (item 0 first).
# Matches sor::backend::BatchView.

@kernel function spmv_batched_kernel!(Y, @Const(row_ptr), @Const(col_idx),
                                     @Const(vals), @Const(X),
                                     n_rows::Int, n_cols::Int)
    idx = @index(Global, Linear)
    @inbounds begin
        b = (idx - 1) ÷ n_rows          # 0-based batch item
        r = (idx - 1) % n_rows + 1      # 1-based row
        xoff = b * n_cols
        acc = zero(eltype(Y))
        for k in row_ptr[r]:(row_ptr[r + 1] - 1)
            acc += vals[k] * X[xoff + col_idx[k]]
        end
        Y[b * n_rows + r] = acc
    end
end

"""
    spmv_batched!(dev, Y, pat, vals, X, n_items)

Y[b] := A * X[b] for every b, in a single launch.
"""
function spmv_batched!(dev::DeviceState, Y, pat::DevicePattern, vals, X,
                       n_items::Int)
    length(X) == n_items * pat.n_cols ||
        error("spmv_batched: |X| ($(length(X))) != n_items*n_cols " *
              "($(n_items * pat.n_cols))")
    length(Y) == n_items * pat.n_rows ||
        error("spmv_batched: |Y| ($(length(Y))) != n_items*n_rows " *
              "($(n_items * pat.n_rows))")
    n_items == 0 && return Y
    kern = spmv_batched_kernel!(dev.backend)
    kern(Y, pat.row_ptr, pat.col_idx, vals, X, pat.n_rows, pat.n_cols;
         ndrange = n_items * pat.n_rows)
    KernelAbstractions.synchronize(dev.backend)
    return Y
end

"""
    project_box_batched!(dev, X, LO, HI)

Per-item boxes: LO and HI have the same shape as X. Reuses the elementwise
kernel, since the batch structure does not change the operation -- the point is
that each item can have DIFFERENT bounds, which is exactly how a strong-branching
child differs from its parent.
"""
function project_box_batched!(dev::DeviceState, X, LO, HI)
    length(X) == length(LO) == length(HI) ||
        error("project_box_batched: mismatched shapes")
    return project_box!(dev, X, LO, HI)
end
