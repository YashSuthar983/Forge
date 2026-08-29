# CSR sparse matrix-vector kernels, written once for every backend.
#
# Algorithm: one work item per row, sequential accumulation within the row.
# From the standard CSR SpMV formulation; no solver source was consulted.
#
# DETERMINISM (commitment C3): spmv_t is NOT implemented with atomic scatter.
# Atomic float accumulation has a nondeterministic summation order, so results
# would vary run to run. Instead the transpose pattern is built ONCE at upload
# time and A'x becomes an ordinary row-wise SpMV. This costs memory and buys
# bit-reproducibility, which is the right trade for this project.

@kernel function spmv_kernel!(y, @Const(row_ptr), @Const(col_idx), @Const(vals),
                              @Const(x))
    r = @index(Global, Linear)
    acc = zero(eltype(y))
    @inbounds for k in row_ptr[r]:(row_ptr[r + 1] - 1)
        acc += vals[k] * x[col_idx[k]]
    end
    @inbounds y[r] = acc
end

"""
A CSR pattern resident on the device, plus its transpose.

`row_ptr` / `col_idx` are 1-BASED here: the C++ side sends 0-based CSR and
`upload_pattern` shifts them once, rather than paying for the shift per call.
"""
struct DevicePattern
    n_rows::Int
    n_cols::Int
    nnz::Int
    row_ptr::Any        # length n_rows+1
    col_idx::Any        # length nnz
    # transpose (CSC of A == CSR of A')
    t_row_ptr::Any      # length n_cols+1
    t_col_idx::Any
    t_perm::Any         # t_perm[k] = index into the original vals array
end

"""
    build_pattern(dev, n_rows, n_cols, row_ptr0, col_idx0) -> (DevicePattern, ms, bytes)

`row_ptr0` and `col_idx0` are 0-based, as sent by C++.
"""
function build_pattern(dev::DeviceState, n_rows::Int, n_cols::Int,
                       row_ptr0::Vector{Int}, col_idx0::Vector{Int})
    nnz = length(col_idx0)
    length(row_ptr0) == n_rows + 1 ||
        error("row_ptr must have n_rows+1 entries, got $(length(row_ptr0))")
    row_ptr0[end] == nnz ||
        error("row_ptr[end] ($(row_ptr0[end])) must equal nnz ($nnz)")

    rp = row_ptr0 .+ 1
    ci = col_idx0 .+ 1
    any(c -> c < 1 || c > n_cols, ci) && error("column index out of range")

    # Transpose by counting sort on columns. Within each column, entries come out
    # ordered by original row, so the accumulation order is fixed.
    counts = zeros(Int, n_cols + 1)
    @inbounds for c in ci
        counts[c + 1] += 1
    end
    t_rp = cumsum(counts) .+ 1          # 1-based offsets, length n_cols+1
    cursor = copy(t_rp)
    t_ci = Vector{Int}(undef, nnz)
    t_perm = Vector{Int}(undef, nnz)
    @inbounds for r in 1:n_rows
        for k in rp[r]:(rp[r + 1] - 1)
            c = ci[k]
            dst = cursor[c]
            cursor[c] += 1
            t_ci[dst] = r
            t_perm[dst] = k
        end
    end

    total_bytes = 0
    d_rp,    ms1, b1 = to_device(dev, rp)
    d_ci,    ms2, b2 = to_device(dev, ci)
    d_trp,   ms3, b3 = to_device(dev, t_rp)
    d_tci,   ms4, b4 = to_device(dev, t_ci)
    d_tperm, ms5, b5 = to_device(dev, t_perm)
    total_bytes = b1 + b2 + b3 + b4 + b5

    p = DevicePattern(n_rows, n_cols, nnz, d_rp, d_ci, d_trp, d_tci, d_tperm)
    return p, (ms1 + ms2 + ms3 + ms4 + ms5), total_bytes
end

"""
    spmv!(dev, y, pat, vals, x)

y := A x. `vals` is in the ORIGINAL CSR order.
"""
function spmv!(dev::DeviceState, y, pat::DevicePattern, vals, x)
    kern = spmv_kernel!(dev.backend)
    kern(y, pat.row_ptr, pat.col_idx, vals, x; ndrange = pat.n_rows)
    KernelAbstractions.synchronize(dev.backend)
    return y
end

@kernel function gather_kernel!(dst, @Const(src), @Const(perm))
    i = @index(Global, Linear)
    @inbounds dst[i] = src[perm[i]]
end

"""
    spmv_t!(dev, y, pat, vals, x, tvals_scratch)

y := A' x, via a row-wise SpMV on the precomputed transpose. `tvals_scratch`
must be an nnz-length device array; the values are gathered into transpose order
each call because `vals` can change between calls while the pattern does not.
"""
function spmv_t!(dev::DeviceState, y, pat::DevicePattern, vals, x, tvals)
    g = gather_kernel!(dev.backend)
    g(tvals, vals, pat.t_perm; ndrange = pat.nnz)
    kern = spmv_kernel!(dev.backend)
    kern(y, pat.t_row_ptr, pat.t_col_idx, tvals, x; ndrange = pat.n_cols)
    KernelAbstractions.synchronize(dev.backend)
    return y
end
