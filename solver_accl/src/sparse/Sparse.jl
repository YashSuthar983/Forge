module SorSparse

# Port of sor/sor_sparse (C++ L1). Pattern is separate from values so batched
# kernels can share one index structure. Julia uses 1-based CSR/CSC pointers
# (row_ptr[1] == 1, row_ptr[end] == nnz+1), the native equivalent of the
# C++ 0-based layout.

using ..SorCore: Index, Offset

export SparsePattern, CsrMatrix, from_triplets
export CscPattern, CscMatrix, to_csc
export n_rows, n_cols, nnz, validate

struct SparsePattern
    n_rows::Index
    n_cols::Index
    row_ptr::Vector{Offset}
    col_idx::Vector{Index}

    function SparsePattern(n_rows::Integer, n_cols::Integer,
                           row_ptr::Vector{Offset}, col_idx::Vector{Index})
        validate(new(Index(n_rows), Index(n_cols), row_ptr, col_idx))
    end
end

function validate(p::SparsePattern)
    if p.n_rows < 0 || p.n_cols < 0
        throw(ArgumentError("SparsePattern: negative dimension"))
    end
    if length(p.row_ptr) != Int(p.n_rows) + 1
        throw(ArgumentError("SparsePattern: row_ptr must have n_rows+1 entries"))
    end
    if p.row_ptr[1] != 1
        throw(ArgumentError("SparsePattern: row_ptr[1] must be 1"))
    end
    nz = length(p.col_idx)
    if p.row_ptr[end] != Offset(nz + 1)
        throw(ArgumentError("SparsePattern: row_ptr[end] must equal nnz+1"))
    end
    for r in 1:Int(p.n_rows)
        if p.row_ptr[r] > p.row_ptr[r + 1]
            throw(ArgumentError("SparsePattern: row_ptr not nondecreasing"))
        end
    end
    for c in p.col_idx
        if c < 1 || c > p.n_cols
            throw(ArgumentError("SparsePattern: column index out of range"))
        end
    end
    return p
end

nnz(p::SparsePattern) = Offset(length(p.col_idx))

struct CsrMatrix
    pattern::SparsePattern
    vals::Vector{Float64}
end

n_rows(a::CsrMatrix) = a.pattern.n_rows
n_cols(a::CsrMatrix) = a.pattern.n_cols
nnz(a::CsrMatrix) = nnz(a.pattern)

"""
    from_triplets(n_rows, n_cols, rows, cols, vals)

Build CSR from unordered 1-based triplets, summing duplicate (row, col) entries.
Column indices within each row come out ascending.
"""
function from_triplets(n_rows::Integer, n_cols::Integer,
                       rows::AbstractVector{<:Integer},
                       cols::AbstractVector{<:Integer},
                       vals::AbstractVector{<:Real})
    length(rows) == length(cols) == length(vals) ||
        throw(ArgumentError("from_triplets: mismatched triplet lengths"))
    nr = Index(n_rows)
    nc = Index(n_cols)
    nt = length(rows)

    row_ptr = zeros(Offset, Int(nr) + 1)
    for k in 1:nt
        r = Index(rows[k])
        c = Index(cols[k])
        (1 <= r <= nr) || throw(ArgumentError("from_triplets: row index out of range"))
        (1 <= c <= nc) || throw(ArgumentError("from_triplets: column index out of range"))
        row_ptr[Int(r) + 1] += Offset(1)
    end
    row_ptr[1] = Offset(1)
    for r in 1:Int(nr)
        row_ptr[r + 1] += row_ptr[r]
    end

    col_idx = Vector{Index}(undef, nt)
    v = Vector{Float64}(undef, nt)
    cursor = copy(row_ptr[1:Int(nr)])
    for k in 1:nt
        r = Int(Index(rows[k]))
        dst = Int(cursor[r])
        col_idx[dst] = Index(cols[k])
        v[dst] = Float64(vals[k])
        cursor[r] += Offset(1)
    end

    out_col = Index[]
    out_val = Float64[]
    sizehint!(out_col, nt)
    sizehint!(out_val, nt)
    out_ptr = zeros(Offset, Int(nr) + 1)
    out_ptr[1] = Offset(1)
    for r in 1:Int(nr)
        b = Int(row_ptr[r])
        e = Int(row_ptr[r + 1]) - 1
        order = sortperm(view(col_idx, b:e))
        i = 1
        while i <= length(order)
            c = col_idx[b + order[i] - 1]
            acc = 0.0
            j = i
            while j <= length(order) && col_idx[b + order[j] - 1] == c
                acc += v[b + order[j] - 1]
                j += 1
            end
            push!(out_col, c)
            push!(out_val, acc)
            i = j
        end
        out_ptr[r + 1] = Offset(length(out_col) + 1)
    end

    CsrMatrix(SparsePattern(nr, nc, out_ptr, out_col), out_val)
end

struct CscPattern
    n_rows::Index
    n_cols::Index
    col_ptr::Vector{Offset}
    row_idx::Vector{Index}

    function CscPattern(n_rows::Integer, n_cols::Integer,
                        col_ptr::Vector{Offset}, row_idx::Vector{Index})
        validate(new(Index(n_rows), Index(n_cols), col_ptr, row_idx))
    end
end

function validate(p::CscPattern)
    if p.n_rows < 0 || p.n_cols < 0
        throw(ArgumentError("CscPattern: negative dimension"))
    end
    if length(p.col_ptr) != Int(p.n_cols) + 1
        throw(ArgumentError("CscPattern: col_ptr must have n_cols+1 entries"))
    end
    if p.col_ptr[1] != 1
        throw(ArgumentError("CscPattern: col_ptr[1] must be 1"))
    end
    if p.col_ptr[end] != Offset(length(p.row_idx) + 1)
        throw(ArgumentError("CscPattern: col_ptr[end] must equal nnz+1"))
    end
    for j in 1:Int(p.n_cols)
        b = p.col_ptr[j]
        e = p.col_ptr[j + 1]
        e < b && throw(ArgumentError("CscPattern: col_ptr not non-decreasing at column $j"))
        for k in Int(b):(Int(e) - 1)
            i = p.row_idx[k]
            if i < 1 || i > p.n_rows
                throw(ArgumentError("CscPattern: row index out of range in column $j"))
            end
        end
    end
    return p
end

nnz(p::CscPattern) = Offset(length(p.row_idx))

struct CscMatrix
    pattern::CscPattern
    vals::Vector{Float64}
end

n_rows(a::CscMatrix) = a.pattern.n_rows
n_cols(a::CscMatrix) = a.pattern.n_cols
nnz(a::CscMatrix) = nnz(a.pattern)

"""
    to_csc(a)

Transpose CSR into CSC. Row indices within each column come out ascending
because CSR rows are visited in order (same as the C++ counting-sort port).
"""
function to_csc(a::CsrMatrix)
    nr = a.pattern.n_rows
    nc = a.pattern.n_cols
    rp = a.pattern.row_ptr
    ci = a.pattern.col_idx
    length(a.vals) == length(ci) ||
        throw(ArgumentError("to_csc: vals size does not match pattern nnz"))

    nz = length(ci)
    col_ptr = zeros(Offset, Int(nc) + 1)
    for j in ci
        col_ptr[Int(j) + 1] += Offset(1)
    end
    for j in 1:Int(nc)
        col_ptr[j + 1] += col_ptr[j]
    end
    col_ptr .+= Offset(1)

    row_idx = Vector{Index}(undef, nz)
    vals = Vector{Float64}(undef, nz)
    next = copy(col_ptr[1:Int(nc)])
    for i in 1:Int(nr)
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            j = Int(ci[k])
            dst = Int(next[j])
            row_idx[dst] = Index(i)
            vals[dst] = a.vals[k]
            next[j] += Offset(1)
        end
    end
    CscMatrix(CscPattern(nr, nc, col_ptr, row_idx), vals)
end

end # module
