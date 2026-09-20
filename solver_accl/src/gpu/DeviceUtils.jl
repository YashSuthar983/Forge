module GPUDeviceUtils

using KernelAbstractions
using SparseArrays

export build_csr, todev, devzeros

# CSC -> CSR via counting sort, O(nnz + m). Pure CPU preprocessing,
# shared by every GPU solver backend that needs a row-parallel matvec
# (single-instance PDHG today, batched PDHG next) -- each instance's
# CSR is built once here, then the arrays are pushed to whatever
# backend/device via `todev`.
function build_csr(A::SparseMatrixCSC{Float64,Int})
    m, n = size(A)
    nz = nnz(A)
    counts = zeros(Int, m)
    for r in A.rowval
        counts[r] += 1
    end
    row_ptr = Vector{Int}(undef, m + 1)
    row_ptr[1] = 1
    for i in 1:m
        row_ptr[i+1] = row_ptr[i] + counts[i]
    end
    row_idx = Vector{Int}(undef, nz)
    row_val = Vector{Float64}(undef, nz)
    cursor = copy(row_ptr[1:m])
    for j in 1:n
        for k in A.colptr[j]:(A.colptr[j+1]-1)
            r = A.rowval[k]
            pos = cursor[r]
            row_idx[pos] = j
            row_val[pos] = A.nzval[k]
            cursor[r] += 1
        end
    end
    return row_ptr, row_idx, row_val
end

todev(backend, v::Vector) = (d = KernelAbstractions.allocate(backend, eltype(v), length(v)); copyto!(d, v); d)
devzeros(backend, T, n) = (v = KernelAbstractions.allocate(backend, T, n); fill!(v, zero(T)); v)

end # module
