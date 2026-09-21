module SorBackend

# Port of sor_backend: TransferStats, KernelBackend/CpuBackend, LpDevice/CpuLpDevice.
# Julia CSR/CSC is 1-based (row_ptr[1] == 1). Vulkan/CUDA factories return nothing.

using ..SorCore: Index, Offset
using ..SorSparse: SparsePattern, CsrMatrix, CscMatrix, CscPattern, n_rows, n_cols, nnz

export TransferStats, DeviceBuffer, BatchView
export KernelBackend, CpuBackend, make_cpu_backend, make_backend
export ScaledLp, StepParams, RestartPoint, Average, Current, Anchor, LpSolution, Kkt
export LpDevice, CpuLpDevice, make_cpu_lp_device, make_vulkan_lp_device, make_cuda_lp_device
export make_lp_device
export name, is_accelerated, spmv, spmv_t, project_box, dot
export spmv_batched, project_box_batched
export upload, hpr_steps, reduce_kkt, snapshot_anchor, restart_to, init_zero, download
export transfer_stats, reset_stats!

const kInf = Inf
const kAtBound = 1e-9

@inline ms_since(t0::UInt64) = (time_ns() - t0) / 1.0e6

# ---- transfer accounting ----------------------------------------------------

Base.@kwdef mutable struct TransferStats
    h2d_bytes::UInt64 = 0
    d2h_bytes::UInt64 = 0
    h2d_ms::Float64 = 0.0
    d2h_ms::Float64 = 0.0
    kernel_ms::Float64 = 0.0
    ipc_ms::Float64 = 0.0
    calls::UInt64 = 0
end

function add!(a::TransferStats, o::TransferStats)
    a.h2d_bytes += o.h2d_bytes
    a.d2h_bytes += o.d2h_bytes
    a.h2d_ms += o.h2d_ms
    a.d2h_ms += o.d2h_ms
    a.kernel_ms += o.kernel_ms
    a.ipc_ms += o.ipc_ms
    a.calls += o.calls
    return a
end

total_ms(s::TransferStats) = s.h2d_ms + s.d2h_ms + s.kernel_ms + s.ipc_ms

# ---- host-staged buffers (CPU reference; same honesty caveats as C++) ------

mutable struct DeviceBuffer{T}
    host::Vector{T}
    DeviceBuffer{T}(host::Vector{T}) where {T} = new{T}(host)
end

DeviceBuffer{T}() where {T} = DeviceBuffer{T}(T[])
DeviceBuffer{T}(n::Integer) where {T} = DeviceBuffer{T}(fill(zero(T), Int(n)))
DeviceBuffer(v::Vector{T}) where {T} = DeviceBuffer{T}(v)

Base.length(b::DeviceBuffer) = length(b.host)
Base.size(b::DeviceBuffer) = size(b.host)
Base.isempty(b::DeviceBuffer) = isempty(b.host)
Base.eltype(::DeviceBuffer{T}) where {T} = T
Base.getindex(b::DeviceBuffer, i::Integer) = b.host[i]
Base.setindex!(b::DeviceBuffer, v, i::Integer) = (b.host[i] = v)
host(b::DeviceBuffer) = b.host

function Base.resize!(b::DeviceBuffer{T}, n::Integer) where {T}
    old = length(b.host)
    n = Int(n)
    resize!(b.host, n)
    if n > old
        @inbounds for i in (old + 1):n
            b.host[i] = zero(T)
        end
    end
    return b
end

function Base.fill!(b::DeviceBuffer{T}, v) where {T}
    fill!(b.host, convert(T, v))
    return b
end

mutable struct BatchView{T}
    data::Vector{T}
    n_items::Int
    item_len::Int
    offset::Int  # 0-based start into data
end

BatchView{T}() where {T} = BatchView{T}(T[], 0, 0, 0)

function BatchView(data::Vector{T}, n_items::Integer, item_len::Integer) where {T}
    BatchView{T}(data, Int(n_items), Int(item_len), 0)
end

function from_buffer(b::DeviceBuffer{T}, n_items::Integer) where {T}
    n = Int(n_items)
    len = n == 0 ? 0 : length(b) ÷ n
    BatchView{T}(b.host, n, len, 0)
end

n_items(v::BatchView) = v.n_items
item_len(v::BatchView) = v.item_len
total(v::BatchView) = v.n_items * v.item_len

function item_range(v::BatchView, k::Integer)
    off = v.offset + (Int(k) - 1) * v.item_len
    return (off + 1):(off + v.item_len)
end

# ---- KernelBackend ----------------------------------------------------------

abstract type KernelBackend end

name(::KernelBackend) = "unknown"
is_accelerated(::KernelBackend) = false
transfer_stats(b::KernelBackend) = TransferStats()
reset_stats!(::KernelBackend) = nothing

mutable struct CpuBackend <: KernelBackend
    stats::TransferStats
    CpuBackend() = new(TransferStats())
end

name(::CpuBackend) = "cpu"
is_accelerated(::CpuBackend) = false
transfer_stats(b::CpuBackend) = b.stats
reset_stats!(b::CpuBackend) = (b.stats = TransferStats(); nothing)

function _check_dims(p::SparsePattern, vals::DeviceBuffer, got::Int, want::Integer, what::String)
    Int(nnz(p)) == length(vals) || throw(ArgumentError("kernel: vals.size() != nnz"))
    got == Int(want) || throw(ArgumentError("kernel: bad size for " * what))
    return nothing
end

function spmv(be::CpuBackend, p::SparsePattern, vals::DeviceBuffer{Float64},
              x::DeviceBuffer{Float64}, y::DeviceBuffer{Float64})
    _check_dims(p, vals, length(x), p.n_cols, "spmv x")
    resize!(y, Int(p.n_rows))
    t0 = time_ns()
    rp = p.row_ptr
    ci = p.col_idx
    vh = vals.host
    xh = x.host
    yh = y.host
    nr = Int(p.n_rows)
    @inbounds for r in 1:nr
        acc = 0.0
        k0 = Int(rp[r])
        k1 = Int(rp[r + 1]) - 1
        for k in k0:k1
            acc += vh[k] * xh[Int(ci[k])]
        end
        yh[r] = acc
    end
    be.stats.kernel_ms += ms_since(t0)
    be.stats.calls += 1
    return y
end

function spmv_t(be::CpuBackend, p::SparsePattern, vals::DeviceBuffer{Float64},
                x::DeviceBuffer{Float64}, y::DeviceBuffer{Float64})
    _check_dims(p, vals, length(x), p.n_rows, "spmv_t x")
    resize!(y, Int(p.n_cols))
    fill!(y, 0.0)
    t0 = time_ns()
    rp = p.row_ptr
    ci = p.col_idx
    vh = vals.host
    xh = x.host
    yh = y.host
    nr = Int(p.n_rows)
    @inbounds for r in 1:nr
        xr = xh[r]
        xr == 0.0 && continue
        k0 = Int(rp[r])
        k1 = Int(rp[r + 1]) - 1
        for k in k0:k1
            yh[Int(ci[k])] += vh[k] * xr
        end
    end
    be.stats.kernel_ms += ms_since(t0)
    be.stats.calls += 1
    return y
end

function project_box(be::CpuBackend, x::DeviceBuffer{Float64},
                     lo::DeviceBuffer{Float64}, hi::DeviceBuffer{Float64})
    length(x) == length(lo) == length(hi) ||
        throw(ArgumentError("project_box: mismatched sizes"))
    t0 = time_ns()
    xh = x.host
    loh = lo.host
    hih = hi.host
    @inbounds for i in eachindex(xh)
        v = xh[i]
        v < loh[i] && (v = loh[i])
        v > hih[i] && (v = hih[i])
        xh[i] = v
    end
    be.stats.kernel_ms += ms_since(t0)
    be.stats.calls += 1
    return x
end

function dot(be::CpuBackend, a::DeviceBuffer{Float64}, b::DeviceBuffer{Float64})
    length(a) == length(b) || throw(ArgumentError("dot: mismatched sizes"))
    t0 = time_ns()
    acc = 0.0
    ah = a.host
    bh = b.host
    @inbounds for i in eachindex(ah)
        acc += ah[i] * bh[i]
    end
    be.stats.kernel_ms += ms_since(t0)
    be.stats.calls += 1
    return acc
end

function spmv_batched(be::CpuBackend, p::SparsePattern, vals::DeviceBuffer{Float64},
                      X::BatchView{Float64}, Y::BatchView{Float64})
    item_len(X) == Int(p.n_cols) ||
        throw(ArgumentError("spmv_batched: X item_len != n_cols"))
    item_len(Y) == Int(p.n_rows) ||
        throw(ArgumentError("spmv_batched: Y item_len != n_rows"))
    n_items(X) == n_items(Y) ||
        throw(ArgumentError("spmv_batched: batch count mismatch"))
    t0 = time_ns()
    rp = p.row_ptr
    ci = p.col_idx
    vh = vals.host
    nr = Int(p.n_rows)
    @inbounds for b in 1:n_items(X)
        xr = item_range(X, b)
        yr = item_range(Y, b)
        xb = X.data
        yb = Y.data
        xoff = first(xr) - 1
        yoff = first(yr) - 1
        for r in 1:nr
            acc = 0.0
            k0 = Int(rp[r])
            k1 = Int(rp[r + 1]) - 1
            for k in k0:k1
                acc += vh[k] * xb[xoff + Int(ci[k])]
            end
            yb[yoff + r] = acc
        end
    end
    be.stats.kernel_ms += ms_since(t0)
    be.stats.calls += 1
    return Y
end

function project_box_batched(be::CpuBackend, X::BatchView{Float64},
                             LO::BatchView{Float64}, HI::BatchView{Float64})
    total(X) == total(LO) == total(HI) ||
        throw(ArgumentError("project_box_batched: mismatched shapes"))
    t0 = time_ns()
    n = total(X)
    xoff = X.offset
    loff = LO.offset
    hoff = HI.offset
    @inbounds for i in 1:n
        v = X.data[xoff + i]
        v < LO.data[loff + i] && (v = LO.data[loff + i])
        v > HI.data[hoff + i] && (v = HI.data[hoff + i])
        X.data[xoff + i] = v
    end
    be.stats.kernel_ms += ms_since(t0)
    be.stats.calls += 1
    return X
end

make_cpu_backend() = CpuBackend()

function make_backend(name::AbstractString)
    name == "cpu" && return make_cpu_backend()
    return nothing
end

# ---- LpDevice ---------------------------------------------------------------

mutable struct ScaledLp
    A_csr::CsrMatrix
    A_csc::CscMatrix
    c::Vector{Float64}
    col_lo::Vector{Float64}
    col_hi::Vector{Float64}
    row_lo::Vector{Float64}
    row_hi::Vector{Float64}
    row_scale::Vector{Float64}
    col_scale::Vector{Float64}
    obj_offset::Float64
    sense::Float64
end

Base.@kwdef mutable struct StepParams
    tau::Float64 = 1.0
    sigma::Float64 = 1.0
    beta::Float64 = 0.0
    update_average::Bool = true
    use_halpern::Bool = false
end

@enum RestartPoint::UInt8 begin
    Average = 0
    Current = 1
    Anchor = 2
end

mutable struct LpSolution
    x::Vector{Float64}
    y::Vector{Float64}
    x_avg::Vector{Float64}
    y_avg::Vector{Float64}
end
LpSolution() = LpSolution(Float64[], Float64[], Float64[], Float64[])

Base.@kwdef mutable struct Kkt
    primal_res::Float64 = 0.0
    dual_res::Float64 = 0.0
    primal_obj::Float64 = 0.0
    dual_obj::Float64 = 0.0
    gap_rel::Float64 = 0.0
    dx_norm::Float64 = 0.0
    dy_norm::Float64 = 0.0
    restart_metric::Float64 = 0.0
    dual_bound_finite::Bool = false
end

abstract type LpDevice end

name(::LpDevice) = "unknown"
is_accelerated(::LpDevice) = false

@inline clamp_to(v::Float64, lo::Float64, hi::Float64) = v < lo ? lo : (v > hi ? hi : v)
@inline mul_zero_safe(a::Float64, b::Float64) = a == 0.0 ? 0.0 : a * b

mutable struct CpuLpDevice <: LpDevice
    uploaded::Bool
    nr::Int
    nc::Int
    A_csr::Union{CsrMatrix,Nothing}
    A_csc::Union{CscMatrix,Nothing}
    c::Vector{Float64}
    col_lo::Vector{Float64}
    col_hi::Vector{Float64}
    row_lo::Vector{Float64}
    row_hi::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    xbar::Vector{Float64}
    Aty::Vector{Float64}
    Ax::Vector{Float64}
    x_avg::Vector{Float64}
    y_avg::Vector{Float64}
    x_anchor::Vector{Float64}
    y_anchor::Vector{Float64}
    avg_count::UInt64
    epoch_step::UInt64
    last_dx::Float64
    last_dy::Float64
    stats::TransferStats
    function CpuLpDevice()
        new(false, 0, 0, nothing, nothing,
            Float64[], Float64[], Float64[], Float64[], Float64[],
            Float64[], Float64[], Float64[], Float64[], Float64[],
            Float64[], Float64[], Float64[], Float64[],
            UInt64(0), UInt64(0), 0.0, 0.0, TransferStats())
    end
end

name(::CpuLpDevice) = "cpu"
is_accelerated(::CpuLpDevice) = false
transfer_stats(d::CpuLpDevice) = d.stats
reset_stats!(d::CpuLpDevice) = (d.stats = TransferStats(); nothing)

function _require_uploaded(d::CpuLpDevice)
    d.uploaded || error("CpuLpDevice: upload() required")
    return nothing
end

function _spmv_csr!(d::CpuLpDevice, x::Vector{Float64}, y::Vector{Float64})
    A = d.A_csr::CsrMatrix
    rp = A.pattern.row_ptr
    ci = A.pattern.col_idx
    v = A.vals
    nr = d.nr
    @inbounds for r in 1:nr
        acc = 0.0
        k0 = Int(rp[r])
        k1 = Int(rp[r + 1]) - 1
        for k in k0:k1
            acc += v[k] * x[Int(ci[k])]
        end
        y[r] = acc
    end
    return y
end

function _spmv_csc!(d::CpuLpDevice, x::Vector{Float64}, y::Vector{Float64})
    A = d.A_csc::CscMatrix
    cp = A.pattern.col_ptr
    ri = A.pattern.row_idx
    v = A.vals
    nc = d.nc
    @inbounds for j in 1:nc
        acc = 0.0
        k0 = Int(cp[j])
        k1 = Int(cp[j + 1]) - 1
        for k in k0:k1
            acc += v[k] * x[Int(ri[k])]
        end
        y[j] = acc
    end
    return y
end

function upload(d::CpuLpDevice, lp::ScaledLp)
    d.nr = Int(n_rows(lp.A_csr))
    d.nc = Int(n_cols(lp.A_csr))
    d.A_csr = CsrMatrix(
        SparsePattern(lp.A_csr.pattern.n_rows, lp.A_csr.pattern.n_cols,
                      copy(lp.A_csr.pattern.row_ptr), copy(lp.A_csr.pattern.col_idx)),
        copy(lp.A_csr.vals))
    d.A_csc = CscMatrix(
        CscPattern(lp.A_csc.pattern.n_rows, lp.A_csc.pattern.n_cols,
                   copy(lp.A_csc.pattern.col_ptr), copy(lp.A_csc.pattern.row_idx)),
        copy(lp.A_csc.vals))
    d.c = copy(lp.c)
    d.col_lo = copy(lp.col_lo)
    d.col_hi = copy(lp.col_hi)
    d.row_lo = copy(lp.row_lo)
    d.row_hi = copy(lp.row_hi)
    nbytes = (length(lp.A_csr.vals) + length(lp.A_csc.vals) + length(d.c) +
              length(d.col_lo) + length(d.col_hi) + length(d.row_lo) +
              length(d.row_hi)) * sizeof(Float64)
    d.stats.h2d_bytes += UInt64(nbytes)
    d.stats.calls += 1
    nc = d.nc
    nr = d.nr
    d.x = zeros(nc)
    d.y = zeros(nr)
    d.xbar = zeros(nc)
    d.Aty = zeros(nc)
    d.Ax = zeros(nr)
    d.x_avg = zeros(nc)
    d.y_avg = zeros(nr)
    d.x_anchor = zeros(nc)
    d.y_anchor = zeros(nr)
    d.avg_count = 0
    d.epoch_step = 0
    d.last_dx = 0.0
    d.last_dy = 0.0
    d.uploaded = true
    return nothing
end

function init_zero(d::CpuLpDevice)
    _require_uploaded(d)
    @inbounds for j in 1:d.nc
        d.x[j] = clamp_to(0.0, d.col_lo[j], d.col_hi[j])
    end
    fill!(d.y, 0.0)
    d.x_avg = copy(d.x)
    d.y_avg = copy(d.y)
    d.avg_count = 1
    d.epoch_step = 0
    snapshot_anchor(d)
    return nothing
end

function hpr_steps(d::CpuLpDevice, k::Integer, p::StepParams)
    _require_uploaded(d)
    nc = d.nc
    nr = d.nr
    tau = p.tau
    sigma = p.sigma
    for _ in 1:Int(k)
        _spmv_csc!(d, d.y, d.Aty)
        dx2 = 0.0
        @inbounds for j in 1:nc
            xn = clamp_to(d.x[j] - tau * (d.c[j] + d.Aty[j]), d.col_lo[j], d.col_hi[j])
            delta = xn - d.x[j]
            dx2 += delta * delta
            d.xbar[j] = 2.0 * xn - d.x[j]
            d.x[j] = xn
        end
        d.last_dx = sqrt(dx2)

        _spmv_csr!(d, d.xbar, d.Ax)
        dy2 = 0.0
        @inbounds for i in 1:nr
            v = d.y[i] + sigma * d.Ax[i]
            z = clamp_to(v / sigma, d.row_lo[i], d.row_hi[i])
            yn = v - sigma * z
            delta = yn - d.y[i]
            dy2 += delta * delta
            d.y[i] = yn
        end
        d.last_dy = sqrt(dy2)

        if p.use_halpern
            beta = 1.0 / (Float64(d.epoch_step) + 2.0)
            om = 1.0 - beta
            @inbounds for j in 1:nc
                d.x[j] = om * d.x[j] + beta * d.x_anchor[j]
            end
            @inbounds for i in 1:nr
                d.y[i] = om * d.y[i] + beta * d.y_anchor[i]
            end
        end

        if p.update_average
            d.avg_count += 1
            inv = 1.0 / Float64(d.avg_count)
            @inbounds for j in 1:nc
                d.x_avg[j] += (d.x[j] - d.x_avg[j]) * inv
            end
            @inbounds for i in 1:nr
                d.y_avg[i] += (d.y[i] - d.y_avg[i]) * inv
            end
        end
        d.epoch_step += 1
        d.stats.calls += 1
    end
    return nothing
end

function reduce_kkt(d::CpuLpDevice)
    _require_uploaded(d)
    kkt = Kkt()
    xp = d.x
    yp = d.y
    _spmv_csr!(d, xp, d.Ax)
    pres = 0.0
    @inbounds for i in 1:d.nr
        a = d.Ax[i]
        a < d.row_lo[i] && (pres = max(pres, d.row_lo[i] - a))
        a > d.row_hi[i] && (pres = max(pres, a - d.row_hi[i]))
    end

    _spmv_csc!(d, yp, d.Aty)
    dres = 0.0
    @inbounds for j in 1:d.nc
        r = d.c[j] + d.Aty[j]
        at_lo = (d.col_lo[j] > -kInf) && (xp[j] <= d.col_lo[j] + kAtBound)
        at_hi = (d.col_hi[j] < kInf) && (xp[j] >= d.col_hi[j] - kAtBound)
        if at_lo && !at_hi
            dres = max(dres, max(0.0, -r))
        elseif at_hi && !at_lo
            dres = max(dres, max(0.0, r))
        elseif !at_lo && !at_hi
            dres = max(dres, abs(r))
        end
    end

    pobj = 0.0
    @inbounds for j in 1:d.nc
        pobj += d.c[j] * xp[j]
    end

    finite = true
    dval = 0.0
    @inbounds for j in 1:d.nc
        r = d.c[j] + d.Aty[j]
        b = r >= 0.0 ? d.col_lo[j] : d.col_hi[j]
        if r != 0.0 && isinf(b)
            finite = false
            break
        end
        dval += mul_zero_safe(r, b)
    end
    if finite
        @inbounds for i in 1:d.nr
            yi = yp[i]
            b = yi >= 0.0 ? d.row_hi[i] : d.row_lo[i]
            if yi != 0.0 && isinf(b)
                finite = false
                break
            end
            dval -= mul_zero_safe(yi, b)
        end
    end

    kkt.primal_res = pres
    kkt.dual_res = dres
    kkt.primal_obj = pobj
    kkt.dual_bound_finite = finite && isfinite(dval)
    kkt.dual_obj = kkt.dual_bound_finite ? dval : NaN
    kkt.gap_rel = kkt.dual_bound_finite ? abs(pobj - dval) / (1.0 + abs(pobj)) : Inf
    kkt.dx_norm = d.last_dx
    kkt.dy_norm = d.last_dy
    kkt.restart_metric = kkt.gap_rel
    d.stats.d2h_bytes += UInt64(8 * sizeof(Float64))
    d.stats.calls += 1
    return kkt
end

function snapshot_anchor(d::CpuLpDevice)
    _require_uploaded(d)
    copyto!(d.x_anchor, d.x)
    copyto!(d.y_anchor, d.y)
    return nothing
end

function restart_to(d::CpuLpDevice, rp::RestartPoint)
    _require_uploaded(d)
    if rp === Average
        copyto!(d.x, d.x_avg)
        copyto!(d.y, d.y_avg)
    elseif rp === Anchor
        copyto!(d.x, d.x_anchor)
        copyto!(d.y, d.y_anchor)
    end
    copyto!(d.x_avg, d.x)
    copyto!(d.y_avg, d.y)
    d.avg_count = 1
    d.epoch_step = 0
    snapshot_anchor(d)
    return nothing
end

function download(d::CpuLpDevice, sol::LpSolution)
    _require_uploaded(d)
    sol.x = copy(d.x)
    sol.y = copy(d.y)
    sol.x_avg = copy(d.x_avg)
    sol.y_avg = copy(d.y_avg)
    d.stats.d2h_bytes += UInt64((length(d.x) + length(d.y) + length(d.x_avg) +
                                 length(d.y_avg)) * sizeof(Float64))
    d.stats.calls += 1
    return sol
end

make_cpu_lp_device() = CpuLpDevice()
make_vulkan_lp_device(device::Integer=-1) = nothing
make_cuda_lp_device(device::Integer=-1) = nothing

function make_lp_device(devname::AbstractString, device::Integer=-1)
    devname == "cpu" && return make_cpu_lp_device()
    devname == "vulkan" && return make_vulkan_lp_device(device)
    devname == "cuda" && return make_cuda_lp_device(device)
    return nothing
end

end # module
