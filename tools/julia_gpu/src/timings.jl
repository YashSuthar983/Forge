# Transfer and kernel timing, reported back to C++ so that a GPU number can
# never be quoted without its host<->device cost.
# Mirrors sor::backend::TransferStats.

mutable struct Stats
    h2d_bytes::Int
    d2h_bytes::Int
    h2d_ms::Float64
    d2h_ms::Float64
    kernel_ms::Float64
end

Stats() = Stats(0, 0, 0.0, 0.0, 0.0)

as_dict(s::Stats) = Dict(
    "h2d_bytes" => s.h2d_bytes,
    "d2h_bytes" => s.d2h_bytes,
    "h2d_ms"    => s.h2d_ms,
    "d2h_ms"    => s.d2h_ms,
    "kernel_ms" => s.kernel_ms,
)

"""
    @timed_ms expr

Elapsed milliseconds for `expr`, plus its value, without allocating a NamedTuple
per call in the hot path.
"""
macro timed_ms(expr)
    quote
        local _t0 = time_ns()
        local _v = $(esc(expr))
        (_v, (time_ns() - _t0) / 1.0e6)
    end
end

nbytes(a::AbstractArray) = sizeof(eltype(a)) * length(a)
