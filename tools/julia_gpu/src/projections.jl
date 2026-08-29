# Box projection, dot product, and elementwise helpers.
# Standard formulations; no solver source consulted.

@kernel function project_box_kernel!(x, @Const(lo), @Const(hi))
    i = @index(Global, Linear)
    @inbounds begin
        v = x[i]
        l = lo[i]
        h = hi[i]
        v = v < l ? l : v
        v = v > h ? h : v
        x[i] = v
    end
end

"""
    project_box!(dev, x, lo, hi)

x := clamp(x, lo, hi), elementwise. Infinite bounds work because the comparison
form never multiplies by the bound.
"""
function project_box!(dev::DeviceState, x, lo, hi)
    length(x) == length(lo) == length(hi) ||
        error("project_box: mismatched lengths")
    isempty(x) && return x
    kern = project_box_kernel!(dev.backend)
    kern(x, lo, hi; ndrange = length(x))
    KernelAbstractions.synchronize(dev.backend)
    return x
end

"""
    device_dot(dev, a, b) -> Float64

Reduction is delegated to the backend's `LinearAlgebra.dot` (GPUArrays provides a
tree reduction for device arrays). This is deterministic for a fixed backend and
fixed length, but it does NOT match the C++ reference's strictly sequential
summation bit-for-bit -- which is why backend parity is checked to a tolerance
rather than to zero. See tools/julia_gpu/README.md.
"""
function device_dot(dev::DeviceState, a, b)
    length(a) == length(b) || error("dot: mismatched lengths")
    isempty(a) && return 0.0
    v = LinearAlgebra.dot(a, b)
    KernelAbstractions.synchronize(dev.backend)
    return Float64(v)
end
