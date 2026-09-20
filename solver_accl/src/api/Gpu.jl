module SorApiGpu

# Runtime discovery of whichever GPU backend this machine actually has.
#
# None of CUDA.jl / AMDGPU.jl / oneAPI.jl / Metal.jl is a declared dependency:
# their artifacts are multi-gigabyte and only one of them is ever useful on a
# given box. So they are located at runtime and used only when the vendor
# package reports a functional device. If none is present we say so and fall
# back to KernelAbstractions' CPU backend -- this module never reports a GPU
# it cannot actually dispatch to.
#
# This is where the vendor-agnostic claim is either true or isn't. The kernels
# in gpu/ are written against KernelAbstractions.Backend and never name a
# vendor; this function is the only place that does.

using KernelAbstractions

export detect_gpu, gpu_info, GpuProbe

struct GpuProbe
    available::Bool
    vendor::String        # "cuda" | "rocm" | "oneapi" | "metal" | "none"
    device_name::String
    backend::Any          # a KernelAbstractions.Backend, or CPU() when none
    detail::String
end

# (vendor package to try, the short name we report it under)
const CANDIDATES = [
    ("CUDA",    "cuda"),
    ("AMDGPU",  "rocm"),
    ("oneAPI",  "oneapi"),
    ("Metal",   "metal"),
]

function _try_vendor(pkg::String, vendor::String)
    try
        Base.locate_package(Base.identify_package(pkg)) === nothing && return nothing
    catch
        return nothing
    end
    try
        m = @eval Main begin
            using $(Symbol(pkg))
            $(Symbol(pkg))
        end
        isdefined(m, :functional) || return nothing
        Base.invokelatest(getfield(m, :functional)) || return nothing

        backend = if vendor == "cuda"
            Base.invokelatest(getfield(m, :CUDABackend))
        elseif vendor == "rocm"
            Base.invokelatest(getfield(m, :ROCBackend))
        elseif vendor == "oneapi"
            Base.invokelatest(getfield(m, :oneAPIBackend))
        else
            Base.invokelatest(getfield(m, :MetalBackend))
        end

        name = try
            if vendor == "cuda"
                string(Base.invokelatest(getfield(m, :name), Base.invokelatest(getfield(m, :device))))
            else
                string(Base.invokelatest(getfield(m, :device)))
            end
        catch
            vendor
        end
        return GpuProbe(true, vendor, name, backend, "$pkg reports a functional device")
    catch
        return nothing
    end
end

"""
    detect_gpu(; prefer=nothing) -> GpuProbe

Probe CUDA, ROCm, oneAPI and Metal in that order and return the first
functional one. `prefer` (a vendor string) is tried first when given.

Never throws: a machine with no GPU, or with a vendor package installed but
broken, comes back with `available=false` and the CPU backend, so a caller can
always use the returned `backend` unconditionally.
"""
function detect_gpu(; prefer::Union{Nothing,AbstractString}=nothing)
    order = copy(CANDIDATES)
    if prefer !== nothing
        i = findfirst(c -> c[2] == prefer, order)
        i === nothing || (order = vcat([order[i]], deleteat!(copy(order), i)))
    end
    for (pkg, vendor) in order
        p = _try_vendor(pkg, vendor)
        p === nothing || return p
    end
    return GpuProbe(false, "none", "cpu", KernelAbstractions.CPU(),
                    "no functional CUDA / ROCm / oneAPI / Metal device found")
end

gpu_info(p::GpuProbe) = Dict{String,Any}(
    "gpu_available" => p.available,
    "gpu_vendor"    => p.vendor,
    "gpu_device"    => p.device_name,
    "gpu_detail"    => p.detail,
)

end # module
