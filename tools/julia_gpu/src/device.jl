# Device selection — vendor agnostic.
#
# The kernels in csr_kernels.jl / projections.jl / batched.jl are written against
# KernelAbstractions.jl, so they are ALREADY vendor agnostic: the same source
# compiles for CPU, NVIDIA CUDA, AMD ROCm, Intel oneAPI, and Apple Metal. Only
# this file needs to know which vendor is present.
#
# None of the GPU packages is a declared dependency. Each is discovered at
# runtime, so a CPU-only machine needs none of their artifacts (they are multi-GB
# each) and this package still precompiles and tests. Install only the one your
# hardware needs:
#
#     julia --project=tools/julia_gpu -e 'using Pkg; Pkg.add("AMDGPU")'   # AMD
#     julia --project=tools/julia_gpu -e 'using Pkg; Pkg.add("CUDA")'     # NVIDIA
#
# UUIDs were resolved from the General registry, not guessed: a wrong UUID makes
# detection fail silently, which would look identical to "no GPU present".

struct BackendCandidate
    name::String
    pkgid::Base.PkgId
    make_backend::Function      # Module -> KernelAbstractions backend
    notes::String
end

const BACKEND_CANDIDATES = BackendCandidate[
    BackendCandidate("cuda",
        Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"),
        m -> m.CUDABackend(), "NVIDIA"),
    BackendCandidate("rocm",
        Base.PkgId(Base.UUID("21141c5a-9bdb-4563-92ae-f87d6854732e"), "AMDGPU"),
        m -> m.ROCBackend(), "AMD ROCm/HIP"),
    BackendCandidate("oneapi",
        Base.PkgId(Base.UUID("8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"), "oneAPI"),
        m -> m.oneAPIBackend(), "Intel"),
    BackendCandidate("metal",
        Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"),
        m -> m.MetalBackend(), "Apple"),
]

# Cache per candidate name so a failed probe is not retried on every call.
const _probed = Dict{String,Any}()

"""
    probe(cand) -> (Module, reason) or (nothing, reason)

Loads a GPU package only if it is installed AND its `functional()` returns true.
Never throws: a missing, broken, or permission-denied driver must degrade to CPU,
not crash the sidecar. `reason` is a human string for diagnostics.
"""
function probe(cand::BackendCandidate)
    haskey(_probed, cand.name) && return _probed[cand.name]
    result = try
        if Base.locate_package(cand.pkgid) === nothing
            (nothing, "$(cand.pkgid.name) not installed")
        else
            m = Base.require(cand.pkgid)
            if !m.functional()
                # The usual causes: no device, no driver, or -- on Linux with
                # ROCm -- the user is not in the 'render' group so /dev/kfd
                # cannot be opened.
                (nothing, "$(cand.pkgid.name) installed but functional() == false")
            else
                (m, "$(cand.pkgid.name) functional")
            end
        end
    catch err
        (nothing, "$(cand.pkgid.name) load failed: $(sprint(showerror, err))")
    end
    _probed[cand.name] = result
    return result
end

struct DeviceState
    backend::Any        # a KernelAbstractions backend
    name::String        # "cpu" | "cuda" | "rocm" | "oneapi" | "metal"
    accelerated::Bool
    gpu::Any            # the vendor module, or nothing
    detail::String      # device description, for the ping response
end

"""
    select_device(prefer="auto") -> DeviceState

`prefer` is "auto" (try every vendor in order), a specific vendor name
("cuda", "rocm", "oneapi", "metal"), or "cpu".

A named vendor that turns out to be unavailable falls back to CPU rather than
erroring, but `accelerated` then reports false, so no timing table can ever
label a CPU run as GPU.
"""
function select_device(prefer::AbstractString = "auto")
    prefer = lowercase(prefer)
    if prefer == "cpu"
        return DeviceState(CPU(), "cpu", false, nothing, cpu_detail())
    end

    candidates = prefer == "auto" ? BACKEND_CANDIDATES :
                 filter(c -> c.name == prefer, BACKEND_CANDIDATES)
    if isempty(candidates)
        @warn "unknown device preference '$prefer'; using CPU"
        return DeviceState(CPU(), "cpu", false, nothing, cpu_detail())
    end

    reasons = String[]
    for cand in candidates
        m, why = probe(cand)
        push!(reasons, why)
        if m !== nothing
            return DeviceState(cand.make_backend(m), cand.name, true, m,
                               gpu_detail(cand, m))
        end
    end

    for r in reasons
        println(stderr, "device probe: $r")
    end
    prefer != "auto" && @warn "$prefer requested but unavailable; using CPU"
    return DeviceState(CPU(), "cpu", false, nothing, cpu_detail())
end

cpu_detail() = "CPU ($(Sys.CPU_NAME), $(Threads.nthreads()) thread(s))"

function gpu_detail(cand::BackendCandidate, m)
    # Each vendor package names its device query differently; none of these is
    # load-bearing, so failure just yields a generic label.
    try
        if cand.name == "cuda"
            d = m.device()
            return "$(m.name(d)) (CUDA)"
        elseif cand.name == "rocm"
            d = m.device()
            return "$(repr(d)) (ROCm)"
        elseif cand.name == "oneapi"
            return "$(repr(m.device())) (oneAPI)"
        elseif cand.name == "metal"
            return "$(repr(m.device())) (Metal)"
        end
    catch
    end
    return "$(cand.notes) device"
end

device_info(d::DeviceState) = Dict(
    "device"      => d.name,
    "accelerated" => d.accelerated,
    "detail"      => d.detail,
    "threads"     => Threads.nthreads(),
    "julia"       => string(VERSION),
)

# Allocate a device array of the right type for this backend.
dev_zeros(d::DeviceState, ::Type{T}, n::Integer) where {T} =
    KernelAbstractions.zeros(d.backend, T, n)

"""
    to_device(d, v) -> (device_array, ms, bytes)

Host -> device copy, timed and measured. On the CPU backend this is a plain copy,
so the reported cost is real (small) rather than fabricated as zero.
"""
function to_device(d::DeviceState, v::Vector{T}) where {T}
    arr, ms = @timed_ms begin
        a = KernelAbstractions.allocate(d.backend, T, length(v))
        copyto!(a, v)
        KernelAbstractions.synchronize(d.backend)
        a
    end
    return arr, ms, nbytes(v)
end

"""
    to_host(d, a) -> (Vector, ms, bytes)
"""
function to_host(d::DeviceState, a)
    v, ms = @timed_ms begin
        h = Array(a)
        KernelAbstractions.synchronize(d.backend)
        h
    end
    return v, ms, nbytes(v)
end
