using Test
using Random
using SparseArrays
using LinearAlgebra
using SORGpuProto

const P = SORGpuProto

# Reference values come from SparseArrays, which is a general sparse linear
# algebra library, NOT an optimization solver -- permitted by
# sor/docs/clean_room_policy.md. It is used only as a test oracle.

@testset "SORGpuProto" begin

dev = P.select_device("auto")
@info "test device" name=dev.name accelerated=dev.accelerated

@testset "device selection" begin
    @test dev.name in ("cpu", "cuda")
    info = P.device_info(dev)
    @test haskey(info, "device")
    @test info["accelerated"] == dev.accelerated
    # On a machine with no GPU this must be false, never silently true.
    if dev.name == "cpu"
        @test dev.accelerated == false
    end
end

# Build a deterministic sparse matrix and its 0-based CSR, as C++ would send it.
# Built from explicit triplets with a guaranteed entry per row, so no row is empty
# and the result is reproducible for a fixed seed.
function make_csr(m, n, density; seed = 20260828)
    rng = MersenneTwister(seed)
    I = Int[]
    J = Int[]
    V = Float64[]
    per_row = max(1, round(Int, density * n))
    for r in 1:m
        push!(I, r); push!(J, ((r - 1) % n) + 1); push!(V, 1.0 + 0.1 * rand(rng))
        for _ in 2:per_row
            push!(I, r); push!(J, rand(rng, 1:n)); push!(V, 2.0 * rand(rng) - 1.0)
        end
    end
    A = sparse(I, J, V, m, n, +)      # sum duplicates
    At = sparse(A')                   # CSC of A' == CSR of A
    row_ptr0 = Int.(At.colptr) .- 1
    col_idx0 = Int.(At.rowval) .- 1
    vals = Float64.(At.nzval)
    return A, row_ptr0, col_idx0, vals
end

@testset "build_pattern" begin
    A, rp, ci, vals = make_csr(9, 7, 0.35)
    pat, ms, bytes = P.build_pattern(dev, 9, 7, rp, ci)
    @test pat.n_rows == 9
    @test pat.n_cols == 7
    @test pat.nnz == length(ci)
    @test bytes > 0
    @test ms >= 0.0

    # Malformed input must be rejected.
    @test_throws Exception P.build_pattern(dev, 9, 7, rp[1:end-1], ci)
    @test_throws Exception P.build_pattern(dev, 9, 7, rp, [c + 100 for c in ci])
end

@testset "spmv matches SparseArrays" begin
    for (m, n, d) in ((9, 7, 0.35), (40, 33, 0.15), (128, 96, 0.05))
        A, rp, ci, vals = make_csr(m, n, d)
        pat, _, _ = P.build_pattern(dev, m, n, rp, ci)
        dvals, _, _ = P.to_device(dev, vals)

        x = collect(range(-1.0, 1.0; length = n))
        dx, _, _ = P.to_device(dev, x)
        y = P.dev_zeros(dev, Float64, m)
        P.spmv!(dev, y, pat, dvals, dx)
        got, _, _ = P.to_host(dev, y)

        want = A * x
        @test isapprox(got, want; rtol = 1e-12, atol = 1e-12)
    end
end

@testset "spmv_t matches SparseArrays" begin
    for (m, n, d) in ((9, 7, 0.35), (40, 33, 0.15))
        A, rp, ci, vals = make_csr(m, n, d)
        pat, _, _ = P.build_pattern(dev, m, n, rp, ci)
        dvals, _, _ = P.to_device(dev, vals)
        tvals = P.dev_zeros(dev, Float64, pat.nnz)

        x = collect(range(0.5, 2.0; length = m))
        dx, _, _ = P.to_device(dev, x)
        y = P.dev_zeros(dev, Float64, n)
        P.spmv_t!(dev, y, pat, dvals, dx, tvals)
        got, _, _ = P.to_host(dev, y)

        want = A' * x
        @test isapprox(got, want; rtol = 1e-12, atol = 1e-12)
    end
end

@testset "spmv_t is deterministic across repeats" begin
    # No atomic scatter, so repeated calls must be bit-identical (commitment C3).
    A, rp, ci, vals = make_csr(64, 50, 0.2)
    pat, _, _ = P.build_pattern(dev, 64, 50, rp, ci)
    dvals, _, _ = P.to_device(dev, vals)
    tvals = P.dev_zeros(dev, Float64, pat.nnz)
    x = collect(range(-2.0, 3.0; length = 64))
    dx, _, _ = P.to_device(dev, x)

    first_result = nothing
    for _ in 1:5
        y = P.dev_zeros(dev, Float64, 50)
        P.spmv_t!(dev, y, pat, dvals, dx, tvals)
        h, _, _ = P.to_host(dev, y)
        if first_result === nothing
            first_result = h
        else
            @test h == first_result      # bit-identical, not isapprox
        end
    end
end

@testset "project_box" begin
    x = [-5.0, -0.5, 0.0, 0.5, 5.0, Inf, -Inf]
    lo = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    hi = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    dx, _, _ = P.to_device(dev, copy(x))
    dlo, _, _ = P.to_device(dev, lo)
    dhi, _, _ = P.to_device(dev, hi)
    P.project_box!(dev, dx, dlo, dhi)
    got, _, _ = P.to_host(dev, dx)
    @test got == [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0, -1.0]

    # Infinite bounds must pass values through untouched.
    y = [3.0, -7.0]
    dy, _, _ = P.to_device(dev, copy(y))
    dl, _, _ = P.to_device(dev, [-Inf, -Inf])
    dh, _, _ = P.to_device(dev, [Inf, Inf])
    P.project_box!(dev, dy, dl, dh)
    got2, _, _ = P.to_host(dev, dy)
    @test got2 == y
end

@testset "dot" begin
    a = collect(1.0:64.0)
    b = collect(64.0:-1.0:1.0)
    da, _, _ = P.to_device(dev, a)
    db, _, _ = P.to_device(dev, b)
    @test isapprox(P.device_dot(dev, da, db), LinearAlgebra.dot(a, b); rtol = 1e-14)
    e = Float64[]
    de, _, _ = P.to_device(dev, e)
    @test P.device_dot(dev, de, de) == 0.0
end

@testset "spmv_batched matches per-item spmv" begin
    m, n, nb = 37, 29, 6
    A, rp, ci, vals = make_csr(m, n, 0.2)
    pat, _, _ = P.build_pattern(dev, m, n, rp, ci)
    dvals, _, _ = P.to_device(dev, vals)

    Xh = Float64[]
    cols = Vector{Vector{Float64}}()
    for b in 1:nb
        col = collect(range(-1.0 * b, 1.0 * b; length = n))
        push!(cols, col)
        append!(Xh, col)
    end
    dX, _, _ = P.to_device(dev, Xh)
    Y = P.dev_zeros(dev, Float64, nb * m)
    P.spmv_batched!(dev, Y, pat, dvals, dX, nb)
    Yh, _, _ = P.to_host(dev, Y)

    for b in 1:nb
        want = A * cols[b]
        got = Yh[((b - 1) * m + 1):(b * m)]
        @test isapprox(got, want; rtol = 1e-12, atol = 1e-12)
    end

    # Shape errors must throw, not silently read out of bounds.
    @test_throws Exception P.spmv_batched!(dev, Y, pat, dvals, dX, nb + 1)
end

@testset "project_box_batched uses per-item bounds" begin
    nb, len = 3, 4
    X = Float64[]
    LO = Float64[]
    HI = Float64[]
    for b in 1:nb
        append!(X, [-10.0, -0.1, 0.1, 10.0])
        append!(LO, fill(-Float64(b), len))
        append!(HI, fill(Float64(b), len))
    end
    dX, _, _ = P.to_device(dev, copy(X))
    dLO, _, _ = P.to_device(dev, LO)
    dHI, _, _ = P.to_device(dev, HI)
    P.project_box_batched!(dev, dX, dLO, dHI)
    got, _, _ = P.to_host(dev, dX)
    for b in 1:nb
        seg = got[((b - 1) * len + 1):(b * len)]
        @test seg == [-Float64(b), -0.1, 0.1, Float64(b)]
    end
end

@testset "protocol round trip over handles" begin
    state = P.ServerState(dev)

    resp, keep = P.handle(state, Dict("op" => "ping", "id" => 1))
    @test keep && resp["ok"] && resp["pong"]
    @test resp["protocol"] == P.PROTOCOL_VERSION

    m, n = 12, 10
    A, rp, ci, vals = make_csr(m, n, 0.3)
    resp, _ = P.handle(state, Dict("op" => "upload_pattern", "id" => 2,
                                   "n_rows" => m, "n_cols" => n,
                                   "row_ptr" => rp, "col_idx" => ci))
    @test resp["ok"]
    pid = resp["pattern_id"]

    resp, _ = P.handle(state, Dict("op" => "upload_vals", "id" => 3,
                                   "pattern_id" => pid, "vals" => vals))
    @test resp["ok"]
    vid = resp["vals_id"]
    @test resp["stats"]["h2d_bytes"] > 0

    x = collect(range(-1.0, 1.0; length = n))
    resp, _ = P.handle(state, Dict("op" => "spmv", "id" => 4,
                                   "pattern_id" => pid, "vals_id" => vid, "x" => x))
    @test resp["ok"]
    @test isapprox(Float64.(resp["y"]), A * x; rtol = 1e-12, atol = 1e-12)

    xt = collect(range(1.0, 2.0; length = m))
    resp, _ = P.handle(state, Dict("op" => "spmv_t", "id" => 5,
                                   "pattern_id" => pid, "vals_id" => vid, "x" => xt))
    @test resp["ok"]
    @test isapprox(Float64.(resp["y"]), A' * xt; rtol = 1e-12, atol = 1e-12)

    # Errors must come back as ok=false, not as a thrown exception.
    resp, keep = P.handle(state, Dict("op" => "spmv", "id" => 6,
                                      "pattern_id" => 999, "vals_id" => vid,
                                      "x" => x))
    @test keep
    @test resp["ok"] == false
    @test occursin("999", resp["error"])

    resp, keep = P.handle(state, Dict("op" => "nonsense", "id" => 7))
    @test keep && resp["ok"] == false

    # Wrong vector length must be caught.
    resp, _ = P.handle(state, Dict("op" => "spmv", "id" => 8,
                                   "pattern_id" => pid, "vals_id" => vid,
                                   "x" => [1.0, 2.0]))
    @test resp["ok"] == false

    resp, _ = P.handle(state, Dict("op" => "free", "id" => 9,
                                   "pattern_id" => pid, "vals_id" => vid))
    @test resp["ok"]
    resp, _ = P.handle(state, Dict("op" => "spmv", "id" => 10,
                                   "pattern_id" => pid, "vals_id" => vid, "x" => x))
    @test resp["ok"] == false

    resp, keep = P.handle(state, Dict("op" => "shutdown", "id" => 11))
    @test resp["ok"] && keep == false
end

@testset "message framing" begin
    io = IOBuffer()
    P.write_message(io, Dict("hello" => "world", "n" => 3))
    seekstart(io)
    msg = P.read_message(io)
    @test String(msg["hello"]) == "world"
    @test msg["n"] == 3
    @test P.read_message(io) === nothing        # clean EOF

    bad = IOBuffer("notanumber\n{}")
    @test_throws Exception P.read_message(bad)

    short = IOBuffer("100\n{}")
    @test_throws Exception P.read_message(short)
end

@testset "clean room: no solver package loaded" begin
    forbidden = ("HiGHS", "SCIP", "Clp", "GLPK", "JuMP", "MathOptInterface",
                 "Ipopt", "Cbc", "cuOpt", "COPT", "Gurobi", "CPLEX")
    loaded = Set(String(nameof(m)) for m in Base.loaded_modules_array())
    for f in forbidden
        @test !(f in loaded)
    end
end

end # testset
