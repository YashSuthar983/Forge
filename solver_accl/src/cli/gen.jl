module SorGen

# Seeded industrial instance generators. Port of sor/cli/sor_gen.cpp.

using Printf
using ..SorCore: Index
using ..SorModel: LpProblem, kInf, n_integer
import ..SorModel: n_rows, n_cols
using ..SorSparse: from_triplets
using ..SorIO: write_mps_file, write_qps_file

export sor_gen_main

# Tiny LCG so generators are seeded without adding Random to Project.toml.
mutable struct LCG
    s::UInt64
end
LCG(seed::UInt32) = LCG(UInt64(seed) ⊻ 0x9e3779b97f4a7c15)
function next_u64!(r::LCG)
    r.s = r.s * UInt64(0x5851f42d4c957f2d) + UInt64(1)
    return r.s
end
function unif(r::LCG, a::Float64, b::Float64)
    u = Float64(next_u64!(r) >> 11) * (1.0 / 9007199254740992.0)
    return a + (b - a) * u
end

function usage()
    print(stderr,
        "usage: sor_gen blend|schedule|dispatch|all [options]\n",
        "  --seed N         RNG seed (recorded in MANIFEST)\n",
        "  --outdir DIR     directory for 'all' (default examples/)\n",
        "  -o PATH          output file for a single generator\n",
        "  blend:    --crudes N --products M\n",
        "  schedule: --periods T --units U\n",
        "  dispatch: --gens G\n")
end

function parse_u32(s::AbstractString, what::AbstractString)
    v = tryparse(UInt32, s)
    if v === nothing
        @printf(stderr, "error: bad %s '%s'\n", what, s)
        return nothing
    end
    return v
end

function make_blend(seed::UInt32, n_crudes::Int, n_products::Int)
    rng = LCG(seed)
    n = n_crudes + n_products
    crude_cap = 5000.0 * Float64(max(n_crudes, 1))
    prod_cap = 4000.0 * Float64(max(n_crudes, 1))
    cdu_cap = crude_cap
    c = zeros(n)
    col_lo = zeros(n)
    col_hi = fill(kInf, n)
    col_names = Vector{String}(undef, n)
    crude_s = zeros(n_crudes)
    for cidx in 1:n_crudes
        crude_cost = unif(rng, 30.0, 80.0)
        crude_s[cidx] = unif(rng, 0.2, 3.5)
        c[cidx] = -crude_cost
        col_names[cidx] = "CRUDE" * string(cidx)
        col_hi[cidx] = crude_cap
    end
    for pr in 1:n_products
        j = n_crudes + pr
        c[j] = unif(rng, 90.0, 160.0)
        col_names[j] = "PROD" * string(pr)
        col_hi[j] = prod_cap
    end
    rows = Index[]; cols = Index[]; vals = Float64[]
    row_lo = Float64[]; row_hi = Float64[]; row_names = String[]
    function add_row(name, lo, hi)
        push!(row_names, name); push!(row_lo, lo); push!(row_hi, hi)
        return Index(length(row_names))
    end
    for pr in 1:n_products
        r = add_row("YLD" * string(pr), 0.0, kInf)
        for cidx in 1:n_crudes
            push!(rows, r); push!(cols, Index(cidx)); push!(vals, unif(rng, 0.15, 0.55))
        end
        push!(rows, r); push!(cols, Index(n_crudes + pr)); push!(vals, -1.0)
    end
    begin
        r = add_row("CAP", -kInf, cdu_cap)
        for cidx in 1:n_crudes
            push!(rows, r); push!(cols, Index(cidx)); push!(vals, 1.0)
        end
    end
    dem_scale = cdu_cap / (4.0 * Float64(max(n_products, 1)))
    for pr in 1:n_products
        dem = min(unif(rng, 1000.0, 5000.0), dem_scale)
        r = add_row("DEM" * string(pr), dem, kInf)
        push!(rows, r); push!(cols, Index(n_crudes + pr)); push!(vals, 1.0)
    end
    begin
        r = add_row("SULF", -kInf, 0.0)
        limit = unif(rng, 0.5, 2.0)
        for cidx in 1:n_crudes
            push!(rows, r); push!(cols, Index(cidx)); push!(vals, crude_s[cidx] - limit)
        end
    end
    p = LpProblem(; name="BLEND_S" * string(Int(seed)),
                  A=from_triplets(length(row_lo), n, rows, cols, vals),
                  c=c, maximize=true, row_lo=row_lo, row_hi=row_hi,
                  col_lo=col_lo, col_hi=col_hi, is_integer=fill(false, n),
                  row_names=row_names, col_names=col_names)
    return p
end

function make_schedule(seed::UInt32, periods::Int, units::Int)
    rng = LCG(seed)
    T = periods; U = units
    n_x = T * U
    n = n_x + n_x
    c = zeros(n)
    col_lo = zeros(n)
    col_hi = ones(n)
    is_integer = fill(true, n)
    col_names = Vector{String}(undef, n)
    xid(t, u) = (t - 1) * U + u
    sid(t, u) = n_x + (t - 1) * U + u
    for t in 1:T, u in 1:U
        col_names[xid(t, u)] = "X_T" * string(t) * "U" * string(u)
        col_names[sid(t, u)] = "S_T" * string(t) * "U" * string(u)
        c[xid(t, u)] = unif(rng, 10.0, 40.0)
        c[sid(t, u)] = 5.0 * unif(rng, 10.0, 40.0)
    end
    rows = Index[]; cols = Index[]; vals = Float64[]
    row_lo = Float64[]; row_hi = Float64[]; row_names = String[]
    function add_row(name, lo, hi)
        push!(row_names, name); push!(row_lo, lo); push!(row_hi, hi)
        return Index(length(row_names))
    end
    for t in 1:T
        r = add_row("DEM_T" * string(t), unif(rng, 1.0, Float64(U) * 0.7), kInf)
        for u in 1:U
            push!(rows, r); push!(cols, Index(xid(t, u))); push!(vals, 1.0)
        end
    end
    for t in 1:T, u in 1:U
        r = add_row("SU_T" * string(t) * "U" * string(u), -kInf, 0.0)
        push!(rows, r); push!(cols, Index(sid(t, u))); push!(vals, -1.0)
        push!(rows, r); push!(cols, Index(xid(t, u))); push!(vals, 1.0)
        if t > 1
            push!(rows, r); push!(cols, Index(xid(t - 1, u))); push!(vals, -1.0)
        end
    end
    return LpProblem(; name="SCHED_S" * string(Int(seed)),
                     A=from_triplets(length(row_lo), n, rows, cols, vals),
                     c=c, maximize=false, row_lo=row_lo, row_hi=row_hi,
                     col_lo=col_lo, col_hi=col_hi, is_integer=is_integer,
                     row_names=row_names, col_names=col_names)
end

function make_dispatch(seed::UInt32, n_gens::Int)
    rng = LCG(seed)
    n = n_gens
    c = zeros(n)
    q_diag = zeros(n)
    col_lo = zeros(n)
    col_hi = zeros(n)
    col_names = Vector{String}(undef, n)
    demand = 0.0
    for j in 1:n
        pmax = unif(rng, 50.0, 200.0)
        col_hi[j] = pmax
        col_lo[j] = 0.1 * pmax
        c[j] = unif(rng, 2.0, 20.0)
        q_diag[j] = unif(rng, 0.05, 0.4)
        col_names[j] = "G" * string(j)
        demand += 0.55 * pmax
    end
    rows = fill(Index(1), n)
    cols = Index[Index(j) for j in 1:n]
    vals = ones(n)
    lp = LpProblem(; name="DISPATCH_S" * string(Int(seed)),
                   A=from_triplets(1, n, rows, cols, vals),
                   c=c, maximize=false,
                   row_lo=[demand], row_hi=[demand],
                   col_lo=col_lo, col_hi=col_hi, is_integer=fill(false, n),
                   row_names=["DEMAND"], col_names=col_names)
    return lp, q_diag
end

function write_manifest_line(io::IO, kind, seed, path, extra; last::Bool=false)
    print(io, "  {\"kind\": \"", kind, "\", \"seed\": ", Int(seed),
          ", \"path\": \"", path, "\"", extra, "}")
    println(io, last ? "" : ",")
end

function sor_gen_main(args)::Cint
    if isempty(args)
        usage()
        return Cint(2)
    end
    cmd = args[1]
    seed = UInt32(42)
    out_path = ""
    outdir = "examples"
    n_crudes = 4
    n_products = 3
    periods = 6
    units = 2
    n_gens = 4
    i = 2
    while i <= length(args)
        a = args[i]
        function next_arg(w)
            if i + 1 > length(args)
                @printf(stderr, "error: %s needs a value\n", w)
                return nothing
            end
            i += 1
            return args[i]
        end
        if a == "--seed"
            v = next_arg("--seed"); v === nothing && return Cint(2)
            p = parse_u32(v, "seed"); p === nothing && return Cint(2)
            seed = p
        elseif a == "-o"
            v = next_arg("-o"); v === nothing && return Cint(2)
            out_path = v
        elseif a == "--outdir"
            v = next_arg("--outdir"); v === nothing && return Cint(2)
            outdir = v
        elseif a == "--crudes"
            v = next_arg("--crudes"); v === nothing && return Cint(2)
            p = parse_u32(v, "crudes"); p === nothing && return Cint(2)
            n_crudes = Int(p)
        elseif a == "--products"
            v = next_arg("--products"); v === nothing && return Cint(2)
            p = parse_u32(v, "products"); p === nothing && return Cint(2)
            n_products = Int(p)
        elseif a == "--periods"
            v = next_arg("--periods"); v === nothing && return Cint(2)
            p = parse_u32(v, "periods"); p === nothing && return Cint(2)
            periods = Int(p)
        elseif a == "--units"
            v = next_arg("--units"); v === nothing && return Cint(2)
            p = parse_u32(v, "units"); p === nothing && return Cint(2)
            units = Int(p)
        elseif a == "--gens"
            v = next_arg("--gens"); v === nothing && return Cint(2)
            p = parse_u32(v, "gens"); p === nothing && return Cint(2)
            n_gens = Int(p)
        elseif a == "-h" || a == "--help"
            usage()
            return Cint(0)
        else
            @printf(stderr, "error: unknown option '%s'\n", a)
            return Cint(2)
        end
        i += 1
    end

    try
        if cmd == "blend"
            isempty(out_path) && (out_path = "blend_s" * string(Int(seed)) * ".mps")
            p = make_blend(seed, n_crudes, n_products)
            write_mps_file(out_path, p)
            @printf("wrote %s  (%d x %d)\n", out_path, Int(n_rows(p)), Int(n_cols(p)))
            return Cint(0)
        elseif cmd == "schedule"
            isempty(out_path) && (out_path = "schedule_s" * string(Int(seed)) * ".mps")
            p = make_schedule(seed, periods, units)
            write_mps_file(out_path, p)
            @printf("wrote %s  (%d x %d, %d integer)\n", out_path,
                    Int(n_rows(p)), Int(n_cols(p)), Int(n_integer(p)))
            return Cint(0)
        elseif cmd == "dispatch"
            isempty(out_path) && (out_path = "dispatch_s" * string(Int(seed)) * ".qps")
            lp, q = make_dispatch(seed, n_gens)
            write_qps_file(out_path, lp, q)
            @printf("wrote %s  (%d gens)\n", out_path, n_gens)
            return Cint(0)
        elseif cmd == "all"
            mkpath(outdir)
            mkpath(joinpath(outdir, "crude_blending"))
            mkpath(joinpath(outdir, "scheduling"))
            mkpath(joinpath(outdir, "dispatch"))
            man_path = joinpath(outdir, "MANIFEST.json")
            open(man_path, "w") do man
                println(man, "{")
                println(man, "  \"generator\": \"sor_gen\",")
                print(man, "  \"seed\": "); print(man, Int(seed)); println(man, ",")
                println(man, "  \"instances\": [")
                p = make_blend(seed, n_crudes, n_products)
                path = joinpath(outdir, "crude_blending", "blend_s" * string(Int(seed)) * ".mps")
                write_mps_file(path, p)
                extra = ", \"crudes\": " * string(n_crudes) * ", \"products\": " * string(n_products)
                write_manifest_line(man, "blend_lp", seed, path, extra)
                println("wrote ", path)
                p = make_schedule(seed, periods, units)
                path = joinpath(outdir, "scheduling", "schedule_s" * string(Int(seed)) * ".mps")
                write_mps_file(path, p)
                extra = ", \"periods\": " * string(periods) * ", \"units\": " * string(units)
                write_manifest_line(man, "schedule_milp", seed, path, extra)
                println("wrote ", path)
                lp, q = make_dispatch(seed, n_gens)
                path = joinpath(outdir, "dispatch", "dispatch_s" * string(Int(seed)) * ".qps")
                write_qps_file(path, lp, q)
                extra = ", \"gens\": " * string(n_gens)
                write_manifest_line(man, "dispatch_qp", seed, path, extra; last=true)
                println("wrote ", path)
                println(man, "  ]")
                println(man, "}")
            end
            println("wrote ", man_path)
            return Cint(0)
        elseif cmd == "-h" || cmd == "--help"
            usage()
            return Cint(0)
        else
            @printf(stderr, "error: unknown command '%s'\n", cmd)
            usage()
            return Cint(2)
        end
    catch e
        @printf(stderr, "error: %s\n", sprint(showerror, e))
        return Cint(1)
    end
end

end # module
