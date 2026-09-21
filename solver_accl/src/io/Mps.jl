# SOR — MPS / QPS / solution I/O (port of sor/sor_io).
#
# LAYER L2. Written from the published MPS format description (IBM MPSX
# convention as documented by Netlib and the MIPLIB/QPLIB format notes).
# No reader source from any solver was consulted.
#
# Include this file from SovereignSolver after SorCore / SorSparse / SorModel
# so `using ..X` resolves. Gzipped `.mps.gz` is not decoded here: Project.toml
# has no CodecZlib/TranscodingStreams, and those must not be added just for I/O.
module SorIO

using ..SorCore: Index, Offset, SolveResult, Status, ProofLevel, to_string, kNaN, kPosInf
using ..SorCore: NotSolved, Optimal, Infeasible, Unbounded, InfeasibleOrUnbounded
using ..SorCore: Feasible, NoSolutionFound, Interrupted, NumericalFailure, Unsupported
using ..SorCore: None, BoundOnly, FeasibleOnly, FeasibleWithGap, ProvedKKT
using ..SorCore: ProvedGlobalEpsilon, ProvedOptimalFP, ProvedOptimalExact, ProvedOptimalCertified
using ..SorSparse: from_triplets, CsrMatrix, nnz, n_rows, n_cols
using ..SorModel: LpProblem, kInf, n_integer

export MpsReadOptions, MpsReadReport, QpsReadReport, QpsProblem, SolutionFile
export read_mps, read_mps_file, read_mps_file_auto
export write_mps, write_mps_file, write_qps, write_qps_file
export write_solution, read_solution, read_qps_file

@enum Section::Int32 begin
    SecNone
    SecName
    SecObjSense
    SecRows
    SecColumns
    SecRhs
    SecRanges
    SecBounds
    SecQuadobj
    SecEnd
end

@enum RowKind::Int32 begin
    RowObjective
    RowLessEqual
    RowGreaterEqual
    RowEqual
end

Base.@kwdef mutable struct MpsReadOptions
    fixed_format::Bool = false
    strict::Bool = false
end

Base.@kwdef mutable struct MpsReadReport
    n_rows::Int = 0
    n_cols::Int = 0
    n_integer::Int = 0
    nnz::Offset = Offset(0)
    had_ranges::Bool = false
    had_objsense_max::Bool = false
    used_fixed_format::Bool = false
    warnings::Vector{String} = String[]
end

Base.@kwdef mutable struct QpsReadReport
    n_rows::Int = 0
    n_cols::Int = 0
    n_integer::Int = 0
    nnz::Offset = Offset(0)
    had_ranges::Bool = false
    had_objsense_max::Bool = false
    used_fixed_format::Bool = false
    warnings::Vector{String} = String[]
    n_quad_entries::Int = 0
    has_off_diagonal::Bool = false
end

struct QpsProblem
    linear::LpProblem
    q_diag::Vector{Float64}
end

Base.@kwdef mutable struct SolutionFile
    status::Status = NotSolved
    proof::ProofLevel = None
    objective::Float64 = kNaN
    x::Vector{Float64} = Float64[]
    y::Vector{Float64} = Float64[]
    ray::Vector{Float64} = Float64[]
end

function _copy_mps_report!(dst, src::MpsReadReport)
    dst.n_rows = src.n_rows
    dst.n_cols = src.n_cols
    dst.n_integer = src.n_integer
    dst.nnz = src.nnz
    dst.had_ranges = src.had_ranges
    dst.had_objsense_max = src.had_objsense_max
    dst.used_fixed_format = src.used_fixed_format
    dst.warnings = copy(src.warnings)
    return dst
end

_upper(s::AbstractString) = uppercase(s)

function _split_ws(line::AbstractString)
    toks = String[]
    for tok in eachsplit(line, isspace; keepempty=false)
        push!(toks, String(tok))
    end
    return toks
end

# Fixed-format MPS field columns (1-based in the spec):
#   f1 2-3, f2 5-12, f3 15-22, f4 25-36, f5 40-47, f6 50-61
# C++ uses 0-based byte offsets {1,4,14,24,39,49} with lengths {2,8,8,12,8,12}.
function _split_fixed(line::AbstractString)
    begs0 = (1, 4, 14, 24, 39, 49)
    lens = (2, 8, 8, 12, 8, 12)
    bytes = codeunits(line)
    n = length(bytes)
    out = String[]
    for i in 1:6
        b0 = begs0[i]
        n <= b0 && break
        b = b0 + 1
        e = min(n, b + lens[i] - 1)
        f = String(@view bytes[b:e])
        stripped = strip(f)
        isempty(stripped) && continue
        push!(out, String(stripped))
    end
    return out
end

function _parse_num(s::AbstractString, line_no::Int)
    v = tryparse(Float64, s)
    v === nothing && error("MPS line $line_no: cannot parse number '$s'")
    return v
end

function _starts_with_max(s::AbstractString)
    return startswith(_upper(s), "MAX")
end

mutable struct _Builder
    row_kind::Vector{RowKind}
    row_names::Vector{String}
    row_of::Dict{String,Index}
    constraint_index::Vector{Index}  # 0 = objective slot, else 1-based constraint
    obj_slot::Index                  # 0 = none; else 1-based slot of first N row
    con_kind::Vector{RowKind}
    con_names::Vector{String}
    col_names::Vector{String}
    col_of::Dict{String,Index}
    tri_r::Vector{Index}
    tri_c::Vector{Index}
    tri_v::Vector{Float64}
    obj::Vector{Float64}
    obj_rhs::Float64
    rhs::Vector{Float64}
    ranges::Vector{Float64}
    has_range::Vector{Bool}
    col_lo::Vector{Float64}
    col_hi::Vector{Float64}
    lo_set::Vector{Bool}
    hi_set::Vector{Bool}
    integer_flag::Vector{Bool}
    _Builder() = new(RowKind[], String[], Dict{String,Index}(), Index[], Index(0),
                     RowKind[], String[], String[], Dict{String,Index}(),
                     Index[], Index[], Float64[], Float64[], 0.0,
                     Float64[], Float64[], Bool[], Float64[], Float64[],
                     Bool[], Bool[], Bool[])
end

function _ensure_col!(b::_Builder, n::AbstractString)
    j = get(b.col_of, n, Index(0))
    j != 0 && return j
    j = Index(length(b.col_names) + 1)
    b.col_of[String(n)] = j
    push!(b.col_names, String(n))
    push!(b.obj, 0.0)
    push!(b.col_lo, 0.0)
    push!(b.col_hi, kInf)
    push!(b.lo_set, false)
    push!(b.hi_set, false)
    push!(b.integer_flag, false)
    return j
end

function _reject_gzip(path::AbstractString)
    lp = lowercase(path)
    if endswith(lp, ".gz") || endswith(lp, ".mps.gz")
        error("gzipped MPS '$path' is not supported (no CodecZlib in Project.toml); decompress to a plain .mps file first")
    end
    return nothing
end

function read_mps(in::IO, report::MpsReadReport=MpsReadReport(),
                  opt::MpsReadOptions=MpsReadOptions())
    b = _Builder()
    sec = SecNone
    name = ""
    maximize = false
    in_integer_marker = false
    line_no = 0

    fields_of(l) = opt.fixed_format ? _split_fixed(l) : _split_ws(l)

    while !eof(in)
        line = readline(in; keep=false)
        line_no += 1
        if !isempty(line) && last(line) == '\r'
            line = chop(line)
        end
        isempty(line) && continue
        first(line) == '*' && continue

        is_header = !isspace(first(line))
        if is_header
            f = _split_ws(line)
            isempty(f) && continue
            key = _upper(f[1])
            if key == "NAME"
                sec = SecName
                length(f) > 1 && (name = f[2])
            elseif key == "OBJSENSE"
                sec = SecObjSense
                if length(f) > 1 && _starts_with_max(f[2])
                    maximize = true
                end
            elseif key == "OBJSENS"
                sec = SecObjSense
            elseif key == "ROWS"
                sec = SecRows
            elseif key == "COLUMNS"
                sec = SecColumns
            elseif key == "RHS"
                sec = SecRhs
            elseif key == "RANGES"
                sec = SecRanges
            elseif key == "BOUNDS"
                sec = SecBounds
            elseif key == "QUADOBJ" || key == "QMATRIX"
                if opt.strict
                    error("MPS line $line_no: QUADOBJ present; this reader is LP-only")
                end
                push!(report.warnings, "QUADOBJ section ignored (LP-only reader)")
                sec = SecQuadobj
            elseif key == "ENDATA"
                sec = SecEnd
                break
            else
                if opt.strict
                    error("MPS line $line_no: unknown section '$(f[1])'")
                end
                push!(report.warnings, "ignored unknown section '$(f[1])'")
                sec = SecNone
            end
            continue
        end

        f = fields_of(line)
        isempty(f) && continue

        if sec == SecObjSense
            if _starts_with_max(f[1])
                maximize = true
            end
        elseif sec == SecRows
            if length(f) < 2
                error("MPS line $line_no: ROWS entry needs kind and name")
            end
            k = _upper(f[1])
            kind = if k == "N"
                RowObjective
            elseif k == "L"
                RowLessEqual
            elseif k == "G"
                RowGreaterEqual
            elseif k == "E"
                RowEqual
            else
                error("MPS line $line_no: unknown row kind '$(f[1])'")
            end
            haskey(b.row_of, f[2]) &&
                error("MPS line $line_no: duplicate row name '$(f[2])'")
            slot = Index(length(b.row_kind) + 1)
            b.row_of[f[2]] = slot
            push!(b.row_kind, kind)
            push!(b.row_names, f[2])
            if kind == RowObjective
                push!(b.constraint_index, Index(0))
                # Only the first N row is the objective; later N rows are
                # free rows and their coefficients are discarded.
                if b.obj_slot == 0
                    b.obj_slot = slot
                end
            else
                push!(b.constraint_index, Index(length(b.rhs) + 1))
                push!(b.con_kind, kind)
                push!(b.con_names, f[2])
                push!(b.rhs, 0.0)
                push!(b.ranges, 0.0)
                push!(b.has_range, false)
            end
        elseif sec == SecColumns
            is_marker = any(t -> occursin("MARKER", _upper(t)), f)
            if is_marker
                for t in f
                    u = _upper(t)
                    occursin("INTORG", u) && (in_integer_marker = true)
                    occursin("INTEND", u) && (in_integer_marker = false)
                end
            elseif length(f) < 3
                error("MPS line $line_no: COLUMNS entry needs col, row, value")
            else
                j = _ensure_col!(b, f[1])
                if in_integer_marker
                    b.integer_flag[Int(j)] = true
                end
                k = 2
                while k + 1 <= length(f)
                    slot = get(b.row_of, f[k], Index(0))
                    slot == 0 && error("MPS line $line_no: unknown row '$(f[k])'")
                    v = _parse_num(f[k + 1], line_no)
                    if b.row_kind[Int(slot)] == RowObjective
                        if slot == b.obj_slot
                            b.obj[Int(j)] += v
                        end
                    else
                        push!(b.tri_r, b.constraint_index[Int(slot)])
                        push!(b.tri_c, j)
                        push!(b.tri_v, v)
                    end
                    k += 2
                end
            end
        elseif sec == SecRhs
            k = isodd(length(f)) ? 2 : 1
            while k + 1 <= length(f)
                slot = get(b.row_of, f[k], Index(0))
                slot == 0 && error("MPS line $line_no: unknown row '$(f[k])' in RHS")
                v = _parse_num(f[k + 1], line_no)
                if b.row_kind[Int(slot)] == RowObjective
                    # Convention: RHS on the objective row is the NEGATIVE
                    # of the objective constant.
                    if slot == b.obj_slot
                        b.obj_rhs = -v
                    end
                else
                    b.rhs[Int(b.constraint_index[Int(slot)])] = v
                end
                k += 2
            end
        elseif sec == SecRanges
            k = isodd(length(f)) ? 2 : 1
            while k + 1 <= length(f)
                slot = get(b.row_of, f[k], Index(0))
                slot == 0 && error("MPS line $line_no: unknown row '$(f[k])' in RANGES")
                if b.row_kind[Int(slot)] != RowObjective
                    ci = Int(b.constraint_index[Int(slot)])
                    b.ranges[ci] = _parse_num(f[k + 1], line_no)
                    b.has_range[ci] = true
                    report.had_ranges = true
                end
                k += 2
            end
        elseif sec == SecBounds
            if length(f) < 3
                error("MPS line $line_no: BOUNDS entry too short")
            end
            typ = _upper(f[1])
            # Layout is TYPE setname colname [value] but the set name is
            # sometimes omitted. Disambiguate by looking for a known column.
            ci = 3
            vi = 4
            if !haskey(b.col_of, f[3]) && haskey(b.col_of, f[2])
                ci = 2
                vi = 3
            end
            haskey(b.col_of, f[ci]) ||
                error("MPS line $line_no: unknown column '$(f[ci])' in BOUNDS")
            j = Int(b.col_of[f[ci]])
            needs_value = typ == "UP" || typ == "LO" || typ == "FX" ||
                          typ == "UI" || typ == "LI"
            if needs_value && vi > length(f)
                error("MPS line $line_no: bound type $typ needs a value")
            end
            v = needs_value ? _parse_num(f[vi], line_no) : 0.0
            if typ == "UP" || typ == "UI"
                b.col_hi[j] = v
                b.hi_set[j] = true
                # Negative UP with still-default lower bound implies lower = -inf.
                if v < 0.0 && !b.lo_set[j]
                    b.col_lo[j] = -kInf
                end
                typ == "UI" && (b.integer_flag[j] = true)
            elseif typ == "LO" || typ == "LI"
                b.col_lo[j] = v
                b.lo_set[j] = true
                typ == "LI" && (b.integer_flag[j] = true)
            elseif typ == "FX"
                b.col_lo[j] = b.col_hi[j] = v
                b.lo_set[j] = b.hi_set[j] = true
            elseif typ == "FR"
                b.col_lo[j] = -kInf
                b.col_hi[j] = kInf
                b.lo_set[j] = b.hi_set[j] = true
            elseif typ == "MI"
                b.col_lo[j] = -kInf
                b.lo_set[j] = true
            elseif typ == "PL"
                b.col_hi[j] = kInf
                b.hi_set[j] = true
            elseif typ == "BV"
                b.col_lo[j] = 0.0
                b.col_hi[j] = 1.0
                b.lo_set[j] = b.hi_set[j] = true
                b.integer_flag[j] = true
            else
                error("MPS line $line_no: unknown bound type '$(f[1])'")
            end
        end
    end

    if sec != SecEnd
        push!(report.warnings, "no ENDATA record found")
    end

    nc = Index(length(b.col_names))
    nr = Index(length(b.rhs))
    A = from_triplets(nr, nc, b.tri_r, b.tri_c, b.tri_v)
    row_lo = Vector{Float64}(undef, Int(nr))
    row_hi = Vector{Float64}(undef, Int(nr))
    for i in 1:Int(nr)
        kind = b.con_kind[i]
        r = b.rhs[i]
        if !b.has_range[i]
            if kind == RowLessEqual
                row_lo[i] = -kInf
                row_hi[i] = r
            elseif kind == RowGreaterEqual
                row_lo[i] = r
                row_hi[i] = kInf
            elseif kind == RowEqual
                row_lo[i] = r
                row_hi[i] = r
            end
        else
            R = b.ranges[i]
            a = abs(R)
            if kind == RowLessEqual
                row_lo[i] = r - a
                row_hi[i] = r
            elseif kind == RowGreaterEqual
                row_lo[i] = r
                row_hi[i] = r + a
            elseif kind == RowEqual
                if R >= 0.0
                    row_lo[i] = r
                    row_hi[i] = r + R
                else
                    row_lo[i] = r + R
                    row_hi[i] = r
                end
            end
        end
    end

    p = LpProblem(; name=name, A=A, c=b.obj, obj_offset=b.obj_rhs, maximize=maximize,
                  row_lo=row_lo, row_hi=row_hi, col_lo=b.col_lo, col_hi=b.col_hi,
                  is_integer=b.integer_flag, row_names=b.con_names, col_names=b.col_names)

    report.n_rows = Int(nr)
    report.n_cols = Int(nc)
    report.nnz = nnz(p.A)
    report.n_integer = n_integer(p)
    report.had_objsense_max = maximize
    return p
end

function read_mps_file(path::AbstractString, report::MpsReadReport=MpsReadReport(),
                       opt::MpsReadOptions=MpsReadOptions())
    _reject_gzip(path)
    isfile(path) || error("cannot open MPS file: $path")
    return open(path, "r") do io
        read_mps(io, report, opt)
    end
end

function read_mps_file_auto(path::AbstractString, report::MpsReadReport=MpsReadReport();
                            strict::Bool=false)
    free_opt = MpsReadOptions(fixed_format=false, strict=strict)
    try
        p = read_mps_file(path, report, free_opt)
        report.used_fixed_format = false
        return p
    catch first_err
        report2 = MpsReadReport()
        fixed_opt = MpsReadOptions(fixed_format=true, strict=strict)
        try
            p = read_mps_file(path, report2, fixed_opt)
            _copy_mps_report!(report, report2)
            report.used_fixed_format = true
            push!(report.warnings,
                  "free-format parse failed ($(sprint(showerror, first_err))); succeeded in fixed format")
            return p
        catch second_err
            error("MPS parse failed in both formats. free: $(sprint(showerror, first_err)) | fixed: $(sprint(showerror, second_err))")
        end
    end
end

function _name_or(names::Vector{String}, i::Integer, prefix::AbstractString)
    ii = Int(i)
    if 1 <= ii <= length(names) && !isempty(names[ii])
        return names[ii]
    end
    return string(prefix, ii)
end

function _write_body(out::IO, p::LpProblem)
    m = Int(n_rows(p.A))
    n = Int(n_cols(p.A))
    println(out, "NAME          ", isempty(p.name) ? "SOR" : p.name)
    if p.maximize
        println(out, "OBJSENSE")
        println(out, " MAX")
    end
    println(out, "ROWS")
    println(out, " N  COST")
    for i in 1:m
        lo = p.row_lo[i]
        hi = p.row_hi[i]
        kind = 'L'
        if lo == hi
            kind = 'E'
        elseif lo > -kInf && hi >= kInf
            kind = 'G'
        elseif lo <= -kInf && hi < kInf
            kind = 'L'
        else
            kind = 'E'
        end
        if lo > -kInf && hi < kInf && lo != hi
            kind = 'L'
        end
        println(out, " ", kind, "  ", _name_or(p.row_names, i, "R"))
    end

    println(out, "COLUMNS")
    rp = p.A.pattern.row_ptr
    ci = p.A.pattern.col_idx
    av = p.A.vals
    cols = [Vector{Tuple{Index,Float64}}() for _ in 1:n]
    for i in 1:m
        for k in Int(rp[i]):(Int(rp[i + 1]) - 1)
            push!(cols[Int(ci[k])], (Index(i), av[k]))
        end
    end

    in_int = false
    for j in 1:n
        is_int = !isempty(p.is_integer) && p.is_integer[j]
        if is_int && !in_int
            println(out, "    MARK0000  'MARKER'                 'INTORG'")
            in_int = true
        elseif !is_int && in_int
            println(out, "    MARK0001  'MARKER'                 'INTEND'")
            in_int = false
        end
        cj = _name_or(p.col_names, j, "X")
        pair = 0
        if p.c[j] != 0.0
            print(out, "    ", cj, "  COST  ", p.c[j])
            pair = 1
        end
        for (i, v) in cols[j]
            if pair == 0
                print(out, "    ", cj)
            end
            print(out, "  ", _name_or(p.row_names, Int(i), "R"), "  ", v)
            pair += 1
            if pair == 2
                println(out)
                pair = 0
            end
        end
        pair == 1 && println(out)
        if pair == 0 && isempty(cols[j]) && p.c[j] == 0.0
            println(out, "    ", cj, "  COST  0")
        end
    end
    in_int && println(out, "    MARK0001  'MARKER'                 'INTEND'")

    println(out, "RHS")
    if p.obj_offset != 0.0
        println(out, "    RHS       COST  ", -p.obj_offset)
    end
    for i in 1:m
        lo = p.row_lo[i]
        hi = p.row_hi[i]
        rhs = 0.0
        write = false
        if lo == hi
            rhs = lo
            write = true
        elseif hi < kInf
            rhs = hi
            write = true
        elseif lo > -kInf
            rhs = lo
            write = true
        end
        if write && rhs != 0.0
            println(out, "    RHS       ", _name_or(p.row_names, i, "R"), "  ", rhs)
        end
    end

    any_range = false
    for i in 1:m
        lo = p.row_lo[i]
        hi = p.row_hi[i]
        if lo > -kInf && hi < kInf && lo != hi
            if !any_range
                println(out, "RANGES")
                any_range = true
            end
            println(out, "    RNG       ", _name_or(p.row_names, i, "R"), "  ", hi - lo)
        end
    end

    println(out, "BOUNDS")
    for j in 1:n
        lo = p.col_lo[j]
        hi = p.col_hi[j]
        cj = _name_or(p.col_names, j, "X")
        is_int = !isempty(p.is_integer) && p.is_integer[j]
        if lo == hi
            println(out, " FX BND       ", cj, "  ", lo)
        elseif lo <= -kInf && hi >= kInf
            println(out, " FR BND       ", cj)
        else
            if lo != 0.0 && lo > -kInf
                println(out, is_int ? " LI" : " LO", " BND       ", cj, "  ", lo)
            end
            # Use UP for both (not UI): PuLP/CBC's MPS reader rejects UI.
            if hi < kInf
                println(out, " UP BND       ", cj, "  ", hi)
            end
            if lo <= -kInf && hi < kInf && !is_int
                println(out, " MI BND       ", cj)
            end
        end
    end
end

function write_mps(out::IO, p::LpProblem)
    _write_body(out, p)
    println(out, "ENDATA")
    return nothing
end

function write_mps_file(path::AbstractString, p::LpProblem)
    open(path, "w") do io
        write_mps(io, p)
    end
    return nothing
end

function write_qps(out::IO, p::LpProblem, q_diag::AbstractVector{<:Real})
    Int(length(q_diag)) == Int(n_cols(p.A)) ||
        throw(ArgumentError("write_qps: q_diag size != n_cols"))
    _write_body(out, p)
    println(out, "QUADOBJ")
    for j in 1:Int(n_cols(p.A))
        q_diag[j] == 0.0 && continue
        cj = _name_or(p.col_names, j, "X")
        println(out, "    ", cj, "  ", cj, "  ", q_diag[j])
    end
    println(out, "ENDATA")
    return nothing
end

function write_qps_file(path::AbstractString, p::LpProblem, q_diag::AbstractVector{<:Real})
    open(path, "w") do io
        write_qps(io, p, q_diag)
    end
    return nothing
end

function _write_f64(out::IO, v::Float64)
    if isnan(v)
        print(out, "nan")
    elseif v == kPosInf
        print(out, "inf")
    elseif v == -kPosInf
        print(out, "-inf")
    else
        print(out, v)
    end
    return nothing
end

function _read_f64_token(tok::AbstractString)
    if tok == "nan" || tok == "-nan"
        return kNaN
    elseif tok == "inf" || tok == "+inf"
        return kPosInf
    elseif tok == "-inf"
        return -kPosInf
    end
    v = tryparse(Float64, tok)
    v === nothing && error("solution file: not a number '$tok'")
    return v
end

function _status_from_string(s::AbstractString)
    s == "NotSolved" && return NotSolved
    s == "Optimal" && return Optimal
    s == "Infeasible" && return Infeasible
    s == "Unbounded" && return Unbounded
    s == "InfeasibleOrUnbounded" && return InfeasibleOrUnbounded
    s == "Feasible" && return Feasible
    s == "NoSolutionFound" && return NoSolutionFound
    s == "Interrupted" && return Interrupted
    s == "NumericalFailure" && return NumericalFailure
    s == "Unsupported" && return Unsupported
    error("solution file: unknown status '$s'")
end

function _proof_from_string(s::AbstractString)
    s == "None" && return None
    s == "BoundOnly" && return BoundOnly
    s == "FeasibleOnly" && return FeasibleOnly
    s == "FeasibleWithGap" && return FeasibleWithGap
    s == "ProvedKKT" && return ProvedKKT
    s == "ProvedGlobalEpsilon" && return ProvedGlobalEpsilon
    s == "ProvedOptimalFP" && return ProvedOptimalFP
    s == "ProvedOptimalExact" && return ProvedOptimalExact
    s == "ProvedOptimalCertified" && return ProvedOptimalCertified
    error("solution file: unknown proof level '$s'")
end

function _write_vec(out::IO, tag::AbstractString, v::Vector{Float64})
    print(out, tag, ' ', length(v))
    for x in v
        print(out, ' ')
        _write_f64(out, x)
    end
    println(out)
    return nothing
end

function write_solution(out::IO, r::SolveResult)
    println(out, "status ", to_string(r.status))
    println(out, "proof ", to_string(r.proof))
    print(out, "objective ")
    _write_f64(out, r.objective)
    println(out)
    _write_vec(out, "x", r.x)
    _write_vec(out, "y", r.y)
    _write_vec(out, "ray", r.ray)
    return nothing
end

function write_solution(path::AbstractString, r::SolveResult)
    open(path, "w") do io
        write_solution(io, r)
    end
    return nothing
end

function _take!(toks::Vector{SubString{String}}, i::Int, expect::AbstractString)
    i > length(toks) && error("solution file: expected '$expect' line")
    return toks[i], i + 1
end

function _read_vec_from!(toks::Vector{SubString{String}}, i::Int, expect_tag::AbstractString)
    tag, i = _take!(toks, i, expect_tag)
    String(tag) != expect_tag && error("solution file: expected '$expect_tag' line")
    i > length(toks) && error("solution file: expected '$expect_tag' line")
    n = tryparse(Int, toks[i])
    n === nothing && error("solution file: expected '$expect_tag' line")
    i += 1
    v = Vector{Float64}(undef, n)
    for k in 1:n
        if i > length(toks)
            error("solution file: '$expect_tag' declared $n values but fewer were present")
        end
        v[k] = _read_f64_token(toks[i])
        i += 1
    end
    return v, i
end

function read_solution(in::IO)
    toks = split(read(in, String), isspace; keepempty=false)
    i = 1
    tag, i = _take!(toks, i, "status")
    String(tag) != "status" && error("solution file: expected 'status' line")
    val, i = _take!(toks, i, "status")
    s = SolutionFile()
    s.status = _status_from_string(val)
    tag, i = _take!(toks, i, "proof")
    String(tag) != "proof" && error("solution file: expected 'proof' line")
    val, i = _take!(toks, i, "proof")
    s.proof = _proof_from_string(val)
    tag, i = _take!(toks, i, "objective")
    String(tag) != "objective" && error("solution file: expected 'objective' line")
    val, i = _take!(toks, i, "objective")
    s.objective = _read_f64_token(val)
    s.x, i = _read_vec_from!(toks, i, "x")
    s.y, i = _read_vec_from!(toks, i, "y")
    s.ray, i = _read_vec_from!(toks, i, "ray")
    return s
end

function read_qps_file(path::AbstractString, report::QpsReadReport=QpsReadReport(),
                       opt::MpsReadOptions=MpsReadOptions())
    mrep = MpsReadReport()
    linear = read_mps_file_auto(path, mrep; strict=opt.strict)
    _copy_mps_report!(report, mrep)
    filter!(w -> !(occursin("QUADOBJ", w) && occursin("ignored", w)), report.warnings)
    q_diag = zeros(Float64, Int(n_cols(linear.A)))

    col_of = Dict{String,Index}()
    nc = Int(n_cols(linear.A))
    for j in 1:nc
        if j <= length(linear.col_names) && !isempty(linear.col_names[j])
            col_of[linear.col_names[j]] = Index(j)
        else
            col_of["X$j"] = Index(j)
        end
    end

    in_quad = false
    line_no = 0
    open(path, "r") do io
        while !eof(io)
            line = readline(io; keep=false)
            line_no += 1
            if !isempty(line) && last(line) == '\r'
                line = chop(line)
            end
            isempty(line) && continue
            header = !isspace(first(line))
            if header
                f = _split_ws(line)
                isempty(f) && continue
                key = _upper(f[1])
                if key == "QUADOBJ" || key == "QMATRIX" || key == "QSECTION"
                    in_quad = true
                    continue
                end
                key == "ENDATA" && break
                in_quad && (in_quad = false)
                continue
            end
            in_quad || continue
            f = _split_ws(line)
            length(f) < 3 && continue
            k = length(f) >= 4 ? 2 : 1
            if k + 2 <= length(f)
                c1 = f[k]
                c2 = f[k + 1]
                v = something(tryparse(Float64, f[k + 2]), 0.0)
                i1 = get(col_of, c1, Index(0))
                i2 = get(col_of, c2, Index(0))
                if i1 == 0 || i2 == 0
                    push!(report.warnings, "QUADOBJ unknown column at line $line_no")
                else
                    report.n_quad_entries += 1
                    if i1 != i2
                        report.has_off_diagonal = true
                        push!(report.warnings,
                              "QUADOBJ off-diagonal entry ignored by diagonal QP engine")
                    else
                        q_diag[Int(i1)] += v
                    end
                end
            end
        end
    end
    return QpsProblem(linear, q_diag)
end

end # module
