# Length-prefixed JSON server over stdin/stdout.
#
# Framing: each message is a decimal byte-count on its own line, then exactly
# that many bytes of JSON.
#
#   17\n{"op":"ping","id":1}
#
# stdout carries ONLY protocol traffic. Every log line goes to stderr, so a stray
# @info can never corrupt the stream.

const READY_MARKER = "SORGPU_READY"

function write_message(io::IO, payload::Dict)
    body = JSON3.write(payload)
    bytes = Vector{UInt8}(body)
    print(io, length(bytes), '\n')
    write(io, bytes)
    flush(io)
    return nothing
end

"""
    read_message(io) -> Dict-like or nothing

Returns `nothing` on clean EOF. Throws on a malformed frame, which the caller
treats as fatal: a desynchronised stream cannot be recovered from.
"""
function read_message(io::IO)
    line = readline(io; keep = false)
    isempty(line) && eof(io) && return nothing
    isempty(line) && return read_message(io)          # tolerate blank lines
    n = tryparse(Int, strip(line))
    n === nothing && error("bad frame header: $(repr(line))")
    n < 0 && error("negative frame length $n")
    body = read(io, n)
    length(body) == n ||
        error("short frame: wanted $n bytes, got $(length(body))")
    return JSON3.read(String(body))
end

"""
    serve(; device="auto", warmup=true, input=stdin, output=stdout)

Long-lived request loop. Prints startup phase timings then
`SORGPU_READY <device>` to stderr so the C++ side can wait for startup instead
of racing it (Julia's JIT warmup is seconds, not milliseconds).

Pass `warmup=false` (CLI `--no-warmup`) only when measuring first-compile cost
via `bench_overhead`. Production always warms: that is the chosen path after
the C++→Julia overhead bench (pay JIT once at READY, not on the first solve).
"""
function serve(; device::AbstractString = "auto",
               warmup::Bool = true,
               input::IO = stdin, output::IO = stdout)
    t_boot = time_ns()
    local select_ms
    local dev
    select_ms = @elapsed(dev = select_device(device)) * 1e3
    state = ServerState(dev)

    warmup_ms = 0.0
    if warmup
        warmup_ms = @elapsed(_warmup!(state)) * 1e3
    end
    ready_ms = (time_ns() - t_boot) / 1.0e6

    # Machine-readable phases for the C++ overhead bench (stderr only).
    println(stderr, "SORGPU_TIMING " * JSON3.write(Dict(
        "select_device_ms" => select_ms,
        "warmup_ms" => warmup_ms,
        "ready_ms" => ready_ms,
        "warmup_enabled" => warmup,
        "device" => dev.name,
        "accelerated" => dev.accelerated,
    )))
    println(stderr, "$READY_MARKER $(dev.name) accelerated=$(dev.accelerated) " *
                    "julia=$(VERSION) threads=$(Threads.nthreads())")
    flush(stderr)

    while true
        req = try
            read_message(input)
        catch err
            println(stderr, "fatal: $(sprint(showerror, err))")
            return 1
        end
        req === nothing && break

        resp, keep = handle(state, req)
        write_message(output, resp)
        keep || break
    end
    return 0
end

"""
    _warmup!(state)

Production path: compile every KernelAbstractions kernel AND specialize the
JSON `handle` dispatch before `SORGPU_READY`. That way the C++ caller's first
real RPC is not paying ~2s of Julia method specialize on top of kernel JIT.

`--no-warmup` is measurement-only (see `bench_overhead`).
"""
function _warmup!(state::ServerState)
    try
        # 2x2 identity-ish CSR — kernel compile
        pat, _, _ = build_pattern(state.dev, 2, 2, [0, 1, 2], [0, 1])
        v, _, _   = to_device(state.dev, [1.0, 1.0])
        tv        = dev_zeros(state.dev, Float64, 2)
        x, _, _   = to_device(state.dev, [1.0, 2.0])
        y         = dev_zeros(state.dev, Float64, 2)
        spmv!(state.dev, y, pat, v, x)
        spmv_t!(state.dev, y, pat, v, x, tv)
        lo, _, _ = to_device(state.dev, [-1.0, -1.0])
        hi, _, _ = to_device(state.dev, [1.0, 1.0])
        project_box!(state.dev, x, lo, hi)
        device_dot(state.dev, x, x)
        X, _, _ = to_device(state.dev, [1.0, 2.0, 3.0, 4.0])
        Y = dev_zeros(state.dev, Float64, 4)
        spmv_batched!(state.dev, Y, pat, v, X, 2)

        # Protocol specialize — same ops the C++ backend hits first.
        # Responses are discarded; this never touches stdin/stdout.
        handle(state, Dict{String,Any}("op" => "ping", "id" => 0))
        handle(state, Dict{String,Any}(
            "op" => "upload_pattern", "id" => 0,
            "n_rows" => 2, "n_cols" => 2,
            "row_ptr" => [0, 1, 2], "col_idx" => [0, 1]))
        handle(state, Dict{String,Any}(
            "op" => "upload_vals", "id" => 0,
            "pattern_id" => 1, "vals" => [1.0, 1.0]))
        handle(state, Dict{String,Any}(
            "op" => "spmv", "id" => 0,
            "pattern_id" => 1, "vals_id" => 1, "x" => [1.0, 1.0]))
        handle(state, Dict{String,Any}(
            "op" => "spmv_t", "id" => 0,
            "pattern_id" => 1, "vals_id" => 1, "x" => [1.0, 1.0]))
        handle(state, Dict{String,Any}(
            "op" => "project_box", "id" => 0,
            "x" => [0.5, 0.5], "lo" => [0.0, 0.0], "hi" => [1.0, 1.0]))
        handle(state, Dict{String,Any}(
            "op" => "dot", "id" => 0,
            "a" => [1.0, 2.0], "b" => [3.0, 4.0]))
        handle(state, Dict{String,Any}(
            "op" => "free", "id" => 0,
            "pattern_id" => 1, "vals_id" => 1))

        # Frame encode/decode specialize (this is what made first ping ~2s).
        buf_out = IOBuffer()
        write_message(buf_out, Dict{String,Any}("ok" => true, "pong" => true, "id" => 0))
        frame = take!(buf_out)
        buf_in = IOBuffer(frame)
        read_message(buf_in)
        # End-to-end: write a ping frame, read it, handle, write response.
        req_buf = IOBuffer()
        write_message(req_buf, Dict{String,Any}("op" => "ping", "id" => 0))
        req = read_message(IOBuffer(take!(req_buf)))
        resp, _ = handle(state, req)
        write_message(IOBuffer(), resp)
    catch err
        println(stderr, "warmup failed (continuing): $(sprint(showerror, err))")
    end
    return nothing
end
