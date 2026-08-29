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
    serve(; device="auto", input=stdin, output=stdout)

Long-lived request loop. Prints `SORGPU_READY <device>` to stderr once the
backend is selected so the C++ side can wait for startup instead of racing it
(Julia's JIT warmup is seconds, not milliseconds).
"""
function serve(; device::AbstractString = "auto",
               input::IO = stdin, output::IO = stdout)
    dev = select_device(device)
    state = ServerState(dev)

    # Warm up the JIT so the first real request is not paying compile time and
    # skewing the timing table.
    _warmup!(state)

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

Runs one tiny instance of every kernel so KernelAbstractions compiles them
before any measured call. Failures are logged, never fatal.
"""
function _warmup!(state::ServerState)
    try
        # 2x2 identity-ish CSR
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
    catch err
        println(stderr, "warmup failed (continuing): $(sprint(showerror, err))")
    end
    return nothing
end
