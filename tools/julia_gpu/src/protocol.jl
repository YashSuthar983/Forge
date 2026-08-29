# JSON request/response handling.
#
# DEVIATION from docs/prompts/julia_gpu_prototype.md's v0 protocol, deliberate:
# the prompt sends row_ptr/col_idx/vals inline on EVERY call. PDHG performs two
# SpMVs per iteration over thousands of iterations, so an inline pattern would
# make JSON serialisation dominate the measurement completely and the timing
# table meaningless.
#
# Instead the pattern crosses the boundary ONCE via `upload_pattern` and is
# referenced by handle. Values get their own handle because `vals` is constant
# across a PDHG solve while the vectors change every iteration. The one-time
# upload cost is still charged to h2d_bytes/h2d_ms, so nothing is hidden.

const PROTOCOL_VERSION = 1

mutable struct ServerState
    dev::DeviceState
    patterns::Dict{Int,DevicePattern}
    vals::Dict{Int,Any}
    tvals::Dict{Int,Any}          # scratch for spmv_t, per vals handle
    next_pattern_id::Int
    next_vals_id::Int
end

ServerState(dev::DeviceState) =
    ServerState(dev, Dict{Int,DevicePattern}(), Dict{Int,Any}(), Dict{Int,Any}(),
                1, 1)

# --- JSON extraction helpers -------------------------------------------------

getint(o, k) = Int(o[k])
getfloat(o, k) = Float64(o[k])

function getvec_f64(o, k)::Vector{Float64}
    haskey(o, k) || error("missing field '$k'")
    return Float64[Float64(v) for v in o[k]]
end

function getvec_int(o, k)::Vector{Int}
    haskey(o, k) || error("missing field '$k'")
    return Int[Int(v) for v in o[k]]
end

ok_response(id, extra::Dict, st::Stats) =
    merge(Dict{String,Any}("id" => id, "ok" => true, "error" => nothing,
                           "stats" => as_dict(st)), extra)

err_response(id, msg::AbstractString) =
    Dict{String,Any}("id" => id, "ok" => false, "error" => String(msg),
                     "stats" => as_dict(Stats()))

# --- operation dispatch ------------------------------------------------------

function pattern_of(s::ServerState, o)
    pid = getint(o, "pattern_id")
    haskey(s.patterns, pid) || error("unknown pattern_id $pid")
    return pid, s.patterns[pid]
end

function vals_of(s::ServerState, o)
    vid = getint(o, "vals_id")
    haskey(s.vals, vid) || error("unknown vals_id $vid")
    return vid, s.vals[vid]
end

"""
    handle(state, request) -> (response::Dict, keep_running::Bool)

Never throws: every failure becomes `ok=false` with a message, so the C++ side
sees an explicit error instead of a dead pipe.
"""
function handle(s::ServerState, o)
    id = haskey(o, "id") ? getint(o, "id") : 0
    op = haskey(o, "op") ? String(o["op"]) : ""

    try
        if op == "ping"
            return ok_response(id, merge(Dict{String,Any}("pong" => true,
                                         "protocol" => PROTOCOL_VERSION),
                                         device_info(s.dev)), Stats()), true

        elseif op == "shutdown"
            return ok_response(id, Dict{String,Any}("bye" => true), Stats()), false

        elseif op == "upload_pattern"
            st = Stats()
            pat, ms, bytes = build_pattern(s.dev, getint(o, "n_rows"),
                                           getint(o, "n_cols"),
                                           getvec_int(o, "row_ptr"),
                                           getvec_int(o, "col_idx"))
            st.h2d_ms = ms
            st.h2d_bytes = bytes
            pid = s.next_pattern_id
            s.next_pattern_id += 1
            s.patterns[pid] = pat
            return ok_response(id, Dict{String,Any}("pattern_id" => pid,
                                                    "nnz" => pat.nnz), st), true

        elseif op == "upload_vals"
            st = Stats()
            _, pat = pattern_of(s, o)
            v = getvec_f64(o, "vals")
            length(v) == pat.nnz ||
                error("upload_vals: |vals| ($(length(v))) != nnz ($(pat.nnz))")
            d, ms, bytes = to_device(s.dev, v)
            st.h2d_ms = ms
            st.h2d_bytes = bytes
            vid = s.next_vals_id
            s.next_vals_id += 1
            s.vals[vid] = d
            s.tvals[vid] = dev_zeros(s.dev, Float64, pat.nnz)
            return ok_response(id, Dict{String,Any}("vals_id" => vid), st), true

        elseif op == "spmv" || op == "spmv_t"
            st = Stats()
            _, pat = pattern_of(s, o)
            vid, v = vals_of(s, o)
            xh = getvec_f64(o, "x")
            want = op == "spmv" ? pat.n_cols : pat.n_rows
            length(xh) == want ||
                error("$op: |x| ($(length(xh))) != $want")
            x, ms_in, b_in = to_device(s.dev, xh)
            st.h2d_ms += ms_in
            st.h2d_bytes += b_in

            nout = op == "spmv" ? pat.n_rows : pat.n_cols
            y = dev_zeros(s.dev, Float64, nout)
            _, kms = @timed_ms begin
                if op == "spmv"
                    spmv!(s.dev, y, pat, v, x)
                else
                    spmv_t!(s.dev, y, pat, v, x, s.tvals[vid])
                end
            end
            st.kernel_ms = kms

            yh, ms_out, b_out = to_host(s.dev, y)
            st.d2h_ms += ms_out
            st.d2h_bytes += b_out
            return ok_response(id, Dict{String,Any}("y" => yh), st), true

        elseif op == "project_box"
            st = Stats()
            xh  = getvec_f64(o, "x")
            loh = getvec_f64(o, "lo")
            hih = getvec_f64(o, "hi")
            x,  m1, b1 = to_device(s.dev, xh)
            lo, m2, b2 = to_device(s.dev, loh)
            hi, m3, b3 = to_device(s.dev, hih)
            st.h2d_ms = m1 + m2 + m3
            st.h2d_bytes = b1 + b2 + b3
            _, kms = @timed_ms project_box!(s.dev, x, lo, hi)
            st.kernel_ms = kms
            xr, ms_out, b_out = to_host(s.dev, x)
            st.d2h_ms = ms_out
            st.d2h_bytes = b_out
            return ok_response(id, Dict{String,Any}("x" => xr), st), true

        elseif op == "dot"
            st = Stats()
            ah = getvec_f64(o, "a")
            bh = getvec_f64(o, "b")
            a, m1, b1 = to_device(s.dev, ah)
            b, m2, b2 = to_device(s.dev, bh)
            st.h2d_ms = m1 + m2
            st.h2d_bytes = b1 + b2
            val, kms = @timed_ms device_dot(s.dev, a, b)
            st.kernel_ms = kms
            return ok_response(id, Dict{String,Any}("value" => val), st), true

        elseif op == "spmv_batched"
            st = Stats()
            _, pat = pattern_of(s, o)
            _, v = vals_of(s, o)
            n_items = getint(o, "n_items")
            Xh = getvec_f64(o, "X")
            X, m1, b1 = to_device(s.dev, Xh)
            st.h2d_ms = m1
            st.h2d_bytes = b1
            Y = dev_zeros(s.dev, Float64, n_items * pat.n_rows)
            _, kms = @timed_ms spmv_batched!(s.dev, Y, pat, v, X, n_items)
            st.kernel_ms = kms
            Yh, ms_out, b_out = to_host(s.dev, Y)
            st.d2h_ms = ms_out
            st.d2h_bytes = b_out
            return ok_response(id, Dict{String,Any}("Y" => Yh), st), true

        elseif op == "project_box_batched"
            st = Stats()
            Xh  = getvec_f64(o, "X")
            LOh = getvec_f64(o, "LO")
            HIh = getvec_f64(o, "HI")
            X,  m1, b1 = to_device(s.dev, Xh)
            LO, m2, b2 = to_device(s.dev, LOh)
            HI, m3, b3 = to_device(s.dev, HIh)
            st.h2d_ms = m1 + m2 + m3
            st.h2d_bytes = b1 + b2 + b3
            _, kms = @timed_ms project_box_batched!(s.dev, X, LO, HI)
            st.kernel_ms = kms
            Xr, ms_out, b_out = to_host(s.dev, X)
            st.d2h_ms = ms_out
            st.d2h_bytes = b_out
            return ok_response(id, Dict{String,Any}("X" => Xr), st), true

        elseif op == "free"
            # Release handles so a long-lived server does not leak device memory.
            if haskey(o, "pattern_id")
                delete!(s.patterns, getint(o, "pattern_id"))
            end
            if haskey(o, "vals_id")
                vid = getint(o, "vals_id")
                delete!(s.vals, vid)
                delete!(s.tvals, vid)
            end
            return ok_response(id, Dict{String,Any}("freed" => true), Stats()), true

        else
            return err_response(id, "unknown op '$op'"), true
        end
    catch err
        return err_response(id, sprint(showerror, err)), true
    end
end
