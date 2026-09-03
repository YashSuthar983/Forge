# Phase timings for JIT / KernelAbstractions first-compile vs warm execute.
#
# Used by the `bench_overhead` protocol op and by bench_overhead.jl so a C++
# caller can separate: Julia boot, package load, KA compile, warm kernel, IPC.

"""
    synthetic_csr(n_rows, n_cols; nnz_per_row=8) -> (row_ptr0, col_idx0, vals)

0-based CSR with a fixed, reproducible pattern. Small enough that launch
overhead dominates arithmetic — which is what we want for an overhead bench.
"""
function synthetic_csr(n_rows::Int, n_cols::Int; nnz_per_row::Int = 8)
    n_rows >= 1 && n_cols >= 1 || error("n_rows/n_cols must be >= 1")
    k = min(nnz_per_row, n_cols)
    row_ptr = Vector{Int}(undef, n_rows + 1)
    col_idx = Vector{Int}(undef, n_rows * k)
    vals = Vector{Float64}(undef, n_rows * k)
    row_ptr[1] = 0
    @inbounds for r in 1:n_rows
        base = (r - 1) * k
        for j in 1:k
            col_idx[base + j] = (r + j - 2) % n_cols
            vals[base + j] = 1.0 + 0.01 * ((r + j) % 17)
        end
        row_ptr[r + 1] = base + k
    end
    return row_ptr, col_idx, vals
end

"""
    measure_ka_spmv(dev; n_rows, n_cols, warm_iters) -> Dict

Times KA SpMV first launch (compile + execute) vs subsequent launches.
Must be called on a process that has NOT already compiled `spmv_kernel!` for
this backend, or `first_ms` collapses to execute-only.
"""
function measure_ka_spmv(dev::DeviceState;
                         n_rows::Int = 4096,
                         n_cols::Int = 4096,
                         nnz_per_row::Int = 8,
                         warm_iters::Int = 20)
    rp, ci, vals_h = synthetic_csr(n_rows, n_cols; nnz_per_row = nnz_per_row)
    pat, upload_pat_ms, upload_pat_bytes = build_pattern(dev, n_rows, n_cols, rp, ci)
    v, upload_vals_ms, upload_vals_bytes = to_device(dev, vals_h)
    xh = ones(Float64, n_cols)
    x, upload_x_ms, upload_x_bytes = to_device(dev, xh)
    y = dev_zeros(dev, Float64, n_rows)

    _, first_ms = @timed_ms spmv!(dev, y, pat, v, x)

    warm = Float64[]
    for _ in 1:warm_iters
        _, ms = @timed_ms spmv!(dev, y, pat, v, x)
        push!(warm, ms)
    end
    sort!(warm)
    # Trim one outlier on each side when we have enough samples.
    core = length(warm) >= 5 ? warm[2:end-1] : warm

    return Dict{String,Any}(
        "kind" => "ka_spmv",
        "n_rows" => n_rows,
        "n_cols" => n_cols,
        "nnz" => length(vals_h),
        "upload_pattern_ms" => upload_pat_ms,
        "upload_vals_ms" => upload_vals_ms,
        "upload_x_ms" => upload_x_ms,
        "upload_bytes" => upload_pat_bytes + upload_vals_bytes + upload_x_bytes,
        "first_kernel_ms" => first_ms,
        "warm_iters" => warm_iters,
        "warm_mean_ms" => sum(core) / length(core),
        "warm_median_ms" => core[(length(core) + 1) ÷ 2],
        "warm_min_ms" => minimum(core),
        "compile_proxy_ms" => max(0.0, first_ms - (sum(core) / length(core))),
    )
end

"""
    measure_ak_map!(dev; n, warm_iters) -> Dict

AcceleratedKernels foreachindex first-compile vs warm. Package is a declared
dependency of SORGpuProto (imported as `AK`).
"""
function measure_ak_map!(dev::DeviceState; n::Int = 1 << 20, warm_iters::Int = 20)
    x = KernelAbstractions.ones(dev.backend, Float64, n)
    y = KernelAbstractions.zeros(dev.backend, Float64, n)
    KernelAbstractions.synchronize(dev.backend)

    # One stable closure type — a fresh `do` body each warm iter would re-JIT.
    f = let y = y, x = x
        i -> (@inbounds y[i] = 2.0 * x[i] + 1.0)
    end

    _, first_ms = @timed_ms begin
        AK.foreachindex(f, y)
        KernelAbstractions.synchronize(dev.backend)
    end

    warm = Float64[]
    for _ in 1:warm_iters
        _, ms = @timed_ms begin
            AK.foreachindex(f, y)
            KernelAbstractions.synchronize(dev.backend)
        end
        push!(warm, ms)
    end
    sort!(warm)
    core = length(warm) >= 5 ? warm[2:end-1] : warm

    return Dict{String,Any}(
        "kind" => "ak_foreachindex",
        "n" => n,
        "first_kernel_ms" => first_ms,
        "warm_iters" => warm_iters,
        "warm_mean_ms" => sum(core) / length(core),
        "warm_median_ms" => core[(length(core) + 1) ÷ 2],
        "warm_min_ms" => minimum(core),
        "compile_proxy_ms" => max(0.0, first_ms - (sum(core) / length(core))),
    )
end

"""
    run_overhead_suite(dev; kwargs...) -> Dict

Full Julia-side suite. Intended for a fresh process (--no-warmup) so
`first_kernel_ms` includes compile.
"""
function run_overhead_suite(dev::DeviceState;
                            n_rows::Int = 4096,
                            n_cols::Int = 4096,
                            nnz_per_row::Int = 8,
                            warm_iters::Int = 20,
                            ak_n::Int = 1 << 20)
    suite = Dict{String,Any}(
        "device" => device_info(dev),
        "ka_spmv" => measure_ka_spmv(dev; n_rows = n_rows, n_cols = n_cols,
                                     nnz_per_row = nnz_per_row,
                                     warm_iters = warm_iters),
    )
    suite["ak_foreachindex"] = try
        measure_ak_map!(dev; n = ak_n, warm_iters = warm_iters)
    catch err
        Dict{String,Any}(
            "kind" => "ak_foreachindex",
            "skipped" => true,
            "reason" => sprint(showerror, err),
        )
    end
    return suite
end
