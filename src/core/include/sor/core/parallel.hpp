// SOR — a small fork-join thread pool with DETERMINISTIC reductions.
//
// LAYER L0.  Depends on nothing but the standard library.
//
// WHY A POOL OF OUR OWN, AND WHY DETERMINISM IS ITS CONTRACT.  The solver's
// results are claims that get checked and compared across runs; a run whose
// iterates depend on how the OS scheduled threads cannot be reproduced, and a
// solver whose answer changes with --threads cannot be debugged.  So the
// rule every caller of this pool follows is:
//
//   * the ARITHMETIC a task performs may depend only on the task's index,
//     never on which worker runs it or in what order tasks complete;
//   * a floating-point reduction is split into a FIXED number of chunks that
//     is a function of the problem size alone, the chunk partials are formed
//     in parallel, and then combined sequentially in chunk order
//     (deterministic_sum below).  The result is bit-identical at 1 and N
//     threads -- it is simply a different, fixed, association of the sum.
//
// OpenMP was not used: its reduction clauses do not promise an order, and
// its presence would be a build-system dependency on every platform.  The
// pool is plain std::thread.
//
// Workers: the calling thread is worker 0; size()-1 helpers sleep on a
// condition variable between jobs.  A job started from inside a job runs
// serially on the caller (no nested parallelism, no deadlock).
#pragma once

#include "sor/core/result.hpp"

#include <atomic>
#include <cstddef>
#include <functional>
#include <vector>

namespace sor::core {

class ThreadPool {
public:
    // n_threads <= 0 means default_thread_count().
    explicit ThreadPool(int n_threads = 0);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const noexcept { return n_; }

    // SPMD: runs f(worker) once on each of the first `workers` workers
    // (clamped to size()), the caller being worker 0, and returns when all
    // have finished.  Workers may synchronize with barrier().
    void run(int workers, const std::function<void(int worker)>& f);

    // f(task, worker) for task in [0, n_tasks), tasks handed out dynamically
    // (an atomic counter).  Dynamic assignment is safe for determinism only
    // because of the contract above: what a task computes must not depend on
    // `worker` beyond choosing which scratch buffer to use.
    void parallel_for(Offset n_tasks, const std::function<void(Offset task, int worker)>& f);

    // Barrier among the workers of the current run().  Must be called by all
    // of them the same number of times.
    void barrier();

    // True on a thread that is currently executing a job of some pool.
    static bool in_job() noexcept;

private:
    void worker_loop(int id);

    struct Impl;
    Impl* impl_;
    int n_ = 1;
};

// Hardware threads / 2 (one per physical core on SMT machines), at least 1,
// at most 8.  The pool is used by the sparse LDL', which is memory-bandwidth
// bound: a second hyperthread per core measured no gain on it (see
// the measurement notes), and on a shared machine more threads than
// cores mostly buys contention.
int default_thread_count();

// Process-wide pool used by the linear algebra.  set_global_threads() may be
// called before or between solves (not during one); 0 = default.
ThreadPool& global_pool();
void set_global_threads(int n_threads);
int global_threads();

// Sum of f(i), i in [0, n), in a fixed association that does not depend on
// the thread count: [0, n) is cut into chunks of `chunk` indices, each chunk
// is summed left to right, and the chunk sums are added left to right.
template <class F>
f64 deterministic_sum(ThreadPool& pool, Offset n, Offset chunk, F&& f) {
    if (n <= 0) return 0.0;
    if (chunk <= 0) chunk = 1;
    const Offset nchunks = (n + chunk - 1) / chunk;
    std::vector<f64> part(static_cast<std::size_t>(nchunks), 0.0);
    auto body = [&](Offset c, int) {
        const Offset lo = c * chunk, hi = lo + chunk < n ? lo + chunk : n;
        f64 acc = 0.0;
        for (Offset i = lo; i < hi; ++i) acc += f(i);
        part[static_cast<std::size_t>(c)] = acc;
    };
    if (nchunks == 1 || pool.size() == 1) {
        for (Offset c = 0; c < nchunks; ++c) body(c, 0);
    } else {
        pool.parallel_for(nchunks, body);
    }
    f64 s = 0.0;
    for (f64 v : part) s += v;
    return s;
}

}  // namespace sor::core
