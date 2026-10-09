// FORGE — thread pool; see parallel.hpp for the determinism contract.
#include "forge/core/parallel.hpp"

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <exception>
#if defined(__linux__)
#include <sched.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define FORGE_CPU_RELAX() _mm_pause()
#else
#define FORGE_CPU_RELAX() ((void)0)
#endif

namespace forge::core {
namespace {
thread_local bool t_in_job = false;
// >0 while this thread runs a job serially (nested or single-worker):
// barrier() is then a no-op for it, whatever the pool's current job is.
thread_local int t_serial = 0;

// Spin a little, then yield.  Waits here are short (a barrier between two
// halves of one supernode), but the machine this runs on is often shared,
// and a thread that spins without yielding steals the core from the very
// worker it is waiting for.
struct Backoff {
    int n = 0;
    void pause() {
        if (n < 64) {
            FORGE_CPU_RELAX();
            ++n;
        } else {
            std::this_thread::yield();
        }
    }
};
}  // namespace

struct ThreadPool::Impl {
    std::vector<std::thread> threads;
    std::mutex run_mu;              // one outside caller's job at a time
    std::mutex mu;
    std::condition_variable cv;
    std::uint64_t generation = 0;   // guarded by mu
    bool stop = false;              // guarded by mu
    const std::function<void(int)>* job = nullptr;
    int job_workers = 1;
    std::atomic<int> remaining{0};
    // sense-reversing barrier
    std::atomic<int> bar_count{0};
    std::atomic<int> bar_sense{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;       // guarded by mu; published by failed
};

ThreadPool::ThreadPool(int n_threads) : impl_(new Impl) {
    n_ = n_threads <= 0 ? default_thread_count() : n_threads;
    try {
        impl_->threads.reserve(static_cast<std::size_t>(n_ - 1));
        for (int i = 1; i < n_; ++i) impl_->threads.emplace_back([this, i] { worker_loop(i); });
    } catch (...) {
        { std::lock_guard<std::mutex> lock(impl_->mu); impl_->stop = true; }
        impl_->cv.notify_all();
        for (auto& thread : impl_->threads) thread.join();
        delete impl_;
        throw;
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->stop = true;
    }
    impl_->cv.notify_all();
    for (auto& t : impl_->threads) t.join();
    delete impl_;
}

bool ThreadPool::in_job() noexcept { return t_in_job; }

void ThreadPool::invoke_job(const std::function<void(int)>& job, int id) {
    const bool previous = t_in_job;
    t_in_job = true;
    try { job(id); }
    catch (...) {
        std::lock_guard<std::mutex> lock(impl_->mu);
        if (!impl_->error) impl_->error = std::current_exception();
        impl_->failed.store(true, std::memory_order_release);
    }
    t_in_job = previous;
}

void ThreadPool::worker_loop(int id) {
    std::uint64_t seen = 0;
    for (;;) {
        const std::function<void(int)>* job = nullptr;
        int workers = 0;
        {
            std::unique_lock<std::mutex> lk(impl_->mu);
            impl_->cv.wait(lk, [&] { return impl_->stop || impl_->generation != seen; });
            if (impl_->stop) return;
            seen = impl_->generation;
            job = impl_->job;
            workers = impl_->job_workers;
        }
        if (id < workers) {
            invoke_job(*job, id);
            impl_->remaining.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
}

void ThreadPool::run(int workers, const std::function<void(int)>& f) {
    workers = std::clamp(workers, 1, n_);
    if (workers == 1 || t_in_job) {
        // Serial: a nested job, or one worker.  barrier() is a no-op then.
        struct SerialScope {
            bool previous = t_in_job;
            SerialScope() { t_in_job = true; ++t_serial; }
            ~SerialScope() { --t_serial; t_in_job = previous; }
        } scope;
        f(0);
        return;
    }
    std::lock_guard<std::mutex> run_lock(impl_->run_mu);
    impl_->remaining.store(workers - 1, std::memory_order_relaxed);
    impl_->bar_count.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->error = {};
        impl_->failed.store(false, std::memory_order_relaxed);
        impl_->job = &f;
        impl_->job_workers = workers;
        ++impl_->generation;
    }
    impl_->cv.notify_all();
    invoke_job(f, 0);
    Backoff b;
    while (impl_->remaining.load(std::memory_order_acquire) != 0) b.pause();
    { std::lock_guard<std::mutex> lock(impl_->mu); impl_->job_workers = 1; }
    if (impl_->error) std::rethrow_exception(impl_->error);
}

void ThreadPool::barrier() {
    const int workers = impl_->job_workers;
    if (t_serial > 0 || workers <= 1) return;
    const int sense = impl_->bar_sense.load(std::memory_order_acquire);
    if (impl_->bar_count.fetch_add(1, std::memory_order_acq_rel) == workers - 1) {
        impl_->bar_count.store(0, std::memory_order_relaxed);
        impl_->bar_sense.store(sense ^ 1, std::memory_order_release);
    } else {
        Backoff b;
        while (impl_->bar_sense.load(std::memory_order_acquire) == sense &&
               !impl_->failed.load(std::memory_order_acquire)) b.pause();
    }
    if (impl_->failed.load(std::memory_order_acquire)) {
        std::exception_ptr error;
        { std::lock_guard<std::mutex> lock(impl_->mu); error = impl_->error; }
        std::rethrow_exception(error);
    }
}

void ThreadPool::parallel_for(Offset n_tasks, const std::function<void(Offset, int)>& f) {
    if (n_tasks <= 0) return;
    const int workers = static_cast<int>(std::min<Offset>(n_, n_tasks));
    std::atomic<Offset> next{0};
    run(workers, [&](int w) {
        for (;;) {
            const Offset t = next.fetch_add(1, std::memory_order_relaxed);
            if (t >= n_tasks) break;
            f(t, w);
        }
    });
}

int default_thread_count() {
    const unsigned hw = std::thread::hardware_concurrency();
    int phys = hw >= 2 ? static_cast<int>(hw / 2) : 1;
#if defined(__linux__)
    // A process restricted to fewer CPUs (taskset, cgroup cpusets) gains
    // nothing from more workers than it may run on: under `taskset -c 4` the
    // six default workers time-sliced one core.
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        const int allowed = CPU_COUNT(&set);
        if (allowed > 0) phys = std::min(phys, allowed);
    }
#endif
    return std::clamp(phys, 1, 8);
}

namespace {
std::mutex g_mu;
std::unique_ptr<ThreadPool> g_pool;
int g_requested = 0;
}  // namespace

ThreadPool& global_pool() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_pool) g_pool = std::make_unique<ThreadPool>(g_requested);
    return *g_pool;
}

void set_global_threads(int n_threads) {
    std::lock_guard<std::mutex> lk(g_mu);
    const int want = n_threads <= 0 ? default_thread_count() : n_threads;
    g_requested = want;
    if (g_pool && g_pool->size() == want) return;
    g_pool.reset();
}

int global_threads() { return global_pool().size(); }

}  // namespace forge::core
