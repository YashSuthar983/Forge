// SOR - cooperative cancellation for concurrent engine racing.
//
// One writer (the racer, when a rival arm wins), many readers (the arms).
// Engines poll at the SAME cadence as their existing wall-clock check --
// every 64 iterations -- so the steady-state cost is one relaxed atomic load
// per 64 pivots, which is unmeasurable next to a pivot's own work. On every
// serial path the pointer in *Options is null and the poll is a null compare
// the branch predictor gets right every time.
//
// Relaxed ordering is deliberate and sufficient: no data is published through
// this flag. Observing the stop one iteration late only costs a losing arm a
// few more pivots it was going to throw away regardless, and the winner's
// result is handed to the caller under the racer's own mutex, which supplies
// the happens-before edge that actually matters.
#pragma once

#include <atomic>

namespace sor::core {

class CancelToken {
public:
    CancelToken() = default;
    CancelToken(const CancelToken&) = delete;
    CancelToken& operator=(const CancelToken&) = delete;

    void request_stop() noexcept {
        stop_.store(true, std::memory_order_relaxed);
    }

    bool stop_requested() const noexcept {
        return stop_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<bool> stop_{false};
};

// Null-safe poll: `nullptr` means "not racing", which is every serial call.
inline bool cancel_requested(const CancelToken* token) noexcept {
    return token != nullptr && token->stop_requested();
}

}  // namespace sor::core
