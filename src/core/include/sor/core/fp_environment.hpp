#pragma once
#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace sor::core {
// Search kernels may discard subnormals. Restore the caller's environment
// before model validation, residual checking or exact certificate generation.
// MXCSR is thread-local, so each worker owns its own scope.
class ScopedFlushSubnormals {
public:
    explicit ScopedFlushSubnormals(bool enabled = true) noexcept {
#if defined(__SSE2__)
        saved_ = _mm_getcsr();
        if (enabled) _mm_setcsr(saved_ | 0x8040u); // FTZ (15), DAZ (6)
#else
        (void)enabled;
#endif
    }
    ~ScopedFlushSubnormals() {
#if defined(__SSE2__)
        _mm_setcsr(saved_);
#endif
    }
    ScopedFlushSubnormals(const ScopedFlushSubnormals&) = delete;
    ScopedFlushSubnormals& operator=(const ScopedFlushSubnormals&) = delete;
private:
#if defined(__SSE2__)
    unsigned saved_ = 0;
#endif
};
} // namespace sor::core
