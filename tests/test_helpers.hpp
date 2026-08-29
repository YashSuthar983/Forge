// Minimal assertion helpers. No gtest/catch2 on purpose: fewer ledger entries,
// and each test is a plain executable that ctest runs.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace sor::test {

inline int g_failures = 0;

inline void report(bool ok, const char* expr, const char* file, int line,
                   const std::string& extra = {}) {
    if (ok) return;
    ++g_failures;
    std::fprintf(stderr, "FAIL %s:%d  %s%s%s\n", file, line, expr,
                 extra.empty() ? "" : "  -- ", extra.c_str());
}

inline int finish(const char* name) {
    if (g_failures == 0) {
        std::printf("PASS %s\n", name);
        return 0;
    }
    std::fprintf(stderr, "%d failure(s) in %s\n", g_failures, name);
    return 1;
}

}  // namespace sor::test

#define CHECK(expr) \
    ::sor::test::report((expr), #expr, __FILE__, __LINE__)

#define CHECK_NEAR(a, b, tol)                                                     \
    do {                                                                          \
        const double _a = (a), _b = (b), _t = (tol);                               \
        const double _d = std::fabs(_a - _b);                                      \
        const double _s = _d / (1.0 + std::fabs(_b));                              \
        ::sor::test::report(_s <= _t, #a " ~= " #b, __FILE__, __LINE__,             \
                            "got " + std::to_string(_a) + " want " +               \
                                std::to_string(_b) + " relerr " +                  \
                                std::to_string(_s));                               \
    } while (0)

#define CHECK_THROWS(stmt)                                                        \
    do {                                                                          \
        bool _threw = false;                                                       \
        try { stmt; } catch (...) { _threw = true; }                                \
        ::sor::test::report(_threw, "throws: " #stmt, __FILE__, __LINE__);          \
    } while (0)
