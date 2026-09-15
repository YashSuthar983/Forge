// CPU-vs-CPU parity: verifies the harness itself, the batched-vs-single
// agreement, and that transfer accounting is populated. The same harness is
// against a candidate KernelBackend.
#include "parity_harness.hpp"

int main() {
    auto ref  = forge::backend::make_cpu_backend();
    auto cand = forge::backend::make_cpu_backend();
    CHECK(ref != nullptr);
    CHECK(cand != nullptr);
    CHECK(ref->name() == "cpu");
    CHECK(!ref->is_accelerated());

    // Same algorithm, same order: parity must be exact, not approximate.
    forge::test::run_parity(*ref, *cand, 0.0);

    // An unknown backend name must return nullptr rather than silently
    // substituting the CPU path.
    CHECK(forge::backend::make_backend("no_such_backend") == nullptr);
    CHECK(forge::backend::make_backend("cpu") != nullptr);

    // Dimension errors must throw, not corrupt memory.
    {
        using namespace forge::backend;
        const auto A = forge::test::make_test_matrix(10, 8, 3);
        DeviceBuffer<forge::core::f64> vals(A.vals.data(), A.vals.size());
        DeviceBuffer<forge::core::f64> wrong(3), y;
        CHECK_THROWS(ref->spmv(A.pattern, vals, wrong, y));
    }

    return forge::test::finish("test_backend_parity");
}
