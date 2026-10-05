#include "sor/sparse/spmv_plan.hpp"
#include "sor/core/fp_environment.hpp"
#include "test_helpers.hpp"
#include <bit>
#include <cstdint>
#include <random>
#include <vector>

int main() {
    using namespace sor::core;
    std::mt19937 gen(742);
    for (int rows : {0, 1, 7, 8, 9, 64, 137}) {
        for (int shape = 0; shape < 3; ++shape) {
            std::vector<Offset> ptr{0}; std::vector<Index> columns;
            std::vector<f64> values, x(53), reference(static_cast<std::size_t>(rows)), y(reference.size());
            for (auto& v : x) v = static_cast<int>(gen()%200)-100;
            for (int r = 0; r < rows; ++r) {
                const int length = shape == 0 ? 15 : shape == 1 ? static_cast<int>(gen()%23) : (r%8 == 0 ? 200 : 1);
                for (int k = 0; k < length; ++k) {
                    columns.push_back(gen()%53);
                    values.push_back((static_cast<int>(gen()%200)-100)*0.001);
                    reference[static_cast<std::size_t>(r)] += values.back()*x[static_cast<std::size_t>(columns.back())];
                }
                ptr.push_back(static_cast<Offset>(values.size()));
            }
            sor::sparse::SpmvPlan plan; plan.build(rows, 53, ptr, columns, values);
            plan.apply(x.data(), y.data());
            for (int r = 0; r < rows; ++r)
                CHECK(std::bit_cast<std::uint64_t>(y[static_cast<std::size_t>(r)]) ==
                      std::bit_cast<std::uint64_t>(reference[static_cast<std::size_t>(r)]));
        }
    }
#if defined(__SSE2__)
    const unsigned before = _mm_getcsr();
    { const ScopedFlushSubnormals fp; CHECK((_mm_getcsr() & 0x8040u) == 0x8040u); }
    CHECK(_mm_getcsr() == before);
#endif
    return sor::test::finish("test_spmv_plan");
}
