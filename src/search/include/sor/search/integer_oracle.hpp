#pragma once
#include "sor/core/cancel.hpp"
#include "sor/model/lp.hpp"
#include <memory>
namespace sor::search {
// Complete finite-domain dynamic programming for pure integer models with
// integral row coefficients. States merge only when every still-live row has
// the same exact activity. Closed rows have already been verified; discarded
// prefixes therefore have an identical set of feasible completions.
class IntegerDomainOracle {
public:
    enum class Status { Ineligible, Pending, Optimal, Infeasible };
    struct Result {
        Status status = Status::Ineligible;
        std::vector<core::f64> point;
        core::f64 lower_bound_min = -core::kPosInf;
    };
    IntegerDomainOracle(const model::LpProblem&, core::f64 feasibility_tolerance);
    ~IntegerDomainOracle();
    IntegerDomainOracle(const IntegerDomainOracle&) = delete;
    IntegerDomainOracle& operator=(const IntegerDomainOracle&) = delete;
    Result advance(std::uint64_t work_allowance, const core::CancelToken* = nullptr, double time_limit_s = 0);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace sor::search
