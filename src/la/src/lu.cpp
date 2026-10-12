#include "forge/la/lu.hpp"
#include "forge/core/env_switches.hpp"
#include "forge/core/work_clock.hpp"
#include "forge/la/basis_numerics.hpp"
#include "forge/la/ldlt.hpp"
#include "dense_lu.hpp"
#include "lu_markowitz.hpp"
#include "forge/core/fp_environment.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>


namespace forge::la {
namespace {


constexpr Index kBigIndex = std::numeric_limits<Index>::max();

inline std::size_t sz(Index i) { return static_cast<std::size_t>(i); }
inline std::size_t sz(Offset i) { return static_cast<std::size_t>(i); }

}  // namespace

struct FactorWorkspace {
    detail::MarkowitzKernel kernel;
    std::vector<Index> free_rows, free_cols;
};

FactorWorkspaceHolder::FactorWorkspaceHolder() = default;
FactorWorkspaceHolder::~FactorWorkspaceHolder() = default;
FactorWorkspaceHolder::FactorWorkspaceHolder(const FactorWorkspaceHolder&) {}
FactorWorkspaceHolder& FactorWorkspaceHolder::operator=(const FactorWorkspaceHolder&) {
    return *this;   // keep this factor's own workspace
}
FactorWorkspaceHolder::FactorWorkspaceHolder(FactorWorkspaceHolder&&) noexcept = default;
FactorWorkspaceHolder& FactorWorkspaceHolder::operator=(FactorWorkspaceHolder&&) noexcept = default;
FactorWorkspace& FactorWorkspaceHolder::get() {
    if (!ws_) ws_ = std::make_unique<FactorWorkspace>();
    return *ws_;
}

// ---------------------------------------------------------------------------
// Factorization
// ---------------------------------------------------------------------------

// Element growth beyond which a factor with threshold-chosen pivots is redone
// with the stable threshold (see factorize()).
constexpr f64 kElementGrowthLimit = 1e8;

bool BasisFactor::factorize(Index m,
                            const std::vector<Offset>& col_ptr,
                            const std::vector<Index>& row_idx,
                            const std::vector<f64>& vals,
                            const LuOptions& opts,
                            std::vector<Index>* singular_slots,
                            std::vector<Index>* vacant_rows) {
    core::WorkScope work_scope_(core::WorkCat::Factor);
    if (m < 0 || col_ptr.size() != sz(m)+1 || col_ptr.front() != 0 ||
        col_ptr.back() != static_cast<Offset>(row_idx.size()) || vals.size() != row_idx.size())
        throw std::invalid_argument("BasisFactor: malformed CSC shape");
    for (Index j = 0; j < m; ++j)
        if (col_ptr[sz(j)] < 0 || col_ptr[sz(j)] > col_ptr[sz(j)+1])
            throw std::invalid_argument("BasisFactor: malformed CSC offsets");
    for (std::size_t k = 0; k < vals.size(); ++k)
        if (row_idx[k] < 0 || row_idx[k] >= m || !std::isfinite(vals[k]))
            throw std::invalid_argument("BasisFactor: malformed CSC entry");
    if (!std::isfinite(opts.markowitz_threshold) || opts.markowitz_threshold <= 0 || opts.markowitz_threshold > 1 ||
        !std::isfinite(opts.pivot_tol) || opts.pivot_tol < 0 || opts.max_search_cols < 1 ||
        !std::isfinite(opts.condition_limit) || opts.condition_limit < 1 ||
        !std::isfinite(opts.stable_threshold) || opts.stable_threshold <= 0 || opts.stable_threshold > 1 ||
        !std::isfinite(opts.dense_nucleus_density) || opts.dense_nucleus_density < 0 || opts.dense_nucleus_density > 1 ||
        opts.dense_nucleus_limit < 0)
        throw std::invalid_argument("BasisFactor: invalid numerical policy");
    border_parent_.reset();
    border_ptr_.clear(); border_slot_.clear(); border_value_.clear();
    projected_ = false; retained_rows_.clear(); retained_slots_.clear();
    if (!base_.unique()) base_ = std::make_shared<BaseStorage>();
    m_ = m;
    valid_ = false;
    stats_ = LuStats{};
    stats_.dimension = m;
    stats_.input_nnz = static_cast<Offset>(row_idx.size());

    base_->piv_row_.clear();  base_->piv_slot_.clear();  base_->piv_val_.clear();
    base_->u_off_.clear();    base_->u_len_.clear();     base_->u_cap_.clear();
    base_->u_idx_.clear();    base_->u_val_.clear();     u_dead_nnz_ = 0;
    base_->l_start_.clear();  base_->l_idx_.clear();     base_->l_val_.clear();
    base_->u_cstart_.clear(); base_->u_clen_.clear();    base_->u_ccap_.clear();
    base_->u_crow_.clear();   base_->u_cval_.clear();    u_cdead_nnz_ = 0;
    base_->l_col_start_.clear(); base_->l_col_row_.clear(); base_->l_row_val_.clear();
    // A fresh factorization eliminates in index order and owes no row etas.
    base_->uord_.clear();     base_->upos_.clear();      u_reordered_ = false;
    base_->u_dead_steps_.clear();
    r_pos_.clear();    r_start_.assign(1, 0);
    r_idx_.clear();    r_val_.clear();
    eta_p_.clear();    eta_start_.assign(1, 0);
    eta_idx_.clear();  eta_val_.clear();   eta_pivot_.clear();
    if (rev_.size() != sz(m)) rev_.resize(sz(m));
    for (auto& readers : rev_) readers.clear();   // keep each list's storage
    // A new factorization: every scratch from the previous one is stale.
    scratch_.factor_gen = ++factor_gen_;
    scratch_.work_.assign(sz(m), 0.0);
    scratch_.pair_work_.assign(sz(m), 0.0);
    // scratch_.work_ is freshly zeroed, so a seeded solve may trust it immediately --
    // but any stale dirty list from the previous factorization refers to a
    // different pivot order and must be dropped.
    scratch_.work_dirty_.clear();
    scratch_.work_all_dirty_ = false;
    scratch_.fire_touched_.clear();
    scratch_.mark_.clear();    scratch_.dfs_stack_.clear();  scratch_.reach_.clear();  scratch_.order_.clear(); scratch_.seed_.clear();
    scratch_.support_stamp_.assign(sz(m), 0);
    scratch_.support_gen_ = 0;
    dense_below_ = 0;
    u_live_nnz_ = 0;
    u_alloc_nnz_ = 0;
    scratch_.work_since_factor_ = 0;
    scratch_.eta_work_since_factor_ = 0;
    scratch_.hypersparse_solves_ = 0;
    scratch_.triangular_solves_ = 0;
    if (singular_slots) singular_slots->clear();
    if (vacant_rows)    vacant_rows->clear();

    base_->piv_row_.reserve(sz(m));  base_->piv_slot_.reserve(sz(m));  base_->piv_val_.reserve(sz(m));
    base_->u_off_.reserve(sz(m));    base_->u_len_.reserve(sz(m));
    base_->l_start_.assign(1, 0);

    if (m == 0) { valid_ = true; return true; }

    // ---- eliminate -----------------------------------------------------------
    // Singletons, Markowitz pivots and dense blocks (lu_markowitz.hpp). U's
    // rows come out in ORIGINAL column indices and L's columns in ORIGINAL
    // row indices; both are translated to pivot coordinates below.
    FactorWorkspace& ws = workspace_.get();
    {
        detail::FactorSink sink{base_->piv_row_, base_->piv_slot_, base_->piv_val_,
                                base_->u_off_, base_->u_len_, base_->u_idx_, base_->u_val_,
                                u_alloc_nnz_, base_->l_start_, base_->l_idx_, base_->l_val_,
                                stats_};
        ws.kernel.run(m, col_ptr, row_idx, vals, opts, sink, ws.free_rows, ws.free_cols);
    }

    // ---- report and repair singularity ------------------------------------
    const auto n_piv = static_cast<Index>(base_->piv_row_.size());
    if (n_piv < m) {
        // Pair the unpivotable columns with the uncovered rows and give each a
        // unit pivot. The factorization returned is therefore of B with those
        // columns replaced by unit columns -- which is what the driver is about
        // to do anyway when it repairs the basis with logicals. is_valid() stays
        // false and the lists below say exactly which slots are fictitious.
        const auto& free_rows = ws.free_rows;
        const auto& free_cols = ws.free_cols;
        stats_.singular_count = static_cast<Index>(free_cols.size());
        if (singular_slots) *singular_slots = free_cols;
        if (vacant_rows)    *vacant_rows    = free_rows;

        const std::size_t n = std::min(free_rows.size(), free_cols.size());
        for (std::size_t t = 0; t < n; ++t) {
            base_->piv_row_.push_back(free_rows[t]);
            base_->piv_slot_.push_back(free_cols[t]);
            base_->piv_val_.push_back(1.0);
            base_->u_off_.push_back(u_alloc_nnz_);
            base_->u_len_.push_back(0);
            base_->l_start_.push_back(static_cast<Offset>(base_->l_idx_.size()));
        }
    }

    // ---- translate original row/column indices to pivot coordinates -------
    base_->rpos_.assign(sz(m), -1);
    base_->cpos_.assign(sz(m), -1);
    for (std::size_t k = 0; k < base_->piv_row_.size(); ++k) {
        base_->rpos_[sz(base_->piv_row_[k])] = static_cast<Index>(k);
        base_->cpos_[sz(base_->piv_slot_[k])] = static_cast<Index>(k);
    }
    for (auto& j : base_->u_idx_) j = base_->cpos_[sz(j)];
    for (auto& i : base_->l_idx_) i = base_->rpos_[sz(i)];

    // NOTE: rows stay in ORIGINAL column order (the pivot permutation does
    // not sort them). No consumer requires sorted rows; a sort here changes
    // the floating-point summation order in solve_upper, which measurably
    // changed degenerate LPs' pivot paths (fit2p: 7.3k -> 23.1k dual
    // iterations). Leave the order alone.

    build_col_patterns();
    scratch_.mark_.assign(base_->piv_val_.size(), 0);
    scratch_.dfs_stack_.clear();
    scratch_.reach_.clear();
    scratch_.order_.clear();
    dense_below_ = static_cast<Index>(base_->piv_val_.size() / 2);

    stats_.factor_nnz = u_live_nnz_ + static_cast<Offset>(base_->l_idx_.size()) +
                        static_cast<Offset>(base_->piv_val_.size());
    // Deterministic work: loading, elimination and the produced factor.
    stats_.factor_work = 16 * static_cast<Offset>(m) + 6 * static_cast<Offset>(row_idx.size()) +
                         4 * stats_.factor_nnz;
    core::work_add(static_cast<std::uint64_t>(stats_.factor_work), ::forge::core::WorkTag::LuFactor);
    valid_ = (n_piv == m);
    stats_.effective_threshold = opts.markowitz_threshold;
    // The estimate only decides whether to redo the elimination with the
    // stable threshold, and the threshold only governs pivots the Markowitz
    // search chose: singletons take none, dense blocks pivot by rows. With no
    // such pivot the redo would return this very factor, so the estimate --
    // two thirds of fit2p's factorization -- is skipped.
    //
    // Element growth decides first, at no cost: threshold pivoting bounds
    // the multipliers, not the entries of U, and a chain of admissible
    // pivots compounded them to 3.5e12 from unit entries on a maros-r7 basis
    // (relative residual 0.27) while the condition estimate -- formed with
    // that inaccurate factor -- stayed under its limit.
    const Index searched_pivots = stats_.nucleus_pivots - stats_.blocked_nucleus_pivots;
    if (valid_ && opts.adaptive_threshold && searched_pivots > 0) {
        const bool grew = stats_.element_growth > kElementGrowthLimit;
        if (!grew)
            stats_.condition_estimate = estimate_basis_condition(*this, m, col_ptr, row_idx, vals);
        if ((grew || stats_.condition_estimate > opts.condition_limit) &&
            opts.markowitz_threshold < opts.stable_threshold) {
            auto stable = opts;
            stable.markowitz_threshold = opts.stable_threshold;
            return factorize(m, col_ptr, row_idx, vals, stable, singular_slots, vacant_rows);
        }
    }
    scratch_.work_since_factor_ = 0; // Estimation is factorization setup, not update-file work.
    scratch_.eta_work_since_factor_ = 0;
    return valid_;
}

// ---------------------------------------------------------------------------
// Column patterns for the hypersparse reach-set DFS, compressed by counting
// sort. Values stay row-major (the solves read them there); these carry only
// the dependency structure.
// ---------------------------------------------------------------------------

void BasisFactor::reset_scratch(SolveScratch& sc) const {
    sc = SolveScratch{};
    sc.work_.assign(sz(m_), 0.0);
    sc.pair_work_.assign(sz(m_), 0.0);
    sc.work_all_dirty_ = false;   // freshly zeroed, as after factorize()
    sc.mark_.assign(base_->piv_val_.size(), 0);
    sc.support_stamp_.assign(sz(m_), 0);
    sc.factor_gen = factor_gen_;
}

void BasisFactor::extend_logical_rows(Index added,
                                     const std::vector<Offset>& row_ptr,
                                     const std::vector<Index>& slots,
                                     const std::vector<f64>& values) {
    if (!valid_ || added < 0 || added > std::numeric_limits<Index>::max() - m_ ||
        row_ptr.size() != sz(added) + 1 || row_ptr.front() != 0 ||
        row_ptr.back() != static_cast<Offset>(slots.size()) || slots.size() != values.size())
        throw std::invalid_argument("BasisFactor: malformed logical border");
    for (Index i = 0; i < added; ++i)
        if (row_ptr[sz(i)] < 0 || row_ptr[sz(i)] > row_ptr[sz(i) + 1])
            throw std::invalid_argument("BasisFactor: malformed border offsets");
    for (std::size_t k = 0; k < slots.size(); ++k)
        if (slots[k] < 0 || slots[k] >= m_ || !std::isfinite(values[k]))
            throw std::invalid_argument("BasisFactor: malformed border entry");
    if (added == 0) return;
    BasisFactor next;
    next.border_parent_ = std::make_shared<const BasisFactor>(*this);
    next.border_ptr_ = row_ptr; next.border_slot_ = slots; next.border_value_ = values;
    next.m_ = m_ + added;
    next.valid_ = true;
    next.stats_ = stats_;
    next.stats_.dimension = next.m_;
    next.stats_.input_nnz += static_cast<Offset>(values.size()) + added;
    next.stats_.factor_nnz += static_cast<Offset>(values.size()) + added;
    next.eta_start_.assign(1, 0); next.r_start_.assign(1, 0);
    next.rev_.resize(sz(next.m_));
    next.factor_gen_ = factor_gen_ + 1;
    core::work_add(values.size() + sz(added), core::WorkTag::LuFactor);
    *this = std::move(next);
}

void BasisFactor::retain_logical_rows(const std::vector<Index>& rows,
                                     const std::vector<Index>& slots) {
    if (!valid_ || rows.size() != slots.size() || rows.size() > sz(m_))
        throw std::invalid_argument("BasisFactor: malformed logical projection");
    const auto validate = [&](const auto& map) {
        std::vector<char> seen(sz(m_), false);
        for (Index j : map) {
            if (j < 0 || j >= m_ || seen[sz(j)])
                throw std::invalid_argument("BasisFactor: malformed projection map");
            seen[sz(j)] = true;
        }
    };
    validate(rows); validate(slots);
    BasisFactor next;
    next.border_parent_ = std::make_shared<const BasisFactor>(*this);
    next.projected_ = true;
    next.retained_rows_ = rows; next.retained_slots_ = slots;
    next.m_ = static_cast<Index>(rows.size()); next.valid_ = true;
    next.stats_ = stats_; next.stats_.dimension = next.m_;
    next.eta_start_.assign(1, 0); next.r_start_.assign(1, 0); next.rev_.resize(rows.size());
    next.factor_gen_ = factor_gen_ + 1;
    core::work_add(rows.size() * 2, core::WorkTag::LuFactor);
    *this = std::move(next);
}

bool BasisFactor::solve_border(SolveScratch& sc, std::vector<f64>& b,
                              std::vector<Index>* support, bool transpose) const {
    if (b.size() != sz(m_))
        throw std::invalid_argument("BasisFactor: bordered RHS size");
    if (sc.factor_gen != factor_gen_) reset_scratch(sc);
    if (!sc.border_scratch) sc.border_scratch = std::make_unique<SolveScratch>();
    const Offset parent_work_before = sc.border_scratch->work_since_factor_;
    const Offset parent_eta_before = sc.border_scratch->eta_work_since_factor_;
    const Index old = border_parent_->dimension();
    auto& rhs = sc.border_rhs;
    if (!projected_) rhs.assign(b.begin(), b.begin() + old);
    if (transpose) {
        // E^-T newest first, then [[B^-T,B^-T D^T],[0,-I]].
        for (std::size_t t = eta_p_.size(); t-- > 0;) {
            f64 value = b[sz(eta_p_[t])];
            for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k)
                value -= eta_val_[sz(k)] * b[sz(eta_idx_[sz(k)])];
            b[sz(eta_p_[t])] = value / eta_pivot_[t];
        }
        if (projected_) {
            rhs.assign(sz(old), 0);
            for (Index i = 0; i < m_; ++i) rhs[sz(retained_slots_[sz(i)])] = b[sz(i)];
        } else {
            std::copy_n(b.begin(), old, rhs.begin());
            for (Index i = old; i < m_; ++i)
                for (Offset k = border_ptr_[sz(i - old)]; k < border_ptr_[sz(i - old) + 1]; ++k)
                    rhs[sz(border_slot_[sz(k)])] += border_value_[sz(k)] * b[sz(i)];
        }
        border_parent_->btran_impl(*sc.border_scratch, rhs, nullptr, nullptr);
        if (projected_)
            for (Index i = 0; i < m_; ++i) b[sz(i)] = rhs[sz(retained_rows_[sz(i)])];
        else {
            std::copy(rhs.begin(), rhs.end(), b.begin());
            for (Index i = old; i < m_; ++i) b[sz(i)] = -b[sz(i)];
        }
    } else {
        if (projected_) {
            rhs.assign(sz(old), 0);
            for (Index i = 0; i < m_; ++i) rhs[sz(retained_rows_[sz(i)])] = b[sz(i)];
        }
        border_parent_->ftran_impl(*sc.border_scratch, rhs, nullptr, nullptr);
        if (projected_)
            for (Index i = 0; i < m_; ++i) b[sz(i)] = rhs[sz(retained_slots_[sz(i)])];
        else {
            std::copy(rhs.begin(), rhs.end(), b.begin());
            for (Index i = old; i < m_; ++i) {
                f64 value = -b[sz(i)];
                for (Offset k = border_ptr_[sz(i - old)]; k < border_ptr_[sz(i - old) + 1]; ++k)
                    value += border_value_[sz(k)] * rhs[sz(border_slot_[sz(k)])];
                b[sz(i)] = value;
            }
        }
        for (std::size_t t = 0; t < eta_p_.size(); ++t) {
            const f64 value = b[sz(eta_p_[t])] / eta_pivot_[t];
            for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k)
                b[sz(eta_idx_[sz(k)])] -= eta_val_[sz(k)] * value;
            b[sz(eta_p_[t])] = value;
        }
    }
    if (support) {
        support->clear();
        for (Index i = 0; i < m_; ++i) if (b[sz(i)] != 0) support->push_back(i);
    }
    const Offset own_etas = static_cast<Offset>(eta_val_.size());
    const Offset work = static_cast<Offset>(border_value_.size()) + m_ + own_etas;
    sc.work_since_factor_ += work + sc.border_scratch->work_since_factor_ - parent_work_before;
    sc.eta_work_since_factor_ += own_etas + sc.border_scratch->eta_work_since_factor_ - parent_eta_before;
    core::work_add(static_cast<std::uint64_t>(work), core::WorkTag::LuSolve);
    return true;
}

// Spare entries reserved behind a U row or column of `len` entries, so an
// update can append the entering column's spike in place. A few entries per
// row cover the common case; the proportional part covers long rows of dense
// bases, where a spike touches most rows at once.
static inline Index u_slack(Index len) { return 4 + len / 8; }

void BasisFactor::build_col_patterns() {
    const auto n = base_->piv_val_.size();

    // U's elimination order starts as the index order; update_ft() is the only
    // thing that ever changes it.
    base_->uord_.resize(n);
    base_->upos_.resize(n);
    base_->u_dead_steps_.clear();
    for (std::size_t k = 0; k < n; ++k) {
        base_->uord_[k] = static_cast<Index>(k);
        base_->upos_[k] = static_cast<Index>(k);
    }
    u_reordered_ = false;

    pack_u();

    // L by rows, for the transposed solves.
    base_->l_col_start_.assign(n + 1, 0);
    base_->l_col_row_.clear();
    base_->l_col_row_.resize(base_->l_idx_.size());
    base_->l_row_val_.resize(base_->l_idx_.size());
    for (std::size_t t = 0; t < base_->l_idx_.size(); ++t) ++base_->l_col_start_[sz(base_->l_idx_[t]) + 1];
    for (std::size_t j = 0; j < n; ++j)
        base_->l_col_start_[j + 1] += base_->l_col_start_[j];
    std::vector<Offset> l_cursor(base_->l_col_start_.begin(), base_->l_col_start_.end() - 1);
    for (std::size_t k = 0; k < n; ++k) {
        for (Offset t = base_->l_start_[k]; t < base_->l_start_[k + 1]; ++t) {
            const auto at = sz(l_cursor[sz(base_->l_idx_[sz(t)])]++);
            base_->l_col_row_[at] = static_cast<Index>(k);
            base_->l_row_val_[at] = base_->l_val_[sz(t)];
        }
    }
}

// Lay U out packed, row store and column store alike, with u_slack() spare
// entries behind every segment, and link every entry's two copies. Contents
// are preserved exactly; column entries come in ascending row order.
//
// The slack lets update_ft() append the entering column's spike in place.
// With zero slack the first update after a factorization relocated every row
// the spike touched -- the spike reaches most rows of a dense-ish basis --
// and those copies were most of an update's cost (fit2p: 271 us against 14 us
// for the same update in HiGHS, measured on the same basis).
void BasisFactor::pack_u() {
    const auto n = base_->piv_val_.size();
    auto& B = *base_;
    Offset total = 0;
    u_live_nnz_ = 0;
    for (std::size_t k = 0; k < n; ++k) {
        total += B.u_len_[k] + u_slack(B.u_len_[k]);
        u_live_nnz_ += static_cast<Offset>(B.u_len_[k]);
    }
    {
        std::vector<Index> idx(sz(total), 0);
        std::vector<f64> val(sz(total), 0.0);
        B.u_cap_.resize(n);
        Offset at = 0;
        for (std::size_t k = 0; k < n; ++k) {
            const Offset beg = B.u_off_[k];
            const Index len = B.u_len_[k];
            std::copy(B.u_idx_.begin() + beg, B.u_idx_.begin() + beg + len, idx.begin() + at);
            std::copy(B.u_val_.begin() + beg, B.u_val_.begin() + beg + len, val.begin() + at);
            B.u_off_[k] = at;
            B.u_cap_[k] = len + u_slack(len);
            at += B.u_cap_[k];
        }
        B.u_idx_ = std::move(idx);
        B.u_val_ = std::move(val);
    }
    u_dead_nnz_ = 0;

    B.u_cstart_.assign(n + 1, 0);
    B.u_clen_.assign(n, 0);
    B.u_ccap_.assign(n, 0);
    for (std::size_t k = 0; k < n; ++k) {
        const Offset beg = B.u_off_[k], end = beg + B.u_len_[k];
        for (Offset t = beg; t < end; ++t) ++B.u_ccap_[sz(B.u_idx_[sz(t)])];
    }
    for (std::size_t j = 0; j < n; ++j) {
        B.u_ccap_[j] += u_slack(B.u_ccap_[j]);
        B.u_cstart_[j + 1] = B.u_cstart_[j] + B.u_ccap_[j];
    }
    const auto csize = std::max<std::size_t>(sz(B.u_cstart_[n]), 1);
    B.u_crow_.assign(csize, 0);
    B.u_cval_.assign(csize, 0.0);
    B.u_cmirror_.assign(csize, 0);
    B.u_rmirror_.assign(B.u_idx_.size(), 0);
    for (std::size_t k = 0; k < n; ++k) {
        const Offset beg = B.u_off_[k], end = beg + B.u_len_[k];
        for (Offset t = beg; t < end; ++t) {
            const auto j = sz(B.u_idx_[sz(t)]);
            const Offset at = B.u_cstart_[j] + B.u_clen_[j];
            B.u_crow_[sz(at)] = static_cast<Index>(k);
            B.u_cval_[sz(at)] = B.u_val_[sz(t)];
            B.u_cmirror_[sz(at)] = static_cast<Index>(t - beg);
            B.u_rmirror_[sz(t)] = B.u_clen_[j];
            ++B.u_clen_[j];
        }
    }
    u_cdead_nnz_ = 0;
    u_alloc_nnz_ = static_cast<Offset>(B.u_idx_.size());
}

// ---------------------------------------------------------------------------
// Mutable U stores (Forrest-Tomlin only)
//
// Both mirrors use the same discipline: a segment [start, start+len) inside a
// capacity [start, start+cap). Appending inside the capacity is O(1); a segment
// that outgrows it is relocated to the end of the arrays with doubled slack and
// its old space is abandoned (counted, so a caller can see the waste). Erasure
// is swap-with-last inside the segment, which is why neither store is sorted --
// nothing downstream requires it.
// ---------------------------------------------------------------------------

// Every entry is stored twice, once per store, and each copy knows where the
// other one is: u_rmirror_[rs] is the twin's offset inside its column's
// segment, u_cmirror_[cs] the twin's offset inside its row's. That is what
// makes erasure O(1): an update removes a whole column and a whole row of U,
// and finding each entry's twin by scanning its row or column cost the length
// of every row the old column met plus every column the old row met. The links
// are RELATIVE so that moving a whole segment (relocation) leaves every link
// valid: it is a plain copy, with no write into the other store per entry.

// Room for `extra` more entries behind row `row`, relocating it to the end
// with doubled slack if its capacity is short.
void BasisFactor::u_row_reserve(Index row, Index extra) {
    auto& B = *base_;
    const auto r = sz(row);
    const Index len = B.u_len_[r];
    if (len + extra <= B.u_cap_[r]) return;
    const Index cap = std::max<Index>(2 * len + 4, len + extra);
    const auto old = sz(B.u_off_[r]);
    const auto at = B.u_idx_.size();
    B.u_idx_.resize(at + sz(cap));
    B.u_val_.resize(at + sz(cap));
    B.u_rmirror_.resize(at + sz(cap));
    std::copy_n(B.u_idx_.begin() + static_cast<std::ptrdiff_t>(old), len,
                B.u_idx_.begin() + static_cast<std::ptrdiff_t>(at));
    std::copy_n(B.u_val_.begin() + static_cast<std::ptrdiff_t>(old), len,
                B.u_val_.begin() + static_cast<std::ptrdiff_t>(at));
    std::copy_n(B.u_rmirror_.begin() + static_cast<std::ptrdiff_t>(old), len,
                B.u_rmirror_.begin() + static_cast<std::ptrdiff_t>(at));
    B.u_off_[r] = static_cast<Offset>(at);
    u_dead_nnz_ += static_cast<Offset>(B.u_cap_[r]);
    B.u_cap_[r] = cap;
}

void BasisFactor::u_col_reserve(Index col, Index extra) {
    auto& B = *base_;
    const auto c = sz(col);
    const Index len = B.u_clen_[c];
    if (len + extra <= B.u_ccap_[c]) return;
    const Index cap = std::max<Index>(2 * len + 4, len + extra);
    const auto old = sz(B.u_cstart_[c]);
    const auto at = B.u_crow_.size();
    B.u_crow_.resize(at + sz(cap));
    B.u_cval_.resize(at + sz(cap));
    B.u_cmirror_.resize(at + sz(cap));
    std::copy_n(B.u_crow_.begin() + static_cast<std::ptrdiff_t>(old), len,
                B.u_crow_.begin() + static_cast<std::ptrdiff_t>(at));
    std::copy_n(B.u_cval_.begin() + static_cast<std::ptrdiff_t>(old), len,
                B.u_cval_.begin() + static_cast<std::ptrdiff_t>(at));
    std::copy_n(B.u_cmirror_.begin() + static_cast<std::ptrdiff_t>(old), len,
                B.u_cmirror_.begin() + static_cast<std::ptrdiff_t>(at));
    B.u_cstart_[c] = static_cast<Offset>(at);
    u_cdead_nnz_ += static_cast<Offset>(B.u_ccap_[c]);
    B.u_ccap_[c] = cap;
}

void BasisFactor::u_append(Index row, Index col, f64 val) {
    auto& B = *base_;
    u_row_reserve(row, 1);
    u_col_reserve(col, 1);
    const Index rlen = B.u_len_[sz(row)], clen = B.u_clen_[sz(col)];
    const auto rs = sz(B.u_off_[sz(row)] + rlen);
    const auto cs = sz(B.u_cstart_[sz(col)] + clen);
    B.u_idx_[rs] = col;
    B.u_val_[rs] = val;
    B.u_crow_[cs] = row;
    B.u_cval_[cs] = val;
    B.u_rmirror_[rs] = clen;
    B.u_cmirror_[cs] = rlen;
    B.u_len_[sz(row)] = rlen + 1;
    B.u_clen_[sz(col)] = clen + 1;
    ++u_live_nnz_;
}

// Swap-with-last inside the segment; the moved entry's twin learns its new
// offset. The caller disposes of the dropped entry's twin.
void BasisFactor::u_row_drop(Index row, Offset rs) {
    auto& B = *base_;
    const Offset beg = B.u_off_[sz(row)];
    const Offset last = beg + B.u_len_[sz(row)] - 1;
    if (rs != last) {
        const Index j = B.u_idx_[sz(last)];
        B.u_idx_[sz(rs)] = j;
        B.u_val_[sz(rs)] = B.u_val_[sz(last)];
        B.u_rmirror_[sz(rs)] = B.u_rmirror_[sz(last)];
        B.u_cmirror_[sz(B.u_cstart_[sz(j)] + B.u_rmirror_[sz(rs)])] = static_cast<Index>(rs - beg);
    }
    --B.u_len_[sz(row)];
}

void BasisFactor::u_col_drop(Index col, Offset cs) {
    auto& B = *base_;
    const Offset beg = B.u_cstart_[sz(col)];
    const Offset last = beg + B.u_clen_[sz(col)] - 1;
    if (cs != last) {
        const Index i = B.u_crow_[sz(last)];
        B.u_crow_[sz(cs)] = i;
        B.u_cval_[sz(cs)] = B.u_cval_[sz(last)];
        B.u_cmirror_[sz(cs)] = B.u_cmirror_[sz(last)];
        B.u_rmirror_[sz(B.u_off_[sz(i)] + B.u_cmirror_[sz(cs)])] = static_cast<Index>(cs - beg);
    }
    --B.u_clen_[sz(col)];
}

Index BasisFactor::u_order_move_to_back(Index p) {
    // Append-only: the position gets a new step at the end and its old step
    // becomes a hole (-1) that the solves skip. Shifting the steps after it
    // down by one was O(m) per update -- on a basis with a few thousand rows
    // a large part of what an update cost. The holes go at the next
    // factorization (build_col_patterns); there are at most as many as
    // updates since then.
    auto& ord = base_->uord_;
    const Index t0 = base_->upos_[sz(p)];
    const auto size = static_cast<Index>(ord.size());
    Index dead_after = 0;
    for (const Index d : base_->u_dead_steps_) dead_after += d > t0;
    ord[sz(t0)] = -1;
    base_->u_dead_steps_.push_back(t0);
    base_->upos_[sz(p)] = size;
    ord.push_back(p);
    u_reordered_ = true;
    // The live steps from p's old place to the end, p included: what the
    // shifting version measured as m - upos[p].
    return size - t0 - dead_after;
}

// ---------------------------------------------------------------------------
// Triangular solves, in pivot coordinates
// ---------------------------------------------------------------------------

namespace {
#ifndef FORGE_RESTRICT
#define FORGE_RESTRICT __restrict
#endif
// Four scatter updates v[idx] -= val * zk, unrolled. (An AVX2 variant that
// packed the multiplies sat here behind __AVX2__, which the portable build
// never defines; a whole -march=native build measured no gain on Netlib-93,
// sum 15.61 -> 15.58 s with identical pivots: the scatter, not the multiply,
// is the cost.)
inline void scatter_axpy4(f64* FORGE_RESTRICT v,
                          const Index* FORGE_RESTRICT idx,
                          const f64* FORGE_RESTRICT val,
                          Offset t, f64 zk) {
    v[sz(idx[t])]     -= val[t]     * zk;
    v[sz(idx[t + 1])] -= val[t + 1] * zk;
    v[sz(idx[t + 2])] -= val[t + 2] * zk;
    v[sz(idx[t + 3])] -= val[t + 3] * zk;
}
}  // namespace

void BasisFactor::solve_lower(std::vector<f64>& v) const {
    const auto n = base_->piv_val_.size();
    for (std::size_t k = 0; k < n; ++k) {
        const f64 zk = v[k];
        if (zk == 0.0) continue;
        const Offset beg = base_->l_start_[k], end = base_->l_start_[k + 1];
        Offset t = beg;
        const Offset lim = beg + ((end - beg) & ~Offset{3});
        const Index* idx = base_->l_idx_.data();
        const f64* val = base_->l_val_.data();
        for (; t < lim; t += 4)
            scatter_axpy4(v.data(), idx, val, t, zk);
        for (; t < end; ++t)
            v[sz(idx[t])] -= val[t] * zk;
    }
}

void BasisFactor::solve_lower_pair(std::vector<f64>& a,
                                   std::vector<f64>& b) const {
    const auto n = base_->piv_val_.size();
    for (std::size_t k = 0; k < n; ++k) {
        const f64 ak = a[k];
        const f64 bk = b[k];
        if (ak == 0.0 && bk == 0.0) continue;
        const Offset beg = base_->l_start_[k], end = base_->l_start_[k + 1];
        for (Offset t = beg; t < end; ++t) {
            const auto i = sz(base_->l_idx_[sz(t)]);
            const f64 multiplier = base_->l_val_[sz(t)];
            a[i] -= multiplier * ak;
            b[i] -= multiplier * bk;
        }
    }
}

// U w = v as a COLUMN scatter in reverse elimination order: once w_k is known,
// column k of U is subtracted from the rows above it, and a zero w_k costs one
// test. The row gather it replaces read every entry of U on every call, which
// is what kept the hypersparse path running up to half of n.
void BasisFactor::solve_upper(std::vector<f64>& v) const {
    const auto n = base_->piv_val_.size();
    const auto step = [&](const std::size_t k) {
        f64 wk = v[k];
        if (wk == 0.0) return;
        wk /= base_->piv_val_[k];
        v[k] = wk;
        const Offset beg = base_->u_cstart_[k], end = beg + base_->u_clen_[k];
        Offset q = beg;
        const Offset lim = beg + ((end - beg) & ~Offset{3});
        const Index* idx = base_->u_crow_.data();
        const f64* val = base_->u_cval_.data();
        for (; q < lim; q += 4)
            scatter_axpy4(v.data(), idx, val, q, wk);
        for (; q < end; ++q)
            v[sz(idx[q])] -= val[q] * wk;
    };
    if (u_reordered_) {
        const auto& ord = base_->uord_;
        for (std::size_t t = ord.size(); t-- > 0;)
            if (ord[t] >= 0) step(sz(ord[t]));
    } else {
        for (std::size_t k = n; k-- > 0;) step(k);
    }
}

void BasisFactor::solve_upper_pair(std::vector<f64>& a,
                                   std::vector<f64>& b) const {
    const auto n = base_->piv_val_.size();
    const auto step = [&](const std::size_t k) {
        f64 ak = a[k], bk = b[k];
        if (ak == 0.0 && bk == 0.0) return;
        ak /= base_->piv_val_[k];
        bk /= base_->piv_val_[k];
        a[k] = ak;
        b[k] = bk;
        const Offset beg = base_->u_cstart_[k], end = beg + base_->u_clen_[k];
        for (Offset q = beg; q < end; ++q) {
            const auto r = sz(base_->u_crow_[sz(q)]);
            const f64 value = base_->u_cval_[sz(q)];
            a[r] -= value * ak;
            b[r] -= value * bk;
        }
    };
    if (u_reordered_) {
        const auto& ord = base_->uord_;
        for (std::size_t t = ord.size(); t-- > 0;)
            if (ord[t] >= 0) step(sz(ord[t]));
    } else {
        for (std::size_t k = n; k-- > 0;) step(k);
    }
}

void BasisFactor::solve_upper_t(std::vector<f64>& v) const {
    const auto n = base_->piv_val_.size();
    if (u_reordered_) {
        const auto& ord = base_->uord_;
        for (std::size_t t = 0; t < ord.size(); ++t) {
            if (ord[t] < 0) continue;
            const auto k = sz(ord[t]);
            const f64 zk = v[k] / base_->piv_val_[k];
            v[k] = zk;
            if (zk == 0.0) continue;
            const Offset beg = base_->u_off_[k], end = beg + base_->u_len_[k];
            for (Offset q = beg; q < end; ++q)
                v[sz(base_->u_idx_[sz(q)])] -= base_->u_val_[sz(q)] * zk;
        }
        return;
    }
    for (std::size_t k = 0; k < n; ++k) {
        const f64 zk = v[k] / base_->piv_val_[k];
        v[k] = zk;
        if (zk == 0.0) continue;
        const Offset beg = base_->u_off_[k], end = beg + base_->u_len_[k];
        for (Offset t = beg; t < end; ++t)
            v[sz(base_->u_idx_[sz(t)])] -= base_->u_val_[sz(t)] * zk;
    }
}

// L' w = v as a ROW scatter in reverse order (L has a unit diagonal): w_i is
// final when reached, and row i of L is subtracted from the columns before it.
void BasisFactor::solve_lower_t(std::vector<f64>& v) const {
    for (std::size_t i = base_->piv_val_.size(); i-- > 0;) {
        const f64 wi = v[i];
        if (wi == 0.0) continue;
        const Offset beg = base_->l_col_start_[i], end = base_->l_col_start_[i + 1];
        for (Offset t = beg; t < end; ++t)
            v[sz(base_->l_col_row_[sz(t)])] -= base_->l_row_val_[sz(t)] * wi;
    }
}

// ---------------------------------------------------------------------------
// Hypersparse solves (Hall & McKinnon 2005): touch only what can be nonzero
// ---------------------------------------------------------------------------

// Row etas (Forrest-Tomlin). M_t = I - e_{p_t} v_t^T, so
//   FTRAN  applies M_1 ... M_k, oldest first: v[p] -= <v_t, v>
//   BTRAN  applies M_k^T ... M_1^T, newest first: v[j] -= v_t[j] * v[p]
// Skipping is exact, not a heuristic: with v[p_t] excluded from the stored
// vector, an FTRAN eta whose whole support is zero writes zero to v[p], and a
// BTRAN eta with v[p] == 0 writes zero everywhere it touches.
void BasisFactor::apply_row_etas_ftran(std::vector<f64>& v,
                                       std::vector<Index>* touched) const {
    if (touched == nullptr) {
        // Dense RHS: skip the s == 0 branch so the write stays predictable /
        // SIMD-friendly. Multiplying through a zero s is cheaper than a
        // mispredicted continue on mid-dense vectors.
        for (std::size_t t = 0; t < r_pos_.size(); ++t) {
            f64 s = 0.0;
            for (Offset q = r_start_[t]; q < r_start_[t + 1]; ++q)
                s += r_val_[sz(q)] * v[sz(r_idx_[sz(q)])];
            v[sz(r_pos_[t])] -= s;
        }
        return;
    }
    for (std::size_t t = 0; t < r_pos_.size(); ++t) {
        f64 s = 0.0;
        for (Offset q = r_start_[t]; q < r_start_[t + 1]; ++q)
            s += r_val_[sz(q)] * v[sz(r_idx_[sz(q)])];
        if (s == 0.0) continue;
        const auto p = sz(r_pos_[t]);
        if (v[p] == 0.0) touched->push_back(r_pos_[t]);
        v[p] -= s;
    }
}

void BasisFactor::apply_row_etas_btran(std::vector<f64>& v,
                                       std::vector<Index>* touched) const {
    if (touched == nullptr) {
        for (std::size_t t = r_pos_.size(); t-- > 0;) {
            const f64 vp = v[sz(r_pos_[t])];
            for (Offset q = r_start_[t]; q < r_start_[t + 1]; ++q)
                v[sz(r_idx_[sz(q)])] -= r_val_[sz(q)] * vp;
        }
        return;
    }
    for (std::size_t t = r_pos_.size(); t-- > 0;) {
        const f64 vp = v[sz(r_pos_[t])];
        if (vp == 0.0) continue;
        for (Offset q = r_start_[t]; q < r_start_[t + 1]; ++q) {
            const auto j = sz(r_idx_[sz(q)]);
            if (v[j] == 0.0) touched->push_back(r_idx_[sz(q)]);
            v[j] -= r_val_[sz(q)] * vp;
        }
    }
}

// Shared DFS core. `seed` is the input's nonzero pattern; `adjacency(k)`
// yields the positions that a nonzero at k can make nonzero. Marks are left
// SET on success (the caller needs them for the fused scatter and clears
// them via the reach list). On a bail-out v is untouched and marks are
// cleared here, so the dense fallback stays correct.
template <typename Adj>
static bool reach_dfs(const std::vector<Index>& seed, Adj adjacency,
                      std::vector<char>& mark, std::vector<Index>& stack,
                      std::vector<Index>& reach, Index dense_below) {
    for (const Index j : seed) {
        if (!mark[sz(j)]) { mark[sz(j)] = 1; stack.push_back(j); }
    }
    reach.clear();
    while (!stack.empty()) {
        const Index k = stack.back();
        stack.pop_back();
        adjacency(k, [&](Index i) {
            if (!mark[sz(i)]) { mark[sz(i)] = 1; stack.push_back(i); }
        });
        reach.push_back(k);
        if (static_cast<Index>(reach.size() + stack.size()) > dense_below) {
            for (const Index j : reach) mark[sz(j)] = 0;
            while (!stack.empty()) { mark[sz(stack.back())] = 0; stack.pop_back(); }
            for (const Index j : seed) mark[sz(j)] = 0;
            reach.clear();
            return false;
        }
    }
    return true;
}

// Output density above which a solve skips the hypersparse attempt. The DFS
// costs a random access per reached edge on top of the arithmetic, while the
// zero-skipping elimination-order pass costs one test per position plus the
// same arithmetic, so the reach must be a small fraction of n to pay for
// itself; 10% is the classical crossover (Gilbert & Peierls's argument,
// Hall & McKinnon's measurements).
constexpr double kHyperOutputDensity = 0.10;

// Running average of a solve family's output density (see SolveScratch).
inline void learn_density(std::array<f64, 33>& table, std::size_t bucket,
                          f64 density) {
    f64& d = table[bucket];
    d = 0.75 * d + 0.25 * density;
}

// Sort distinct indices in [0, m) ascending -- the same result as std::sort,
// which callers rely on for their visiting order (tie-breaking), at a cost
// linear in the size. A solve's support was sorted with std::sort, 6 us of
// a 33 us hypersparse FTRAN on supportcase12 (189 entries). Measured on 20k
// random supports with m = 28k: 189 entries std::sort 4.5 us, 8-bit LSD
// radix 1.3 us; 2000 entries std::sort 66 us, radix 11 us, bitmap 6 us.
static void sort_support(std::vector<Index>& v, Index m, std::vector<Index>& tmp,
                         std::vector<std::uint64_t>& bits) {
    const std::size_t k = v.size();
    if (k <= 16) {
        std::sort(v.begin(), v.end());
        return;
    }
    if (32 * k > static_cast<std::size_t>(m)) {
        // Dense enough that a pass over m/64 words beats digit passes.
        bits.resize((static_cast<std::size_t>(m) + 63) / 64, 0);
        for (const Index x : v)
            bits[static_cast<std::size_t>(x) >> 6] |= std::uint64_t{1} << (x & 63);
        v.clear();
        for (std::size_t w = 0; w < bits.size(); ++w) {
            std::uint64_t word = bits[w];
            if (word == 0) continue;
            bits[w] = 0;
            while (word != 0) {
                v.push_back(static_cast<Index>(w * 64 + static_cast<std::size_t>(std::countr_zero(word))));
                word &= word - 1;
            }
        }
        return;
    }
    constexpr int kDigit = 8, kBuckets = 1 << kDigit;
    const int top = std::bit_width(static_cast<unsigned>(std::max<Index>(m, 1)));
    std::uint32_t count[kBuckets];
    tmp.resize(k);
    for (int shift = 0; shift < top; shift += kDigit) {
        std::fill(count, count + kBuckets, 0u);
        for (const Index x : v) ++count[(static_cast<unsigned>(x) >> shift) & (kBuckets - 1)];
        std::uint32_t sum = 0;
        for (auto& c : count) { const std::uint32_t here = c; c = sum; sum += here; }
        for (const Index x : v) tmp[count[(static_cast<unsigned>(x) >> shift) & (kBuckets - 1)]++] = x;
        v.swap(tmp);
    }
}

// The reach of `seed` in depth-first POST-order (Gilbert & Peierls 1988):
// every node is listed after every node reachable from it. For the two
// row-oriented (gather) solves below an edge k -> i means "row i reads x_k",
// so walking this list backwards visits each x_k after everything its row
// reads -- a valid solve order with no sort. A gather computes each x_k
// from its own row in storage order, so the result is bit-identical to the
// elimination-order walk it replaces. Same reach set, marks and dense
// cutoff as reach_dfs; `children(k)` returns k's adjacency as [begin, end).
template <class Children>
static bool reach_postorder(const std::vector<Index>& seed, Children children,
                            std::vector<char>& mark, std::vector<DfsFrame>& frames,
                            std::vector<Index>& reach, Index dense_below) {
    // One frame per node on the path: the node and its remaining children,
    // so returning to a node does not look its adjacency up again. Frames and
    // reach are sized once and written by index (each holds at most
    // dense_below + 1 entries before the cutoff aborts).
    const auto cap = static_cast<std::size_t>(std::max<Index>(dense_below, 0)) + 2;
    if (frames.size() < cap) frames.resize(cap);
    if (reach.capacity() < cap) reach.reserve(cap);
    reach.clear();
    DfsFrame* fr = frames.data();
    std::size_t top = 0;
    const auto limit = static_cast<std::size_t>(std::max<Index>(dense_below, 0));
    const auto abort = [&]() {
        for (const Index j : reach) mark[sz(j)] = 0;
        for (std::size_t t = 0; t < top; ++t) mark[sz(fr[t].node)] = 0;
        reach.clear();
        return false;
    };
    for (const Index s : seed) {
        if (mark[sz(s)]) continue;
        mark[sz(s)] = 1;
        {
            const auto [first, last] = children(s);
            fr[top++] = {s, first, last};
        }
        while (top > 0) {
            DfsFrame& f = fr[top - 1];
            bool descended = false;
            while (f.cur < f.end) {
                const Index i = *f.cur++;
                if (mark[sz(i)]) continue;
                if (reach.size() + top + 1 > limit) return abort();
                mark[sz(i)] = 1;
                const auto [first, last] = children(i);
                fr[top++] = {i, first, last};
                descended = true;
                break;
            }
            if (descended) continue;
            reach.push_back(f.node);
            --top;
        }
        if (reach.size() > limit) return abort();
    }
    return true;
}

// The four hypersparse solves share one shape. A post-order DFS over the edges
// "x_k feeds x_i" finds the reach of the input's nonzeros. The arithmetic is a
// SCATTER along those edges, run in the SAME elimination order as the dense
// pass in solve_*(): every position outside the reach is exactly zero and the
// dense pass skips zeros, so both perform the same operations in the same
// order and agree bit for bit. That is what lets the caller choose a path from
// history (SolveScratch's density tables) while a parallel worker with a fresh
// scratch, or a serial rerun, still returns identical numbers. Every scatter
// target is in the reach by construction, so no mark test is needed. Ordering
// the reach is a linear radix pass over positions (sort_support), not a
// comparison sort; for a reordered U the keys are elimination positions.

namespace {
// Reach (pivot positions) in ascending INDEX order, written to sc.order_.
inline void order_by_index(SolveScratch& sc, std::size_t n) {
    sc.order_ = sc.reach_;
    sort_support(sc.order_, static_cast<Index>(n), sc.sort_tmp_, sc.sort_bits_);
}
}  // namespace

// Reach in ascending U elimination order, written to sc.order_.
void BasisFactor::order_by_elimination(SolveScratch& sc) const {
    const auto n = base_->piv_val_.size();
    if (!u_reordered_) { order_by_index(sc, n); return; }
    sc.order_.clear();
    for (const Index k : sc.reach_) sc.order_.push_back(base_->upos_[sz(k)]);
    // Steps run past n once updates have appended (u_order_move_to_back).
    sort_support(sc.order_, static_cast<Index>(base_->uord_.size()), sc.sort_tmp_, sc.sort_bits_);
    for (Index& t : sc.order_) t = base_->uord_[sz(t)];
}

// L z = v, ascending. Edge k -> i for each entry l_ik of L's column k.
bool BasisFactor::sparse_lower(SolveScratch& sc, const std::vector<Index>& seed,
                               std::vector<f64>& v) const {
    if (!reach_postorder(
            seed,
            [&](Index k) {
                const Index* col = base_->l_idx_.data() + base_->l_start_[sz(k)];
                return std::pair<const Index*, const Index*>(
                    col, base_->l_idx_.data() + base_->l_start_[sz(k) + 1]);
            },
            sc.mark_, sc.dfs_frames_, sc.reach_, dense_below_))
        return false;
    order_by_index(sc, base_->piv_val_.size());
    for (const Index kk : sc.order_) {
        const auto k = sz(kk);
        const f64 zk = v[k];
        if (zk == 0.0) continue;
        for (Offset t = base_->l_start_[k]; t < base_->l_start_[k + 1]; ++t)
            v[sz(base_->l_idx_[sz(t)])] -= base_->l_val_[sz(t)] * zk;
    }
    return true;
}

// U w = v, descending elimination order. Edge k -> r for each u_rk in column k.
bool BasisFactor::sparse_upper(SolveScratch& sc, const std::vector<Index>& seed,
                               std::vector<f64>& v) const {
    if (!reach_postorder(
            seed,
            [&](Index k) {
                const Index* col = base_->u_crow_.data() + base_->u_cstart_[sz(k)];
                return std::pair<const Index*, const Index*>(col, col + base_->u_clen_[sz(k)]);
            },
            sc.mark_, sc.dfs_frames_, sc.reach_, dense_below_))
        return false;
    order_by_elimination(sc);
    for (auto it = sc.order_.rbegin(); it != sc.order_.rend(); ++it) {
        const auto k = sz(*it);
        f64 wk = v[k];
        if (wk == 0.0) continue;
        wk /= base_->piv_val_[k];
        v[k] = wk;
        const Offset beg = base_->u_cstart_[k], end = beg + base_->u_clen_[k];
        for (Offset q = beg; q < end; ++q)
            v[sz(base_->u_crow_[sz(q)])] -= base_->u_cval_[sz(q)] * wk;
    }
    return true;
}

// U' z = v, ascending elimination order. Edge k -> j for each u_kj in row k.
bool BasisFactor::sparse_upper_t(SolveScratch& sc, const std::vector<Index>& seed,
                                 std::vector<f64>& v) const {
    if (!reach_postorder(
            seed,
            [&](Index k) {
                const Index* row = base_->u_idx_.data() + base_->u_off_[sz(k)];
                return std::pair<const Index*, const Index*>(row, row + base_->u_len_[sz(k)]);
            },
            sc.mark_, sc.dfs_frames_, sc.reach_, dense_below_))
        return false;
    order_by_elimination(sc);
    for (const Index kk : sc.order_) {
        const auto k = sz(kk);
        f64 zk = v[k];
        if (zk == 0.0) continue;
        zk /= base_->piv_val_[k];
        v[k] = zk;
        const Offset beg = base_->u_off_[k], end = beg + base_->u_len_[k];
        for (Offset t = beg; t < end; ++t)
            v[sz(base_->u_idx_[sz(t)])] -= base_->u_val_[sz(t)] * zk;
    }
    return true;
}

// L' w = v, descending. Edge i -> c for each entry l_ic of L's row i.
bool BasisFactor::sparse_lower_t(SolveScratch& sc, const std::vector<Index>& seed,
                                 std::vector<f64>& v) const {
    if (!reach_postorder(
            seed,
            [&](Index k) {
                const Index* row = base_->l_col_row_.data() + base_->l_col_start_[sz(k)];
                return std::pair<const Index*, const Index*>(
                    row, base_->l_col_row_.data() + base_->l_col_start_[sz(k) + 1]);
            },
            sc.mark_, sc.dfs_frames_, sc.reach_, dense_below_))
        return false;
    order_by_index(sc, base_->piv_val_.size());
    for (auto it = sc.order_.rbegin(); it != sc.order_.rend(); ++it) {
        const auto i = sz(*it);
        const f64 wi = v[i];
        if (wi == 0.0) continue;
        const Offset beg = base_->l_col_start_[i], end = base_->l_col_start_[i + 1];
        for (Offset t = beg; t < end; ++t)
            v[sz(base_->l_col_row_[sz(t)])] -= base_->l_row_val_[sz(t)] * wi;
    }
    return true;
}

// ---------------------------------------------------------------------------
// FTRAN / BTRAN
// ---------------------------------------------------------------------------

void BasisFactor::ftran(std::vector<f64>& b, SpikeCapture* spike) const {
    const core::ScopedFlushSubnormals fp_scope;
    ftran_impl(scratch_, b, nullptr, nullptr, spike);
}

// work_ between the row etas and the U solve IS R_k...R_1 L^-1 a_q, in
// position coordinates. Record its nonzeros. `have_seed` says seed_ is a valid
// over-approximation of the support; it is a union and may repeat a position,
// hence the stamp.
void BasisFactor::capture_spike(SolveScratch& sc, SpikeCapture& out, bool have_seed) const {
    out.pos.clear();
    out.val.clear();
    const auto n = base_->piv_val_.size();
    if (have_seed) {
        if (sc.spike_stamp_.size() != n) sc.spike_stamp_.assign(n, 0);
        if (++sc.spike_gen_ == 0) {
            std::fill(sc.spike_stamp_.begin(), sc.spike_stamp_.end(), 0);
            sc.spike_gen_ = 1;
        }
        for (const Index k : sc.seed_) {
            const auto u = sz(k);
            if (u >= n || sc.spike_stamp_[u] == sc.spike_gen_) continue;
            sc.spike_stamp_[u] = sc.spike_gen_;
            if (sc.work_[u] == 0.0) continue;
            out.pos.push_back(k);
            out.val.push_back(sc.work_[u]);
        }
    } else {
        for (std::size_t k = 0; k < n; ++k) {
            if (sc.work_[k] == 0.0) continue;
            out.pos.push_back(static_cast<Index>(k));
            out.val.push_back(sc.work_[k]);
        }
    }
    out.valid = true;
}

void BasisFactor::ftran_pair(std::vector<f64>& a,
                             std::vector<f64>& b,
                             SpikeCapture* spike_a) const {
    const core::ScopedFlushSubnormals fp_scope;
    if (border_parent_) {
        if (spike_a) spike_a->clear();
        solve_border(scratch_, a, nullptr, false);
        if (&a != &b) solve_border(scratch_, b, nullptr, false);
        return;
    }
    if (&a == &b) {
        ftran(a, spike_a);
        return;
    }
    if (spike_a) spike_a->clear();
    if (m_ == 0) return;
    const auto n = base_->piv_val_.size();
    if (a.size() != n || b.size() != n) return;
    if (scratch_.factor_gen != factor_gen_) reset_scratch(scratch_);
    scratch_.eta_work_since_factor_ += 2 * eta_nnz();

    // Dense permutation. scratch_.work_ is the ordinary FTRAN scratch, so record that
    // every entry was overwritten before a later seeded solve tries to reuse
    // it. scratch_.pair_work_ is private to this dense primitive.
    for (std::size_t k = 0; k < n; ++k) {
        const auto row = sz(base_->piv_row_[k]);
        scratch_.work_[k] = a[row];
        scratch_.pair_work_[k] = b[row];
    }
    scratch_.work_all_dirty_ = true;
    scratch_.work_dirty_.clear();

    solve_lower_pair(scratch_.work_, scratch_.pair_work_);

    // Forrest--Tomlin row etas sit between L and U. Interleave the two dot
    // products while preserving the coefficient order seen by each scalar
    // solve.
    for (std::size_t t = 0; t < r_pos_.size(); ++t) {
        f64 sa = 0.0;
        f64 sb = 0.0;
        for (Offset q = r_start_[t]; q < r_start_[t + 1]; ++q) {
            const auto j = sz(r_idx_[sz(q)]);
            const f64 value = r_val_[sz(q)];
            sa += value * scratch_.work_[j];
            sb += value * scratch_.pair_work_[j];
        }
        const auto p = sz(r_pos_[t]);
        scratch_.work_[p] -= sa;
        scratch_.pair_work_[p] -= sb;
    }

    // Same point as the scalar path: L and the row etas applied, U not yet.
    // This traversal is dense throughout, so there is no seed to read.
    if (spike_a) capture_spike(scratch_, *spike_a, /*have_seed=*/false);

    solve_upper_pair(scratch_.work_, scratch_.pair_work_);
    for (std::size_t k = 0; k < n; ++k) {
        const auto slot = sz(base_->piv_slot_[k]);
        a[slot] = scratch_.work_[k];
        b[slot] = scratch_.pair_work_[k];
    }

    // Product-form etas apply oldest first. The operations for either RHS
    // occur in precisely the same order as in ftran_impl(scratch_, ).
    for (std::size_t t = 0; t < eta_p_.size(); ++t) {
        const auto p = sz(eta_p_[t]);
        const f64 pa = a[p] / eta_pivot_[t];
        const f64 pb = b[p] / eta_pivot_[t];
        for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k) {
            const auto i = sz(eta_idx_[sz(k)]);
            const f64 value = eta_val_[sz(k)];
            a[i] -= value * pa;
            b[i] -= value * pb;
        }
        a[p] = pa;
        b[p] = pb;
    }

    // Refactorization work accounting remains expressed in logical RHS work,
    // not wall-clock traversals: batching changes memory traffic, not the
    // amount of numerical update history each solution crosses.
    { const auto w_ = (4 * static_cast<Offset>(n)); scratch_.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }
    { const auto w_ = (2 * static_cast<Offset>(r_start_.back())); scratch_.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }
    { const auto w_ = (2 * static_cast<Offset>(eta_nnz())); scratch_.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }
}

bool BasisFactor::ftran_with_support(std::vector<f64>& b,
                                     std::vector<Index>& support) const {
    const core::ScopedFlushSubnormals fp_scope;
    return ftran_impl(scratch_, b, &support, nullptr);
}

bool BasisFactor::ftran_seeded_with_support(
    std::vector<f64>& b, const std::vector<Index>& seed_rows,
    std::vector<Index>& support, SpikeCapture* spike) const {
    const core::ScopedFlushSubnormals fp_scope;
    return ftran_impl(scratch_, b, &support, &seed_rows, spike);
}

bool BasisFactor::ftran_impl(SolveScratch& sc, std::vector<f64>& b,
                             std::vector<Index>* support,
                             const std::vector<Index>* seed_rows,
                             SpikeCapture* spike) const {
    if (support) support->clear();
    if (spike) spike->clear();
    if (border_parent_) return solve_border(sc, b, support, false);
    if (m_ == 0) return true;
    if (sc.factor_gen != factor_gen_) reset_scratch(sc);
    sc.eta_work_since_factor_ += eta_nnz();
    const auto n = base_->piv_val_.size();
    bool seed_is_sparse = n >= 64;
    sc.seed_.clear();

    // SEEDED permute: the caller has declared the rows where b may be nonzero,
    // so only those need moving into sc.work_ -- O(|seed|) instead of the O(n)
    // gather below. Valid only because sc.work_ is known zero outside the
    // positions this path writes (sc.work_dirty_ / sc.work_all_dirty_), and because
    // the caller guarantees b is zero outside seed_rows. Any surprise (no
    // pivot for a declared row, oversized seed) abandons the attempt and falls
    // through to the classic path, which overwrites sc.work_ completely and is
    // therefore always safe to reach from a half-finished seeded permute.
    bool seeded = false;
    if (seed_rows != nullptr && seed_is_sparse && !seed_rows->empty() &&
        4 * seed_rows->size() <= n && base_->rpos_.size() >= sz(m_)) {
        if (sc.work_all_dirty_) {
            std::fill(sc.work_.begin(), sc.work_.end(), 0.0);
            sc.work_all_dirty_ = false;
        } else {
            for (const Index k : sc.work_dirty_) sc.work_[sz(k)] = 0.0;
        }
        sc.work_dirty_.clear();
        seeded = true;
        for (const Index i : *seed_rows) {
            if (i < 0 || sz(i) >= base_->rpos_.size()) { seeded = false; break; }
            const Index k = base_->rpos_[sz(i)];
            if (k < 0 || sz(k) >= n) { seeded = false; break; }
            const f64 value = b[sz(i)];
            if (value == 0.0) continue;
            // A caller may declare the same row twice (a column scatter that
            // accumulates); only the first sighting enters the seed.
            if (sc.work_[sz(k)] == 0.0) {
                sc.seed_.push_back(k);
                sc.work_dirty_.push_back(k);
            }
            sc.work_[sz(k)] = value;
        }
        if (seeded && 4 * sc.seed_.size() > n) seeded = false;
        if (!seeded) sc.seed_.clear();
    }

    // Classic path: permute the RHS and collect its sparse seed in the SAME
    // pass. The old path copied all n entries, scanned all n again to count
    // nonzeros, then scanned all n a third time to materialize the seed. Once
    // more than n/4 nonzeros have been seen the sparse path cannot be
    // selected, so stop tracking immediately while completing only the
    // mandatory permutation.
    if (!seeded) {
        for (std::size_t k = 0; k < n; ++k) {
            const f64 value = b[sz(base_->piv_row_[k])];
            sc.work_[k] = value;
            if (seed_is_sparse && value != 0.0) {
                sc.seed_.push_back(static_cast<Index>(k));
                if (4 * sc.seed_.size() > n) {
                    seed_is_sparse = false;
                    sc.seed_.clear();
                }
            }
        }
        // sc.work_ now holds a value at every position; a later seeded call must
        // clear all of it before it can assume zeros.
        sc.work_all_dirty_ = true;
        sc.work_dirty_.clear();
    }

    // Collect a seed after the first triangular solve. Abort the scan as soon
    // as the existing 25%-density gate is exceeded; the dense fallback does
    // not need the remaining nonzero count.
    const auto collect_seed = [&]() {
        sc.seed_.clear();
        if (n < 64) return false;
        for (std::size_t k = 0; k < n; ++k) {
            if (sc.work_[k] == 0.0) continue;
            sc.seed_.push_back(static_cast<Index>(k));
            if (4 * sc.seed_.size() > n) {
                sc.seed_.clear();
                return false;
            }
        }
        return true;
    };

    // Predict from the history of solves with inputs of this size whether
    // the output will be sparse enough for the reach-set path to pay.
    const std::size_t in_bucket = seed_is_sparse
        ? static_cast<std::size_t>(std::bit_width(sc.seed_.size())) : 0;
    const bool predicted_dense = seed_is_sparse &&
        (solve_path_ == SolvePath::Dense ||
         (solve_path_ == SolvePath::Auto &&
          sc.ftran_density_[in_bucket] > kHyperOutputDensity));
    if (predicted_dense) seed_is_sparse = false;
    const bool learn = seed_is_sparse || predicted_dense;

    // L-solve (sparse or dense). The seed is built only when the input is
    // sparse enough to plausibly win.
    bool sp = false;
    bool lower_sparse = false;
    if (seed_is_sparse) {
        if (sparse_lower(sc, sc.seed_, sc.work_)) {
            sp = true;
            lower_sparse = true;
            // The solve wrote sc.work_ over its reach; record that so a later
            // seeded call knows exactly what to clear.
            sc.work_dirty_.insert(sc.work_dirty_.end(), sc.reach_.begin(), sc.reach_.end());
            // This reach is an exact over-approximation of the U input support.
            // Preserve it before sparse_upper(sc, ) reuses sc.reach_, avoiding the
            // collect_seed() O(n) scan between the two triangular solves.
            sc.seed_ = sc.reach_;
            for (const Index k : sc.reach_) sc.mark_[sz(k)] = 0;
        }
    }
    if (!sp) {
        solve_lower(sc.work_);
        sc.work_all_dirty_ = true;      // dense pass touched every position
    }
    { const auto w_ = (sp ? static_cast<Offset>(sc.reach_.size()) : static_cast<Offset>(n)); sc.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }

    // Row etas, between L and U: B^-1 = U^-1 M_k ... M_1 L^-1. A position the
    // pass turns nonzero has to join sc.work_dirty_, or the next seeded call will
    // trust it to be zero and read a stale value. collect_seed() below rescans
    // sc.work_ in full, so the U seed needs no separate bookkeeping.
    if (!r_pos_.empty()) {
        if (lower_sparse) {
            sc.row_eta_touched_.clear();
            apply_row_etas_ftran(sc.work_, &sc.row_eta_touched_);
            sc.work_dirty_.insert(sc.work_dirty_.end(),
                               sc.row_eta_touched_.begin(), sc.row_eta_touched_.end());
            sc.seed_.insert(sc.seed_.end(), sc.row_eta_touched_.begin(),
                         sc.row_eta_touched_.end());
        } else {
            // Dense after L: no support bookkeeping; branchless eta apply.
            apply_row_etas_ftran(sc.work_, nullptr);
        }
        { const auto w_ = (static_cast<Offset>(r_start_.back())); sc.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }
    }

    // The spike is exactly this state: L and the row etas applied, U not yet.
    // Taken before collect_seed(), which reuses sc.seed_.
    if (spike) capture_spike(sc, *spike, lower_sparse);

    // U-solve (sparse or dense) + scatter.
    sp = false;
    const bool have_upper_seed =
        !predicted_dense && (lower_sparse || collect_seed());
    if (have_upper_seed) {
        if (sparse_upper(sc, sc.seed_, sc.work_)) {
            sp = true;                   // marks stay set for the scatter
            sc.work_dirty_.insert(sc.work_dirty_.end(), sc.reach_.begin(), sc.reach_.end());
        }
    }
    // Slot-space membership stamps for the support (O(1) tests in the eta
    // loop below). A fresh generation per call; slots are stamped as they
    // enter the support.
    // The dense pass reports a support too when one is asked for: it writes
    // every slot anyway, so collecting the nonzeros is free, and a caller
    // that loses the support falls back to O(m) and O(nnz(A)) loops of its
    // own (a 17%-dense rho then priced the pivotal row column by column).
    if (support) {
        if (++sc.support_gen_ == 0) ++sc.support_gen_;   // 0 is the "unstamped" value
    }
    const std::uint32_t sgen = support ? sc.support_gen_ : 0;
    bool support_sorted = false;
    if (sp) {
        // Only the U reach can be nonzero: the fused scatter zeroes the rest,
        // keeping the all-m-outputs-written contract of the dense path. The
        // support variant writes ONLY the reach (values identical: sc.mark_ is
        // set exactly on the reach) and records it, leaving other entries
        // stale for a caller that resets by support.
        if (support) {
            support->reserve(sc.reach_.size() + eta_p_.size());
            for (const Index k : sc.reach_) {
                const auto slot = sz(base_->piv_slot_[sz(k)]);
                b[slot] = sc.work_[sz(k)];
                support->push_back(slot);
                sc.support_stamp_[slot] = sgen;
            }
        } else {
            for (std::size_t k = 0; k < n; ++k)
                b[sz(base_->piv_slot_[k])] = sc.mark_[k] ? sc.work_[k] : 0.0;
        }
        for (const Index k : sc.reach_) sc.mark_[sz(k)] = 0;
    } else {
        solve_upper(sc.work_);
        sc.work_all_dirty_ = true;      // dense pass touched every position
        std::size_t nonzeros = 0;
        if (support && n == sz(m_)) {
            // Walk the OUTPUT in slot order: the support comes out sorted, so
            // it needs no sort, and the append is branch-free -- a branch on
            // a value's zeroness mispredicts on exactly the mid-density
            // outputs this path now serves.
            support->resize(n);
            Index* out = support->data();
            for (std::size_t slot = 0; slot < n; ++slot) {
                const f64 value = sc.work_[sz(base_->cpos_[slot])];
                b[slot] = value;
                out[nonzeros] = static_cast<Index>(slot);
                nonzeros += value != 0.0;
            }
            support->resize(nonzeros);
            for (const Index slot : *support) sc.support_stamp_[sz(slot)] = sgen;
            support_sorted = true;
        } else {
            for (std::size_t k = 0; k < n; ++k) {
                const f64 value = sc.work_[k];
                const auto slot = sz(base_->piv_slot_[k]);
                b[slot] = value;
                if (value == 0.0) continue;
                ++nonzeros;
                if (support) {
                    support->push_back(static_cast<Index>(slot));
                    sc.support_stamp_[slot] = sgen;
                }
            }
        }
        if (learn)
            learn_density(sc.ftran_density_, in_bucket,
                          static_cast<f64>(nonzeros) / static_cast<f64>(n));
    }
    ++sc.triangular_solves_;
    if (sp) ++sc.hypersparse_solves_;
    if (sp && learn)
        learn_density(sc.ftran_density_, in_bucket,
                      static_cast<f64>(sc.reach_.size()) / static_cast<f64>(n));
    { const auto w_ = (sp ? static_cast<Offset>(sc.reach_.size()) : static_cast<Offset>(n)); sc.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }
    { const auto w_ = (static_cast<Offset>(eta_nnz())); sc.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }

    const std::size_t support_before_etas = support ? support->size() : 0;
    // B_k^-1 = E_k^-1 ... E_1^-1 B_0^-1, so the etas apply oldest first.
    // An eta with b[p] == 0 is the identity on b (E^-1 x = x when x_p = 0),
    // so hypersparse states skip the eta arithmetic entirely. In partial-
    // write mode the base solve wrote ONLY the support: a nonzero b[p]
    // outside the support is a STALE input-scatter value (the caller's
    // entering column landed on this slot index), not an output -- the
    // output is zero there, so the eta is the identity there and must be
    // skipped. Slice entries outside the support are stale too, so their
    // update is an ASSIGNMENT from zero, not an accumulation. Both tests
    // are O(1) via the stamps set during the scatter.
    for (std::size_t t = 0; t < eta_p_.size(); ++t) {
        const auto p = sz(eta_p_[t]);
        // Keep the historical zero skip FIRST: a zero b[p] makes the eta the
        // identity regardless of mode, and skipping the slice scan is what
        // keeps hypersparse states cheap.
        if (b[p] == 0.0) continue;
        if (sgen != 0 && sc.support_stamp_[p] != sgen) continue;   // stale, not output
        const f64 pv = b[p] / eta_pivot_[t];
        for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k) {
            const auto i = sz(eta_idx_[sz(k)]);
            // update() excludes p when constructing every eta slice.
            if (sgen != 0 && sc.support_stamp_[i] != sgen) {
                b[i] = -eta_val_[sz(k)] * pv;          // stale/zero: assign
                if (support) {
                    support->push_back(i);
                    sc.support_stamp_[i] = sgen;
                }
            } else {
                b[i] -= eta_val_[sz(k)] * pv;          // genuine accumulation
            }
        }
        b[p] = pv;
        // p passed the stamp gate, so it is already in the support from the
        // scatter -- no push needed (a duplicate would double-apply every
        // downstream per-pivot update at p).
    }
    // sp == false: the dense path wrote every slot (caller must treat b as
    // fully overwritten and reset with a full clear next time). Sorting the
    // support preserves the dense loops' ascending visiting order, so
    // ratio-test tie-breaking cannot change (same rationale as
    // btran_impl).
    if (support && !(support_sorted && support->size() == support_before_etas))
        sort_support(*support, m_, sc.sort_tmp_, sc.sort_bits_);
    // Without a support request, sp == false means the dense path wrote every
    // slot. With one, the support is exact either way: the dense path wrote
    // zeros everywhere else, which a caller resetting by support also accepts.
    return sp || support != nullptr;
}

void BasisFactor::btran(std::vector<f64>& d) const {
    const core::ScopedFlushSubnormals fp_scope;
    (void)btran_impl(scratch_, d, nullptr, nullptr);
}

bool BasisFactor::btran_with_support(std::vector<f64>& d,
                                     std::vector<Index>& support) const {
    const core::ScopedFlushSubnormals fp_scope;
    return btran_impl(scratch_, d, &support, nullptr);
}

bool BasisFactor::btran_seeded_with_support(
    std::vector<f64>& d, const std::vector<Index>& seed_slots,
    std::vector<Index>& support, RowEtaCapture* row_eta) const {
    const core::ScopedFlushSubnormals fp_scope;
    return btran_impl(scratch_, d, &support, &seed_slots, row_eta);
}

bool BasisFactor::btran_impl(SolveScratch& sc, std::vector<f64>& d,
                             std::vector<Index>* support,
                             const std::vector<Index>* seed_slots,
                             RowEtaCapture* row_eta) const {
    if (support) support->clear();
    if (row_eta) row_eta->clear();
    // A row-eta capture describes U alone: it is the state after the U' solve
    // of a unit input, so it needs a unit input that no product-form eta has
    // changed. Decided before anything is touched; the capture itself happens
    // between the U' solve and the row etas below.
    const bool capture_row_eta = row_eta != nullptr && !border_parent_ &&
        eta_p_.empty() && seed_slots != nullptr && seed_slots->size() == 1 &&
        (*seed_slots)[0] >= 0 && sz((*seed_slots)[0]) < d.size() &&
        d[sz((*seed_slots)[0])] == 1.0;
    if (border_parent_) return solve_border(sc, d, support, true);
    if (m_ == 0) return true;
    if (sc.factor_gen != factor_gen_) reset_scratch(sc);
    sc.eta_work_since_factor_ += eta_nnz();

    // B_k^-T = B_0^-T E_1^-T ... E_k^-T, so the etas apply newest first.
    //
    // Firing-set path for sparse inputs: an eta is an EXACT no-op when every
    // slot it reads (its pivot p and its slice) is zero at its turn, so it
    // suffices to process the etas reachable from the input's nonzero
    // positions through the reverse incidence rev_ - a skipped eta writes
    // zero to an already-zero position, and processing order among fired
    // etas stays newest-first, so values are bit-identical to the full scan.
    // Nonzeros can only originate at (a) the input seed and (b) a FIRED
    // eta's write position; a 0→nonzero write at slot p can newly fire only
    // etas OLDER than the writer (newer ones have already been processed and
    // legitimately saw the pre-write value), so pushes filter on u < t.
    bool used_firing = false;
    // Eta-written input slots must be cleared before a seeded sparse scatter
    // turns this vector from slot space into row space. Keep the exact list in
    // the firing path; the eta-poor fallback records its (small) pivot list.
    sc.fire_touched_.clear();
    if (eta_p_.size() >= 8 && rev_.size() == sz(m_)) {
        std::size_t nz = 0;
        std::vector<Index>& input_seed = sc.fire_input_seed_;   // reused buffer
        input_seed.clear();
        bool too_dense = false;
        if (seed_slots != nullptr && seed_slots->size() * 8 <= sz(m_)) {
            // The caller declared where d is nonzero (the dual's rho is a
            // single unit slot), so the O(m) scan below is unnecessary. Zeros
            // inside the declared seed are harmless: an eta reached from a
            // zero slot simply computes the same value it would have anyway.
            for (const Index i : *seed_slots) {
                if (i < 0 || sz(i) >= sz(m_)) { too_dense = true; break; }
                input_seed.push_back(i);
            }
        } else {
            for (Index i = 0; i < m_; ++i) {
                if (d[sz(i)] == 0.0) continue;
                ++nz;
                if (nz * 8 > sz(m_)) { too_dense = true; break; }
                input_seed.push_back(i);
            }
        }
        if (!too_dense) {
            used_firing = true;
            if (sc.fire_stamp_.size() < eta_p_.size())
                sc.fire_stamp_.assign(eta_p_.size(), 0);
            if (++sc.fire_gen_ == 0) {
                std::fill(sc.fire_stamp_.begin(), sc.fire_stamp_.end(), 0);
                ++sc.fire_gen_;
            }
            std::vector<Index>& heap = sc.fire_heap_;   // reused buffer
            heap.clear();
            const auto push = [&](Index t) {
                if (sc.fire_stamp_[sz(t)] == sc.fire_gen_) return;
                sc.fire_stamp_[sz(t)] = sc.fire_gen_;
                heap.push_back(t);
                std::push_heap(heap.begin(), heap.end());
            };
            for (const Index i : input_seed)
                for (const Index t : rev_[sz(i)]) push(t);
            Offset fired_entries = 0;
            const std::size_t n_eta = eta_p_.size();
            while (!heap.empty()) {
                std::pop_heap(heap.begin(), heap.end());
                const Index t = heap.back();
                heap.pop_back();
                const auto p = sz(eta_p_[sz(t)]);
                f64 s = d[p];
                for (Offset k = eta_start_[sz(t)]; k < eta_start_[sz(t) + 1]; ++k) {
                    const auto i = sz(eta_idx_[sz(k)]);
                    // update() excludes p when constructing every eta slice.
                    s -= eta_val_[sz(k)] * d[i];
                }
                const f64 old = d[p];
                d[p] = s / eta_pivot_[sz(t)];
                // This slot now carries an eta-produced value that the U'
                // solve must see, so the seeded permute has to include it.
                sc.fire_touched_.push_back(static_cast<Index>(p));
                fired_entries += 1 + static_cast<Offset>(
                                            eta_start_[sz(t) + 1] - eta_start_[sz(t)]);
                if (d[p] != 0.0 && old == 0.0) {
                    // Only OLDER readers can newly fire on this write.
                    for (const Index u : rev_[p])
                        if (u < t) push(u);
                }
            }
            (void)n_eta;
            { const auto w_ = (fired_entries); sc.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }
        }
    }
    if (!used_firing) {
        // Dense-input / eta-poor fallback: the transposed eta E^-T only
        // writes position p; every other entry is read-only. The eta index
        // scan is still required by product form, so keep this loop
        // branch-free; checking every d[i] for zero was measured as a
        // regression on eta-heavy Netlib instances.
        for (std::size_t t = eta_p_.size(); t-- > 0;) {
            const auto p = sz(eta_p_[t]);
            f64 s = d[p];
            for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k) {
                const auto i = sz(eta_idx_[sz(k)]);
                // update() excludes p when constructing every eta slice.
                s -= eta_val_[sz(k)] * d[i];
            }
            d[p] = s / eta_pivot_[t];
            if (seed_slots != nullptr)
                sc.fire_touched_.push_back(static_cast<Index>(p));
        }
        { const auto w_ = (static_cast<Offset>(eta_nnz())); sc.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }
    }

    const auto n = base_->piv_val_.size();
    // Fuse the basis-slot permutation with sparse-seed construction, exactly
    // as in ftran(). Etas have already been applied above, so this observes
    // the true input pattern to the base U' solve.
    bool seed_is_sparse = n >= 64;
    sc.seed_.clear();

    // SEEDED permute (slot space, via base_->cpos_). Same contract and same fallback
    // discipline as ftran_impl's. Note the eta loops above may have written
    // slots OUTSIDE the declared input seed, so the seed used here is the
    // caller's slots PLUS every eta pivot that fired -- collected during the
    // firing-set pass. Eta-poor fallback pivots are recorded above as well, so
    // an empty or short eta file still avoids the O(m) permutation scan.
    bool seeded = false;
    if (seed_slots != nullptr && seed_is_sparse &&
        base_->cpos_.size() >= sz(m_)) {
        const std::size_t declared = seed_slots->size() + sc.fire_touched_.size();
        if (declared > 0 && 4 * declared <= n) {
            if (sc.work_all_dirty_) {
                std::fill(sc.work_.begin(), sc.work_.end(), 0.0);
                sc.work_all_dirty_ = false;
            } else {
                for (const Index k : sc.work_dirty_) sc.work_[sz(k)] = 0.0;
            }
            sc.work_dirty_.clear();
            seeded = true;
            const auto take = [&](Index i) {
                if (i < 0 || sz(i) >= base_->cpos_.size()) { seeded = false; return; }
                const Index k = base_->cpos_[sz(i)];
                if (k < 0 || sz(k) >= n) { seeded = false; return; }
                const f64 value = d[sz(i)];
                if (value == 0.0) return;
                if (sc.work_[sz(k)] == 0.0) {
                    sc.seed_.push_back(k);
                    sc.work_dirty_.push_back(k);
                }
                sc.work_[sz(k)] = value;
            };
            for (const Index i : *seed_slots)   { if (!seeded) break; take(i); }
            for (const Index i : sc.fire_touched_) { if (!seeded) break; take(i); }
            if (seeded && 4 * sc.seed_.size() > n) seeded = false;
            if (!seeded) sc.seed_.clear();
        }
    }

    if (!seeded) {
        for (std::size_t k = 0; k < n; ++k) {
            const f64 value = d[sz(base_->piv_slot_[k])];
            sc.work_[k] = value;
            if (seed_is_sparse && value != 0.0) {
                sc.seed_.push_back(static_cast<Index>(k));
                if (4 * sc.seed_.size() > n) {
                    seed_is_sparse = false;
                    sc.seed_.clear();
                }
            }
        }
        sc.work_all_dirty_ = true;
        sc.work_dirty_.clear();
    }
    const auto collect_seed = [&]() {
        sc.seed_.clear();
        if (n < 64) return false;
        for (std::size_t k = 0; k < n; ++k) {
            if (sc.work_[k] == 0.0) continue;
            sc.seed_.push_back(static_cast<Index>(k));
            if (4 * sc.seed_.size() > n) {
                sc.seed_.clear();
                return false;
            }
        }
        return true;
    };

    // Same prediction as ftran_impl, from this family's history.
    const std::size_t in_bucket = seed_is_sparse
        ? static_cast<std::size_t>(std::bit_width(sc.seed_.size())) : 0;
    const bool predicted_dense = seed_is_sparse &&
        (solve_path_ == SolvePath::Dense ||
         (solve_path_ == SolvePath::Auto &&
          sc.btran_density_[in_bucket] > kHyperOutputDensity));
    if (predicted_dense) seed_is_sparse = false;
    const bool learn = seed_is_sparse || predicted_dense;

    // U'-solve (sparse or dense).
    bool sp = false;
    bool upper_t_sparse = false;
    if (seed_is_sparse) {
        if (sparse_upper_t(sc, sc.seed_, sc.work_)) {
            sp = true;
            upper_t_sparse = true;
            sc.work_dirty_.insert(sc.work_dirty_.end(), sc.reach_.begin(), sc.reach_.end());
            // Preserve the U' reach as the L' input seed. Row etas below may
            // append new nonzeros, but no full collect_seed() scan is needed.
            sc.seed_ = sc.reach_;
            for (const Index k : sc.reach_) sc.mark_[sz(k)] = 0;
        }
    }
    if (!sp) {
        solve_upper_t(sc.work_);
        sc.work_all_dirty_ = true;      // dense pass touched every position
    }
    { const auto w_ = (sp ? static_cast<Offset>(sc.reach_.size()) : static_cast<Offset>(n)); sc.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }

    // The row eta of the input slot: sc.work_ is e_p^T U^-1 right here, in
    // position coordinates, before the row etas and L' change it. Nonzeros
    // only, in ascending position order on both paths (the reach-set solve
    // delivers them in DFS order, the dense pass in position order);
    // update_ft()'s own solve sorts the same way, so the eta a later solve
    // sums is the same whichever path produced it.
    if (capture_row_eta) {
        const Index slot = (*seed_slots)[0];
        const Index ps = base_->cpos_.size() > sz(slot) ? base_->cpos_[sz(slot)] : -1;
        if (ps >= 0 && sz(ps) < n) {
            row_eta->pos.clear();
            row_eta->val.clear();
            if (upper_t_sparse) {
                for (const Index k : sc.reach_)
                    if (sc.work_[sz(k)] != 0.0) row_eta->pos.push_back(k);
                sort_support(row_eta->pos, static_cast<Index>(n), sc.sort_tmp_, sc.sort_bits_);
            } else {
                for (std::size_t k = 0; k < n; ++k)
                    if (sc.work_[k] != 0.0) row_eta->pos.push_back(static_cast<Index>(k));
            }
            row_eta->val.reserve(row_eta->pos.size());
            for (const Index k : row_eta->pos) row_eta->val.push_back(sc.work_[sz(k)]);
            row_eta->pivot_pos = ps;
            row_eta->factor_gen = factor_gen_;
            row_eta->n_row_etas = static_cast<Index>(r_pos_.size());
            row_eta->valid = true;
        }
    }

    // Row etas transposed, between U' and L': B^-T = L^-T M_1^T ... M_k^T U^-T.
    // Newest first, mirroring ftran's oldest-first order.
    if (!r_pos_.empty()) {
        if (upper_t_sparse) {
            sc.row_eta_touched_.clear();
            apply_row_etas_btran(sc.work_, &sc.row_eta_touched_);
            sc.work_dirty_.insert(sc.work_dirty_.end(),
                               sc.row_eta_touched_.begin(), sc.row_eta_touched_.end());
            sc.seed_.insert(sc.seed_.end(), sc.row_eta_touched_.begin(),
                         sc.row_eta_touched_.end());
        } else {
            apply_row_etas_btran(sc.work_, nullptr);
        }
        { const auto w_ = (static_cast<Offset>(r_start_.back())); sc.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }
    }

    // L'-solve (sparse or dense) + scatter.
    sp = false;
    const bool have_lower_t_seed =
        !predicted_dense && (upper_t_sparse || collect_seed());
    if (have_lower_t_seed) {
        if (sparse_lower_t(sc, sc.seed_, sc.work_)) {
            sp = true;
            sc.work_dirty_.insert(sc.work_dirty_.end(), sc.reach_.begin(), sc.reach_.end());
        }
    }
    if (sp) {
        const bool partial_seeded_scatter = support && seed_slots != nullptr;
        if (partial_seeded_scatter) {
            // At entry the caller guarantees zeros outside its declared seed.
            // Product-form etas can write additional pivot slots before the
            // triangular solves; sc.fire_touched_ is precisely that set. Clear
            // both input sets BEFORE writing row-indexed outputs, so an index
            // shared by an input slot and an output row receives the output.
            for (const Index i : *seed_slots)
                if (i >= 0 && sz(i) < d.size()) d[sz(i)] = 0.0;
            for (const Index i : sc.fire_touched_)
                if (i >= 0 && sz(i) < d.size()) d[sz(i)] = 0.0;

            support->reserve(sc.reach_.size());
            for (const Index k : sc.reach_) {
                if (sc.work_[sz(k)] == 0.0) continue;
                const Index row = base_->piv_row_[sz(k)];
                d[sz(row)] = sc.work_[sz(k)];
                support->push_back(row);
            }
            // The dual pivotal-row kernel historically visits rho in original
            // row order. Preserve that floating-point accumulation order so
            // exposing the sparse support cannot alter pivot decisions.
            sort_support(*support, m_, sc.sort_tmp_, sc.sort_bits_);
        } else if (support) {
            support->reserve(sc.reach_.size());
            for (const Index k : sc.reach_)
                if (sc.work_[sz(k)] != 0.0)
                    support->push_back(base_->piv_row_[sz(k)]);
            sort_support(*support, m_, sc.sort_tmp_, sc.sort_bits_);
        }
        if (!partial_seeded_scatter) {
            for (std::size_t k = 0; k < n; ++k)
                d[sz(base_->piv_row_[k])] = sc.mark_[k] ? sc.work_[k] : 0.0;
        }
        for (const Index k : sc.reach_) sc.mark_[sz(k)] = 0;
    } else {
        solve_lower_t(sc.work_);
        sc.work_all_dirty_ = true;      // dense pass touched every position
        std::size_t nonzeros = 0;
        if (support && n == sz(m_)) {
            // Row order, branch-free and born sorted: see ftran_impl.
            support->resize(n);
            Index* out = support->data();
            for (std::size_t row = 0; row < n; ++row) {
                const f64 value = sc.work_[sz(base_->rpos_[row])];
                d[row] = value;
                out[nonzeros] = static_cast<Index>(row);
                nonzeros += value != 0.0;
            }
            support->resize(nonzeros);
        } else {
            for (std::size_t k = 0; k < n; ++k) {
                const f64 value = sc.work_[k];
                d[sz(base_->piv_row_[k])] = value;
                if (value == 0.0) continue;
                ++nonzeros;
                if (support) support->push_back(base_->piv_row_[k]);   // see ftran_impl
            }
            if (support) sort_support(*support, m_, sc.sort_tmp_, sc.sort_bits_);
        }
        if (learn)
            learn_density(sc.btran_density_, in_bucket,
                          static_cast<f64>(nonzeros) / static_cast<f64>(n));
    }
    ++sc.triangular_solves_;
    if (sp) ++sc.hypersparse_solves_;
    if (sp && learn)
        learn_density(sc.btran_density_, in_bucket,
                      static_cast<f64>(sc.reach_.size()) / static_cast<f64>(n));
    { const auto w_ = (sp ? static_cast<Offset>(sc.reach_.size()) : static_cast<Offset>(n)); sc.work_since_factor_ += w_; core::work_add(static_cast<std::uint64_t>(w_), ::forge::core::WorkTag::LuSolve); }
    // Eta work was already accounted inside the firing-set / plain loops
    // above (actual entries touched in the firing path). As in ftran_impl, a
    // requested support is exact on both paths.
    return sp || support != nullptr;
}

// ---------------------------------------------------------------------------
// Product-form update
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Internal consistency check (test support -- never called on the solve path)
// ---------------------------------------------------------------------------

bool BasisFactor::check_invariants(std::string* why) const {
    const auto fail = [&](const std::string& msg) {
        if (why) *why = msg;
        return false;
    };
    const auto n = base_->piv_val_.size();
    if (border_parent_) {
        const Index old = border_parent_->dimension();
        if (projected_) {
            if (!valid_ || retained_rows_.size() != sz(m_) || retained_slots_.size() != sz(m_) || old < m_)
                return fail("invalid projection shape");
            std::vector<char> rows(sz(old), false), slots(sz(old), false);
            for (Index i = 0; i < m_; ++i) {
                const Index r = retained_rows_[sz(i)], s = retained_slots_[sz(i)];
                if (r < 0 || r >= old || s < 0 || s >= old || rows[sz(r)] || slots[sz(s)])
                    return fail("invalid projection maps");
                rows[sz(r)] = slots[sz(s)] = true;
            }
        } else {
        if (!valid_ || old >= m_ || border_ptr_.size() != sz(m_ - old) + 1 ||
            border_ptr_.front() != 0 || border_ptr_.back() != static_cast<Offset>(border_value_.size()) ||
            border_slot_.size() != border_value_.size()) return fail("invalid border shape");
        for (Index i = 0; i < m_ - old; ++i)
            if (border_ptr_[sz(i)] > border_ptr_[sz(i) + 1]) return fail("invalid border offsets");
        for (std::size_t k = 0; k < border_slot_.size(); ++k)
            if (border_slot_[k] < 0 || border_slot_[k] >= old || !std::isfinite(border_value_[k]))
                return fail("invalid border entry");
        }
        if (eta_start_.empty() || eta_start_.front() != 0 || eta_start_.size() != eta_p_.size() + 1 || eta_pivot_.size() != eta_p_.size() ||
            eta_start_.back() != static_cast<Offset>(eta_val_.size()) ||
            eta_idx_.size() != eta_val_.size()) return fail("invalid border eta shape");
        for (std::size_t t = 0; t < eta_p_.size(); ++t)
            if (eta_start_[t] > eta_start_[t + 1] || eta_p_[t] < 0 || eta_p_[t] >= m_ || !std::isfinite(eta_pivot_[t]) || eta_pivot_[t] == 0)
                return fail("invalid border eta pivot");
        for (std::size_t k = 0; k < eta_idx_.size(); ++k)
            if (eta_idx_[k] < 0 || eta_idx_[k] >= m_ || !std::isfinite(eta_val_[k]))
                return fail("invalid border eta entry");
        return border_parent_->check_invariants(why);
    }
    if (n != sz(m_)) return fail("base_->piv_val_ size != m");

    // Pivot maps are permutations, and base_->rpos_/base_->cpos_ invert them.
    {
        std::vector<char> seen_row(n, 0), seen_slot(n, 0);
        for (std::size_t k = 0; k < n; ++k) {
            const auto r = base_->piv_row_[k], c = base_->piv_slot_[k];
            if (r < 0 || sz(r) >= n || seen_row[sz(r)]) return fail("base_->piv_row_ not a permutation");
            if (c < 0 || sz(c) >= n || seen_slot[sz(c)]) return fail("base_->piv_slot_ not a permutation");
            seen_row[sz(r)] = seen_slot[sz(c)] = 1;
            if (base_->rpos_.size() >= n && base_->rpos_[sz(r)] != static_cast<Index>(k))
                return fail("base_->rpos_ does not invert base_->piv_row_");
            if (base_->cpos_.size() >= n && base_->cpos_[sz(c)] != static_cast<Index>(k))
                return fail("base_->cpos_ does not invert base_->piv_slot_");
        }
    }

    // Elimination order and its inverse.
    // The order is append-only between factorizations: n live steps plus one
    // hole (-1) per update, each hole listed in u_dead_steps_.
    if (base_->uord_.size() != n + base_->u_dead_steps_.size() || base_->upos_.size() != n)
        return fail("base_->uord_/base_->upos_ size");
    {
        std::size_t holes = 0;
        for (std::size_t t = 0; t < base_->uord_.size(); ++t) {
            const auto k = base_->uord_[t];
            if (k == -1) { ++holes; continue; }
            if (k < 0 || sz(k) >= n) return fail("base_->uord_ out of range");
            if (base_->upos_[sz(k)] != static_cast<Index>(t)) return fail("base_->upos_ does not invert base_->uord_");
        }
        if (holes != base_->u_dead_steps_.size()) return fail("uord_ holes disagree with u_dead_steps_");
        for (const Index d : base_->u_dead_steps_)
            if (d < 0 || sz(d) >= base_->uord_.size() || base_->uord_[sz(d)] != -1)
                return fail("u_dead_steps_ names a live step");
    }

    // U row store: in-range, no duplicate columns, no self-entry, triangular
    // with respect to the elimination order, inside its capacity.
    std::vector<char> hit(n, 0);
    Offset live = 0;
    for (std::size_t k = 0; k < n; ++k) {
        if (base_->u_len_[k] > base_->u_cap_[k]) return fail("base_->u_len_ exceeds base_->u_cap_");
        const auto beg = sz(base_->u_off_[k]);
        if (beg + sz(base_->u_cap_[k]) > base_->u_idx_.size() || beg + sz(base_->u_cap_[k]) > base_->u_val_.size())
            return fail("U row segment runs past the arrays");
        for (Index t = 0; t < base_->u_len_[k]; ++t) {
            const auto j = base_->u_idx_[beg + sz(t)];
            if (j < 0 || sz(j) >= n) return fail("U row column out of range");
            if (sz(j) == k) return fail("U row carries its own diagonal");
            if (hit[sz(j)]) return fail("U row repeats a column");
            hit[sz(j)] = 1;
            if (base_->upos_[sz(j)] <= base_->upos_[k])
                return fail("U is not triangular for the elimination order");
        }
        for (Index t = 0; t < base_->u_len_[k]; ++t) hit[sz(base_->u_idx_[beg + sz(t)])] = 0;
        live += static_cast<Offset>(base_->u_len_[k]);
    }

    // U column store holds exactly the same entries with the same values.
    {
        Offset clive = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (base_->u_clen_[j] > base_->u_ccap_[j]) return fail("base_->u_clen_ exceeds base_->u_ccap_");
            const auto beg = sz(base_->u_cstart_[j]);
            if (beg + sz(base_->u_ccap_[j]) > base_->u_crow_.size() || beg + sz(base_->u_ccap_[j]) > base_->u_cval_.size())
                return fail("U column segment runs past the arrays");
            for (Index t = 0; t < base_->u_clen_[j]; ++t) {
                const auto r = base_->u_crow_[beg + sz(t)];
                if (r < 0 || sz(r) >= n) return fail("U column row out of range");
                if (hit[sz(r)]) return fail("U column repeats a row");
                hit[sz(r)] = 1;
                // The same entry must be present, with the same value, in the
                // row store. This is the check that catches a mirror drifting.
                bool found = false;
                const auto rbeg = sz(base_->u_off_[sz(r)]);
                for (Index q = 0; q < base_->u_len_[sz(r)]; ++q) {
                    if (base_->u_idx_[rbeg + sz(q)] != static_cast<Index>(j)) continue;
                    if (base_->u_val_[rbeg + sz(q)] != base_->u_cval_[beg + sz(t)])
                        return fail("U mirrors disagree on a value");
                    found = true;
                    break;
                }
                if (!found) return fail("U column entry missing from the row store");
                // The twin links must name each other's copy of this entry.
                const Offset cs = base_->u_cstart_[j] + t;
                if (base_->u_cmirror_.size() != base_->u_crow_.size() ||
                    base_->u_rmirror_.size() != base_->u_idx_.size())
                    return fail("U twin-link arrays sized wrong");
                const Index rel = base_->u_cmirror_[sz(cs)];
                if (rel < 0 || rel >= base_->u_len_[sz(r)])
                    return fail("U twin link outside its row");
                const Offset rs = base_->u_off_[sz(r)] + rel;
                if (base_->u_idx_[sz(rs)] != static_cast<Index>(j) || base_->u_rmirror_[sz(rs)] != t)
                    return fail("U twin links disagree");
            }
            for (Index t = 0; t < base_->u_clen_[j]; ++t) hit[sz(base_->u_crow_[beg + sz(t)])] = 0;
            clive += static_cast<Offset>(base_->u_clen_[j]);
        }
        if (clive != live) return fail("U mirrors hold different entry counts");
        if (u_live_nnz_ != live) return fail("u_live_nnz_ disagrees with the row store");
    }

    // Diagonal must be usable: a zero would make U singular.
    for (std::size_t k = 0; k < n; ++k)
        if (!std::isfinite(base_->piv_val_[k]) || base_->piv_val_[k] == 0.0)
            return fail("U has a zero or non-finite diagonal");

    // L: strictly lower triangular in its OWN (index) order, which no update
    // is allowed to disturb.
    if (base_->l_start_.size() != n + 1) return fail("base_->l_start_ size");
    for (std::size_t k = 0; k < n; ++k) {
        if (base_->l_start_[k] > base_->l_start_[k + 1]) return fail("base_->l_start_ not monotone");
        for (Offset t = base_->l_start_[k]; t < base_->l_start_[k + 1]; ++t) {
            const auto i = base_->l_idx_[sz(t)];
            if (i < 0 || sz(i) >= n) return fail("L row index out of range");
            if (sz(i) <= k) return fail("L is not strictly lower triangular");
        }
    }
    if (base_->l_start_.back() != static_cast<Offset>(base_->l_idx_.size()) ||
        base_->l_idx_.size() != base_->l_val_.size())
        return fail("L arrays inconsistent");

    // Row etas: in range, and never carrying their own pivot position (which
    // would make the elimination read the row it is eliminating).
    if (r_start_.size() != r_pos_.size() + 1) return fail("r_start_ size");
    if (r_start_.back() != static_cast<Offset>(r_idx_.size()) ||
        r_idx_.size() != r_val_.size())
        return fail("row-eta arrays inconsistent");
    for (std::size_t t = 0; t < r_pos_.size(); ++t) {
        if (r_start_[t] > r_start_[t + 1]) return fail("r_start_ not monotone");
        const auto p = r_pos_[t];
        if (p < 0 || sz(p) >= n) return fail("row-eta position out of range");
        for (Offset q = r_start_[t]; q < r_start_[t + 1]; ++q) {
            const auto j = r_idx_[sz(q)];
            if (j < 0 || sz(j) >= n) return fail("row-eta index out of range");
            if (j == p) return fail("row eta carries its own pivot position");
            if (!std::isfinite(r_val_[sz(q)])) return fail("row-eta value not finite");
        }
    }
    if (!r_pos_.empty() && !u_reordered_)
        return fail("row etas exist but the elimination order was never changed");
    return true;
}

Offset BasisFactor::eta_nnz() const {
    // Both update files count. A Forrest-Tomlin run has no product-form etas
    // at all, so without the row etas here every refactorization trigger in
    // needs_refactor() would key off a permanently empty structure and NONE of
    // them would ever fire -- see the note there.
    return static_cast<Offset>(eta_val_.size() + r_val_.size()) +
           (border_parent_ ? border_parent_->eta_nnz() : 0);
}

// A Forrest-Tomlin factorization MUST be refactorized on a schedule, and more
// often than a product-form one. FT does no pivoting in the update, so error
// compounds through the row-eta file: measured on a 200x200 basis with a
// 300-update chain and no refactorization at all, the residual grows
// 2e-13 -> 8e-12 -> 8e-10 -> 9e-8 -> 1e-6 -> 4e-4, roughly a decade per 45
// updates, where product form over the identical sequence stays at 1e-8.
// The decay is inherent to the method. What is NOT inherent, and was the
// defect here, is the cadence: the eta-nnz and update-count triggers were
// calibrated for product-form etas, and FT row etas are ~12x sparser, so those
// triggers fire about ten times less often on an FT run and the update_limit
// ceiling never binds at all. FT therefore ran hundreds of updates past the
// point where its accuracy was gone -- d2q06c interrupted at 171k pivots with
// a DSE log error of 0.29 and a cost shift of 6.3e+47. u_nnz_ratio and
// ft_update_limit below are the FT-specific cadence that fixes that; they are
// checked only once a row eta exists, so the product-form path is untouched.
f64 BasisFactor::hypersparse_hit_rate() const noexcept {
    const std::uint64_t hits = scratch_.hypersparse_solves_;
    const std::uint64_t total = scratch_.triangular_solves_;
    if (total == 0) return 1.0;
    return static_cast<f64>(hits) / static_cast<f64>(total);
}

bool BasisFactor::needs_refactor(int update_limit, f64 eta_nnz_ratio,
                                 Index bump_width_max, f64 work_ratio_max,
                                 f64 u_nnz_ratio, int ft_update_limit,
                                 f64 hypersparse_hit_floor) const {
    if (update_limit > 0 && static_cast<int>(n_updates()) >= update_limit)
        return true;
    if (eta_nnz_ratio > 0.0 && stats_.factor_nnz > 0) {
        const f64 limit = eta_nnz_ratio * static_cast<f64>(stats_.factor_nnz);
        if (static_cast<f64>(eta_nnz()) > limit) return true;
    }
    if (bump_width_max > 0 && bump_width_ > bump_width_max) return true;
    // Forrest-Tomlin cadence: U's growth, and a plain count of row etas. Both
    // apply only once update_ft() has actually run, so the product-form path
    // is untouched by them.
    if (!r_pos_.empty()) {
        if (u_nnz_ratio > 0.0 && stats_.factor_nnz > 0 &&
            static_cast<f64>(u_live_nnz_) >
                u_nnz_ratio * static_cast<f64>(stats_.factor_nnz))
            return true;
        // Hard floor: if the caller disarmed FT count, still refuse to fly
        // past a decade of compounding (kForrestTomlinHardUpdateFloor).
        const int armed_ft_limit =
            ft_update_limit > 0 ? ft_update_limit : kForrestTomlinHardUpdateFloor;
        if (static_cast<int>(r_pos_.size()) >= armed_ft_limit)
            return true;
    }
    if (work_ratio_max > 0.0 && stats_.factor_nnz > 0) {
        const f64 limit = work_ratio_max * static_cast<f64>(stats_.factor_nnz);
        if (static_cast<f64>(scratch_.work_since_factor_) > limit) return true;
    }
    // Hypersparse miss rate: once enough solves have been observed on a
    // non-trivial factor, a sustained drop below the floor means the update
    // file has densified the solves and a refactor is cheaper than more dense
    // triangular work. Small bases (factor_nnz < 4096) and short samples
    // (< 256 solves) are skipped: early dense-history noise otherwise forces
    // extra full factorizations on Netlib-scale models (grow7 regression).
    if (hypersparse_hit_floor > 0.0 && hypersparse_hit_floor < 1.0 &&
        stats_.factor_nnz >= 4096) {
        const std::uint64_t total = scratch_.triangular_solves_;
        if (total >= 256 && hypersparse_hit_rate() < hypersparse_hit_floor &&
            n_updates() > 0)
            return true;
    }
    return false;
}

namespace {

// Per-update instrumentation for the WS4 diagnosis, compiled out by default.
//
// Build with -DFORGE_LU_UPDATE_LOG_ENABLED and set FORGE_LU_UPDATE_LOG=<file> to
// get one line per basis update:
//
//   kind update spike_nnz bump_width delta_nnz u_nnz factor_nnz
//
// kind is P (product form: delta is the eta's nonzeros) or F (Forrest-Tomlin:
// delta is the ROW eta's nonzeros). This is what showed that FT's row eta is
// 10-12x SPARSER than the product-form eta and that its real cost is fill-in
// in U.
//
// It is compile-time rather than runtime because update() runs once per pivot,
// and on a machine quiet enough to measure it, even a predictable branch there
// is not worth defending. Off, this costs nothing at all.
#ifdef FORGE_LU_UPDATE_LOG_ENABLED
std::FILE* const g_lu_update_log = [] {
    const std::string& path = core::env_switches().lu_update_log;
    if (path.empty()) return static_cast<std::FILE*>(nullptr);
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f)
        std::fputs("# kind update spike_nnz bump_width delta_nnz u_nnz "
                   "factor_nnz max_atilde tiny_nnz\n", f);
    return f;
}();
inline std::FILE* lu_update_log() { return g_lu_update_log; }
#else
constexpr std::FILE* lu_update_log() { return nullptr; }
#endif

}  // namespace

bool BasisFactor::update(Index p, const std::vector<f64>& alpha, f64 min_pivot) {
    if (p < 0 || p >= m_ || alpha.size() != sz(m_) ||
        !std::all_of(alpha.begin(), alpha.end(), [](f64 v) { return std::isfinite(v); })) return false;
    const f64 ap = (sz(p) < alpha.size()) ? alpha[sz(p)] : 0.0;
    if (!(std::fabs(ap) > min_pivot)) return false;
    eta_p_.push_back(p);
    eta_pivot_.push_back(ap);
    // Reverse incidence: eta t reads slot p and every slice slot i.
    const auto t = static_cast<Index>(eta_p_.size() - 1);
    if (rev_.size() == sz(m_)) {
        rev_[sz(p)].push_back(t);
        for (Index i = 0; i < m_; ++i) {
            if (alpha[sz(i)] != 0.0 && i != p) {
                eta_idx_.push_back(i);
                eta_val_.push_back(alpha[sz(i)]);
                rev_[sz(i)].push_back(t);
            }
        }
    } else {
        // rev_ not built (stale or truncated factorize) - etas still work;
        // btran falls back to the plain full scan.
        for (Index i = 0; i < m_; ++i) {
            if (alpha[sz(i)] != 0.0 && i != p) {
                eta_idx_.push_back(i);
                eta_val_.push_back(alpha[sz(i)]);
            }
        }
    }
    eta_start_.push_back(static_cast<Offset>(eta_idx_.size()));
    if (std::FILE* fp = lu_update_log()) {
        const std::size_t t2 = eta_p_.size() - 1;
        const Offset delta = static_cast<Offset>(eta_idx_.size()) -
                             (t2 == 0 ? 0 : eta_start_[t2]);
        std::fprintf(fp, "P %zu %lld %d %lld %lld %lld\n", t2,
                     static_cast<long long>(delta), 0,
                     static_cast<long long>(delta),
                     static_cast<long long>(u_live_nnz_),
                     static_cast<long long>(stats_.factor_nnz));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Collective FT: fold every pending product-form eta into L/U via a
// sequence of update_ft() calls, then clear the eta file. See lu.hpp for
// why this reuses update_ft() unchanged instead of a new combined
// multi-column bump algorithm.
// ---------------------------------------------------------------------------

bool BasisFactor::collapse_pending_into_ft(const LuOptions& opts, f64 min_pivot) {
    // The border has no U bump representation. Its product-form journal
    // remains live until the normal full basis refactorization.
    if (border_parent_) return eta_p_.empty();
    detach_base();
    if (eta_p_.empty()) return true;

    // Snapshot every pending eta's ORIGINAL (p, alpha) as recorded by
    // update() -- alpha is already alpha[eta_p_[t]] = eta_pivot_[t] plus the
    // sparse eta_idx_/eta_val_ slice, i.e. exactly the vector update_ft()
    // needs, in the SAME slot-coordinate space update_ft() expects (each
    // alpha_t is B_{t-1}^-1 a_new_t relative to the basis as it stood right
    // before pivot t, which is exactly the state replaying etas 0..t-1 in
    // order reproduces).
    struct Pending { Index p; std::vector<f64> alpha; };
    std::vector<Pending> pending;
    pending.reserve(eta_p_.size());
    for (std::size_t t = 0; t < eta_p_.size(); ++t) {
        Pending pe;
        pe.p = eta_p_[t];
        pe.alpha.assign(sz(m_), 0.0);
        pe.alpha[sz(pe.p)] = eta_pivot_[t];
        for (Offset k = eta_start_[t]; k < eta_start_[t + 1]; ++k)
            pe.alpha[sz(eta_idx_[sz(k)])] = eta_val_[sz(k)];
        pending.push_back(std::move(pe));
    }

    // The live eta file is about to be superseded pivot-by-pivot below; clear
    // it up front so a failure partway through can re-insert exactly the
    // etas that did NOT get folded (see the catch-up loop), rather than
    // trying to surgically erase a prefix of the SoA arrays.
    eta_p_.clear(); eta_pivot_.clear();
    eta_idx_.clear(); eta_val_.clear();
    eta_start_.assign(1, 0);
    for (auto& r : rev_) r.clear();

    for (std::size_t t = 0; t < pending.size(); ++t) {
        if (update_ft(pending[t].p, pending[t].alpha, opts, min_pivot)) continue;
        // Bump for this eta turned out near-singular. Etas [0, t) are
        // already correctly folded into L/U (update_ft() mutates
        // atomically-on-success, never partially). Re-insert [t, end) via
        // the cheap product-form update() -- valid because the CURRENT
        // factorization (after folding [0, t)) is exactly the basis each
        // remaining alpha was computed relative to, same as it was live.
        // Each alpha[p] already passed this same min_pivot check once when
        // it was first recorded, so this re-insertion cannot itself fail.
        for (std::size_t u = t; u < pending.size(); ++u)
            update(pending[u].p, pending[u].alpha, min_pivot);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Forrest-Tomlin update
//
// Same contract as update(): p is a SLOT LABEL and alpha = B_old^-1 a_new is
// the ordinary FTRAN result in slot coordinates.
//
// Forrest & Tomlin (1972); the row-eta form and the free multipliers are
// Tomlin (1974); the "do not restore explicit triangularity" point is Huangfu
// & Hall, COAP 60:587-608 (2015) §2.1.
//
// THE ALGEBRA. Work in position coordinates, where position k means pivot row
// base_->piv_row_[k] and basis slot base_->piv_slot_[k]; both maps are fixed by factorize()
// and no update touches them. In those coordinates A = L U, and replacing the
// column at position ps gives
//
//     A' = L U',     U' = U + (atilde - U e_ps) e_ps^T,   atilde = L^-1 a_new.
//
// U' is not upper triangular for the current elimination order, but it becomes
// so if position ps is eliminated LAST instead of where it sits now: every
// entry of U's old row ps then lies to the left of the diagonal, and one row
// elimination clears them. Writing that elimination as M = I - e_ps v^T,
//
//     A' = L M^-1 Ubar,   Ubar = M U' upper triangular for the new order,
//
// so A'^-1 = Ubar^-1 M L^-1 and L is untouched. That is the whole point of the
// method: L and U are independent operation sequences over shared coordinates,
// so re-ordering one leaves the other alone.
//
// The three quantities needed are each O(nnz):
//
//   atilde = U * alpha           -- an algebraic identity, not a solve, since
//                                   alpha = U^-1 (L^-1 a_new) already.
//   v      = e_ps - u_pp * etilde_ps,  etilde_ps^T = e_ps^T U^-1   (Tomlin 1974)
//                                   -- one transposed triangular solve on a
//                                   unit vector, so hypersparse.
//   new diagonal = u_pp * alpha[p]
//                                   -- because v^T atilde telescopes:
//                                   atilde[ps] - v^T atilde
//                                     = u_pp * (e_ps^T U^-1 atilde)
//                                     = u_pp * alpha[ps].
//
// v[ps] = 1 - u_pp * etilde_ps[ps] = 0, so the eliminating row never uses
// itself and the stored eta excludes that position.
//
// COST is O(nnz(col ps) + nnz(row ps) + nnz(atilde) + nnz(v)) plus the order
// shift, independent of how many updates have accumulated. The version this
// replaced re-eliminated the whole trailing bump with pivoting and rebuilt L
// and U from scratch on every call, which measured 165 ms per update on a
// basis of 821 (25fv47: 127 366 ms over 770 updates) against product form's
// 5 us, i.e. unusable -- while showing, on dfl001, that FT keeps 100% of
// solves hypersparse where product form manages 15.5%.
//
// FAILURE leaves the factorization COMPLETELY UNTOUCHED and returns false, so
// the caller can refactorize. Everything that can fail is checked before the
// first mutation.
bool BasisFactor::update_ft(Index p, const std::vector<f64>& alpha,
                            const LuOptions& opts, f64 min_pivot,
                            const std::vector<Index>* alpha_support,
                            const SpikeCapture* spike,
                            const RowEtaCapture* row_eta) {
    if (border_parent_)
        throw std::logic_error("BasisFactor: FT requires an unbordered LU base");
    detach_base();
    (void)opts;
    if (m_ == 0 || sz(p) >= sz(m_) || alpha.size() < sz(m_)) return false;
    const auto n = base_->piv_val_.size();
    if (n != sz(m_) || base_->cpos_.size() < sz(m_)) return false;

    if (scratch_.factor_gen != factor_gen_) reset_scratch(scratch_);
    const Index ps = base_->cpos_[sz(p)];
    if (ps < 0 || sz(ps) >= n) return false;

    // alpha[p] is the pivot element in position coordinates; it is what the new
    // diagonal is built from, so reject a tiny one before touching anything.
    const f64 ap = alpha[sz(p)];
    if (!(std::fabs(ap) > min_pivot)) return false;
    const f64 u_pp = base_->piv_val_[sz(ps)];
    const f64 new_diag = u_pp * ap;
    if (!(std::fabs(new_diag) > min_pivot) || !std::isfinite(new_diag)) return false;

    // ---- atilde = U * alpha, column-driven ------------------------------
    // Column-driven visits, for each nonzero of alpha, only the rows that
    // actually have an entry in that column -- O(sum of touched column
    // lengths) rather than O(nnz(U)). The column store carries values
    // precisely so this needs no search.
    //
    // Both scratch vectors are cleared BY THEIR PREVIOUS SUPPORT rather than
    // refilled: an assign(n, 0.0) here is an O(m) pass per update, and there
    // are enough of those in an update to dominate it outright.
    if (ft_atilde_.size() != n) {
        ft_atilde_.assign(n, 0.0);
        ft_v_.assign(n, 0.0);
        ft_stamp_.assign(n, 0);
        ft_gen_ = 0;
    } else {
        for (const Index k : ft_support_) ft_atilde_[sz(k)] = 0.0;
        for (const Index k : ft_vsupport_) ft_v_[sz(k)] = 0.0;
    }
    ft_support_.clear();
    if (++ft_gen_ == 0) { std::fill(ft_stamp_.begin(), ft_stamp_.end(), 0); ft_gen_ = 1; }
    const auto touch = [&](Index k) {
        if (ft_stamp_[sz(k)] == ft_gen_) return;
        ft_stamp_[sz(k)] = ft_gen_;
        ft_support_.push_back(k);
    };
    // Iterate alpha's SUPPORT when the caller declared one. Scanning all m
    // slots is a random gather through base_->piv_slot_ and was, on a hypersparse
    // basis, most of the update's cost.
    const auto add_alpha_column = [&](std::size_t k, f64 xk) {
        touch(static_cast<Index>(k));
        ft_atilde_[k] += base_->piv_val_[k] * xk;             // the diagonal
        const auto cbeg = sz(base_->u_cstart_[k]);
        const auto cend = cbeg + sz(base_->u_clen_[k]);
        for (auto t = cbeg; t < cend; ++t) {
            touch(base_->u_crow_[t]);
            ft_atilde_[sz(base_->u_crow_[t])] += base_->u_cval_[t] * xk;
        }
    };
    if (spike != nullptr && spike->valid) {
        // The FTRAN that produced `alpha` already formed this exact vector on
        // its way through, between the row etas and the U solve. Use it. The
        // U * alpha recomputation below is the same thing in exact arithmetic
        // and NOT the same thing in floating point: it does not cancel to zero,
        // so it manufactures residue with alpha's density and every residue
        // entry becomes fill in U (~85% of installed entries, measured).
        for (std::size_t t = 0; t < spike->pos.size(); ++t) {
            const Index k = spike->pos[t];
            if (k < 0 || sz(k) >= n) continue;
            const f64 v = spike->val[t];
            if (v == 0.0) continue;
            touch(k);
            ft_atilde_[sz(k)] = v;
        }
    } else if (alpha_support != nullptr) {
        for (const Index slot : *alpha_support) {
            if (slot < 0 || sz(slot) >= n) continue;
            const f64 xk = alpha[sz(slot)];
            if (xk == 0.0) continue;
            const Index k = base_->cpos_[sz(slot)];
            if (k < 0 || sz(k) >= n) continue;
            if (ft_stamp_[sz(k)] == ft_gen_) continue;   // duplicate slot
            add_alpha_column(sz(k), xk);
        }
    } else {
        for (std::size_t k = 0; k < n; ++k) {
            const f64 xk = alpha[sz(base_->piv_slot_[k])];
            if (xk == 0.0) continue;
            add_alpha_column(k, xk);
        }
    }

    // ---- v = e_ps - u_pp * (e_ps^T U^-1) --------------------------------
    // Solve U^T y = e_ps. Hypersparse when it can be; the dense fallback is
    // still only O(nnz(U)) and this runs once per update, not per solve.
    ft_vsupport_.clear();
    // The BTRAN that priced this leaving row already formed e_ps^T U^-1 on
    // its way through (RowEtaCapture); a capture of this very U and slot
    // replaces the solve below with a copy of its nonzeros.
    const bool have_row_eta = row_eta != nullptr && row_eta->valid &&
        row_eta->pivot_pos == ps && row_eta->factor_gen == factor_gen_ &&
        row_eta->n_row_etas == static_cast<Index>(r_pos_.size()) &&
        row_eta->pos.size() == row_eta->val.size();
    if (have_row_eta) {
        for (std::size_t t = 0; t < row_eta->pos.size(); ++t) {
            const Index k = row_eta->pos[t];
            if (k < 0 || sz(k) >= n) continue;
            ft_v_[sz(k)] = row_eta->val[t];
            ft_vsupport_.push_back(k);
        }
    } else {
        ft_v_[sz(ps)] = 1.0;
        ft_seed_.assign(1, ps);
        if (sparse_upper_t(scratch_, ft_seed_, ft_v_)) {
            // Nonzeros only, in ascending position order: the canonical
            // order shared with the BTRAN capture.
            for (const Index k : scratch_.reach_) {
                scratch_.mark_[sz(k)] = 0;
                if (ft_v_[sz(k)] != 0.0) ft_vsupport_.push_back(k);
            }
            sort_support(ft_vsupport_, static_cast<Index>(n), scratch_.sort_tmp_, scratch_.sort_bits_);
        } else {
            solve_upper_t(ft_v_);
            for (std::size_t k = 0; k < n; ++k)
                if (ft_v_[k] != 0.0) ft_vsupport_.push_back(static_cast<Index>(k));
        }
    }
    for (const Index k : ft_vsupport_) ft_v_[sz(k)] *= -u_pp;
    // v[ps] is exactly zero in exact arithmetic (1 - u_pp/u_pp); make that
    // exact here so the stored eta can drop the position unconditionally and
    // the elimination provably never reads the row it is eliminating.
    ft_v_[sz(ps)] = 0.0;

    // Stability guard: an enormous row-eta multiplier amplifies every later
    // solve through this update; refuse it so the caller refactorizes. It
    // must run here, before any mutation: false promises an untouched
    // factor (collapse_pending_into_ft relies on that). Checking after U
    // was rewritten left a half-updated factor that was no longer
    // triangular in its elimination order.
    for (const Index k : ft_vsupport_)
        if (std::fabs(ft_v_[sz(k)]) > 1e8) return false;

    // Nothing below can fail, so the factorization is safe to mutate now.

    // ---- U: drop old column ps and old row ps ---------------------------
    // Column ps's entries are the old row-ps coefficients of other rows; the
    // entering column replaces all of them. Row ps's entries are the ones the
    // row eta is about to absorb.
    // Neither loop disturbs the segment it walks: a row meets column ps once,
    // so a drop inside row r moves an entry of another column, and likewise
    // for the columns row ps meets.
    {
        const Offset cbeg = base_->u_cstart_[sz(ps)];
        for (Index t = 0; t < base_->u_clen_[sz(ps)]; ++t) {
            const Index r = base_->u_crow_[sz(cbeg + t)];
            u_row_drop(r, base_->u_off_[sz(r)] + base_->u_cmirror_[sz(cbeg + t)]);
        }
        u_live_nnz_ -= static_cast<Offset>(base_->u_clen_[sz(ps)]);
        base_->u_clen_[sz(ps)] = 0;

        const Offset rbeg = base_->u_off_[sz(ps)];
        for (Index t = 0; t < base_->u_len_[sz(ps)]; ++t) {
            const Index j = base_->u_idx_[sz(rbeg + t)];
            u_col_drop(j, base_->u_cstart_[sz(j)] + base_->u_rmirror_[sz(rbeg + t)]);
        }
        u_live_nnz_ -= static_cast<Offset>(base_->u_len_[sz(ps)]);
        base_->u_len_[sz(ps)] = 0;
    }

    // ---- U: install the entering column at position ps ------------------
    // ps is about to become the last position in the elimination order, so
    // every other row may legally carry an entry in this column. Column ps is
    // empty now; give it room for the whole spike at once rather than
    // doubling it entry by entry.
    {
        Index spike_nnz = 0;
        for (const Index k : ft_support_) spike_nnz += k != ps && ft_atilde_[sz(k)] != 0.0;
        u_col_reserve(ps, spike_nnz);
    }
    for (const Index k : ft_support_) {
        if (k == ps) continue;
        const f64 val = ft_atilde_[sz(k)];
        if (val == 0.0) continue;
        u_append(k, ps, val);
    }
    base_->piv_val_[sz(ps)] = new_diag;

    // ---- the row eta ----------------------------------------------------
    for (const Index k : ft_vsupport_) {
        if (k == ps) continue;
        const f64 val = ft_v_[sz(k)];
        if (val == 0.0) continue;
        r_idx_.push_back(k);
        r_val_.push_back(val);
    }
    r_pos_.push_back(ps);
    r_start_.push_back(static_cast<Offset>(r_idx_.size()));

    // ---- eliminate ps last ----------------------------------------------
    // How far back the position had to travel. There is no bump to eliminate
    // any more -- that is the point of the rewrite -- but this is the same
    // quantity the old implementation's bump width measured (how much of the
    // trailing order this update disturbed), so the bump_width_max trigger
    // keeps its meaning.
    const Index shift = u_order_move_to_back(ps);
    bump_width_ = shift;
    if (std::FILE* fp = lu_update_log()) {
        const std::size_t t2 = r_pos_.size() - 1;
        const Offset row_eta_nnz =
            r_start_[t2 + 1] - (t2 == 0 ? 0 : r_start_[t2]);
        // spike nnz = nonzeros of atilde = U * alpha, which is what gets
        // installed into U. The question the extra columns answer is whether
        // those are REAL nonzeros or rounding residue: atilde is formed from
        // the full FTRAN result alpha, and in floating point U * alpha does not
        // cancel to exact zeros, so a residue entry survives the `!= 0.0` test
        // and becomes fill. `tiny` counts entries below 1e-13 of the largest.
        Offset spike_nnz = 0, tiny = 0;
        f64 max_atilde = 0.0;
        for (const Index k : ft_support_)
            max_atilde = std::max(max_atilde, std::fabs(ft_atilde_[sz(k)]));
        const f64 cut = 1e-13 * max_atilde;
        for (const Index k : ft_support_) {
            const f64 v = std::fabs(ft_atilde_[sz(k)]);
            if (v != 0.0) ++spike_nnz;
            if (v != 0.0 && v < cut) ++tiny;
        }
        std::fprintf(fp, "F %zu %lld %lld %lld %lld %lld %.6e %lld\n", t2,
                     static_cast<long long>(spike_nnz),
                     static_cast<long long>(shift),
                     static_cast<long long>(row_eta_nnz),
                     static_cast<long long>(u_live_nnz_),
                     static_cast<long long>(stats_.factor_nnz),
                     static_cast<double>(max_atilde),
                     static_cast<long long>(tiny));
    }

    // Storage housekeeping. Relocation leaves holes; once they outweigh the
    // live entries, rebuilding both mirrors from the row store is cheaper than
    // carrying them. This is bounded amortized work, not a per-update cost.
    // u_live_nnz_ is maintained incrementally (u_append and the two drops
    // above) -- an O(m) recount here would undo the point of the sparse edits.
    u_alloc_nnz_ = static_cast<Offset>(base_->u_idx_.size());
    if (u_dead_nnz_ > u_live_nnz_ + 64 || u_cdead_nnz_ > u_live_nnz_ + 64)
        compact_u_storage();
    return true;
}

// Rebuild both U stores packed, preserving contents exactly. Only the storage
// layout changes, so no solve can observe this beyond running faster.
void BasisFactor::compact_u_storage() { pack_u(); }

}  // namespace forge::la
