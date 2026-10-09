#pragma once
#include "sor/core/result.hpp"
#include "sor/core/work_clock.hpp"
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sor::search {
// Immutable node domains with page sharing. A selected worker materializes
// ordinary arrays and exclusively owns their writes. Capture compares those
// arrays with the prior checkpoint, sharing every unchanged page. No borrowed
// mutable array or reference is ever made part of another node's checkpoint.
class DomainCheckpoint {
public:
  static constexpr std::size_t page_size = 256;
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  core::f64 lower(std::size_t j) const {
    if (j >= size_)
      throw std::out_of_range("domain checkpoint column");
    return pages_->at(j / page_size)->bounds[j % page_size].lo;
  }
  core::f64 upper(std::size_t j) const {
    if (j >= size_)
      throw std::out_of_range("domain checkpoint column");
    return pages_->at(j / page_size)->bounds[j % page_size].hi;
  }
  static DomainCheckpoint capture(const std::vector<core::f64> &lo,
                                  const std::vector<core::f64> &hi,
                                  const DomainCheckpoint &prior) {
    if (lo.size() != hi.size())
      throw std::logic_error("domain checkpoint shape");
    core::work_add(lo.size() * 2, core::WorkTag::BabNode);
    DomainCheckpoint result;
    result.size_ = lo.size();
    auto table = std::make_shared<Table>();
    table->reserve((lo.size() + page_size - 1) / page_size);
    for (std::size_t first = 0; first < lo.size(); first += page_size) {
      const auto end = std::min(lo.size(), first + page_size);
      const auto q = first / page_size;
      bool same = prior.size_ == lo.size();
      for (std::size_t j = first; j < end && same; ++j)
        same = lo[j] == prior.lower(j) && hi[j] == prior.upper(j);
      if (same)
        table->push_back((*prior.pages_)[q]);
      else {
        auto page = std::make_shared<Page>();
        for (std::size_t j = first; j < end; ++j)
          page->bounds[j - first] = {lo[j], hi[j]};
        table->push_back(std::move(page));
      }
    }
    result.pages_ = std::move(table);
    return result;
  }
  void materialize(std::vector<core::f64> &lo,
                   std::vector<core::f64> &hi) const {
    core::work_add(size_ * 2, core::WorkTag::BabNode);
    lo.resize(size_);
    hi.resize(size_);
    for (std::size_t q = 0; q < pages_->size(); ++q) {
      const auto first = q * page_size,
                 end = std::min(size_, first + page_size);
      const auto &page = *(*pages_)[q];
      for (std::size_t j = first; j < end; ++j) {
        lo[j] = page.bounds[j - first].lo;
        hi[j] = page.bounds[j - first].hi;
      }
    }
  }
  // Edits detach the page table and the edited page. All other checkpoints
  // remain immutable, including siblings read simultaneously by workers.
  void set_lower(std::size_t j, core::f64 value) { edit(j, value, false); }
  void set_upper(std::size_t j, core::f64 value) { edit(j, value, true); }
  struct Edit {
    std::size_t column;
    core::f64 value;
    bool upper;
  };
  void apply(const std::vector<Edit> &edits) {
    if (edits.empty())
      return;
    core::work_add(pages_->size() + edits.size(), core::WorkTag::BabNode);
    auto table = std::make_shared<Table>(*pages_);
    std::vector<std::shared_ptr<Page>> changed(table->size());
    for (const auto &e : edits) {
      if (e.column >= size_)
        throw std::out_of_range("domain checkpoint column");
      const auto q = e.column / page_size, offset = e.column % page_size;
      const auto &old = (*table)[q]->bounds[offset];
      if ((e.upper ? old.hi : old.lo) == e.value)
        continue;
      if (!changed[q]) {
        core::work_add(page_size * 2, core::WorkTag::BabNode);
        changed[q] = std::make_shared<Page>(*(*table)[q]);
        (*table)[q] = changed[q];
      }
      auto &bound = changed[q]->bounds[offset];
      (e.upper ? bound.hi : bound.lo) = e.value;
    }
    pages_ = std::move(table);
  }
  bool shares_page(const DomainCheckpoint &other, std::size_t j) const {
    return j < size_ && j < other.size_ &&
           (*pages_)[j / page_size] == (*other.pages_)[j / page_size];
  }
  template<class Fn> void for_each_changed_column(const DomainCheckpoint& prior, Fn&& visit) const {
    for (std::size_t q = 0; q < pages_->size(); ++q) {
      if (size_ == prior.size_ && (*pages_)[q] == (*prior.pages_)[q]) continue;
      const auto end = std::min(size_, (q+1)*page_size);
      for (std::size_t j = q*page_size; j < end; ++j) visit(j);
    }
  }

private:
  struct Bounds {
    core::f64 lo = 0, hi = 0;
  };
  struct Page {
    std::array<Bounds, page_size> bounds;
  };
  using Table = std::vector<std::shared_ptr<const Page>>;
  void edit(std::size_t j, core::f64 value, bool upper_side) {
    if (j >= size_)
      throw std::out_of_range("domain checkpoint column");
    if (upper_side ? upper(j) == value : lower(j) == value)
      return;
    core::work_add(pages_->size() + page_size * 2, core::WorkTag::BabNode);
    auto table = std::make_shared<Table>(*pages_);
    auto page = std::make_shared<Page>(*(*table)[j / page_size]);
    auto &bound = page->bounds[j % page_size];
    (upper_side ? bound.hi : bound.lo) = value;
    (*table)[j / page_size] = std::move(page);
    pages_ = std::move(table);
  }
  std::size_t size_ = 0;
  std::shared_ptr<const Table> pages_ = std::make_shared<Table>();
};
} // namespace sor::search
