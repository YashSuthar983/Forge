// The option binding tables: setting, listing, and refusing bad input.
//
// The property worth protecting is that a mistyped or malformed option is
// REFUSED, never quietly ignored. A silently dropped option is the worst
// outcome here: a tuning run reports a number the caller attributes to a
// setting that never took effect.
#include "sor/core/options.hpp"
#include "sor/engines/qcqp.hpp"
#include "sor/engines/qp.hpp"
#include "sor/search/global_qp.hpp"
#include "sor/search/miqp_bb.hpp"
#include "test_helpers.hpp"

#include <string>

using namespace sor;

namespace {

void check_table(const std::vector<core::OptionBinding>& t, const char* what) {
    CHECK(!t.empty());
    for (const auto& b : t) {
        // every option is named, typed, documented, and can report its value
        CHECK(!b.name.empty());
        CHECK(b.type == "bool" || b.type == "int" || b.type == "real");
        CHECK(!b.help.empty());
        CHECK(static_cast<bool>(b.set));
        CHECK(static_cast<bool>(b.show));
        CHECK(!b.show().empty());
    }
    // names are unique, or one would shadow the other
    for (std::size_t i = 0; i < t.size(); ++i)
        for (std::size_t j = i + 1; j < t.size(); ++j)
            CHECK(t[i].name != t[j].name);
    std::string err;
    CHECK(!core::apply_option(t, "definitely_not_an_option=1", err));
    CHECK(!err.empty());
    CHECK(!core::apply_option(t, "missing_equals", err));
    std::printf("  %s: %zu options\n", what, t.size());
}

}  // namespace

int main() {
    engines::QpOptions qp;
    engines::QcqpLocalOptions qcqp;
    search::MiqpBbOptions miqp;
    search::GlobalQpOptions glob;

    check_table(engines::qp_option_bindings(qp), "qp");
    check_table(engines::qcqp_local_option_bindings(qcqp), "qcqp");
    check_table(search::miqp_bb_option_bindings(miqp), "miqp");
    check_table(search::global_qp_option_bindings(glob), "global");

    std::string err;

    // a set actually reaches the field
    CHECK(engines::set_qcqp_local_option(qcqp, "starts=7", err));
    CHECK(qcqp.starts == 7);
    CHECK(engines::set_qcqp_local_option(qcqp, "feas_tol=1e-9", err));
    CHECK(qcqp.feas_tol == 1e-9);

    // flags accept the documented spellings, and only those
    CHECK(engines::set_qp_option(qp, "polish=off", err));
    CHECK(!qp.polish);
    CHECK(engines::set_qp_option(qp, "polish=1", err));
    CHECK(qp.polish);
    CHECK(!engines::set_qp_option(qp, "polish=maybe", err));

    // a fractional value for an integer option is refused, not truncated: a
    // caller who wrote 2.5 meant something, and reading 2 would hide it
    CHECK(!search::set_miqp_bb_option(miqp, "max_nodes=2.5", err));
    CHECK(search::set_miqp_bb_option(miqp, "max_nodes=2", err));
    CHECK(miqp.max_nodes == 2u);

    // a negative count is out of range, not wrapped into something enormous
    CHECK(!search::set_miqp_bb_option(miqp, "max_nodes=-1", err));

    // the listing is non-empty and mentions an option that exists
    const std::string listing = core::format_options(search::global_qp_option_bindings(glob));
    CHECK(listing.find("psd_cuts") != std::string::npos);
    CHECK(listing.find("gap_tol") != std::string::npos);

    return sor::test::finish("test_options");
}
