// sor_gen - seeded industrial instance generators (PS demo cases).
//
//   sor_gen blend    --seed 42 --crudes 4 --products 3 -o blend.mps
//   sor_gen schedule --seed 7  --periods 6 --units 2 -o schedule.mps
//   sor_gen dispatch --seed 3  --gens 4 -o dispatch.qps
//   sor_gen all      --seed 42 --outdir examples/
#include "sor/io/write_mps.hpp"
#include "sor/model/lp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

using sor::core::Index;
using sor::core::f64;
using sor::model::LpProblem;
using sor::model::kInf;

template <class T>
std::size_t sz(T i) { return static_cast<std::size_t>(i); }

void usage() {
    std::fputs(
        "usage: sor_gen blend|schedule|dispatch|all [options]\n"
        "  --seed N         RNG seed (recorded in MANIFEST)\n"
        "  --outdir DIR     directory for 'all' (default examples/)\n"
        "  -o PATH          output file for a single generator\n"
        "  blend:    --crudes N --products M\n"
        "  schedule: --periods T --units U\n"
        "  dispatch: --gens G\n",
        stderr);
}

std::uint32_t parse_u32(const std::string& s, const char* what) {
    char* end = nullptr;
    const auto v = std::strtoul(s.c_str(), &end, 10);
    if (end == s.c_str()) {
        std::fprintf(stderr, "error: bad %s '%s'\n", what, s.c_str());
        std::exit(2);
    }
    return static_cast<std::uint32_t>(v);
}

// ---- crude blending LP ---------------------------------------------------
// Maximize margin: sum_p price_p * product_p - sum_c cost_c * crude_c
// Mass balance per product quality (sulfur): blend sulfur <= limit.
// Capacity on total crude.
LpProblem make_blend(std::uint32_t seed, int n_crudes, int n_products) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f64> cost_d(30.0, 80.0);
    std::uniform_real_distribution<f64> price_d(90.0, 160.0);
    std::uniform_real_distribution<f64> sulf_c(0.2, 3.5);
    std::uniform_real_distribution<f64> sulf_lim(0.5, 2.0);
    std::uniform_real_distribution<f64> dem_d(1000.0, 5000.0);

    const Index n = static_cast<Index>(n_crudes + n_products);
    LpProblem p;
    p.name = "BLEND_S" + std::to_string(seed);
    p.maximize = true;
    p.c.assign(static_cast<std::size_t>(n), 0.0);
    p.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    p.col_hi.assign(static_cast<std::size_t>(n), kInf);
    p.col_names.resize(static_cast<std::size_t>(n));
    p.is_integer.assign(static_cast<std::size_t>(n), false);

    std::vector<f64> crude_cost(n_crudes), crude_s(n_crudes);
    // Scale capacities with problem size so large instances stay feasible.
    const f64 crude_cap = 5000.0 * static_cast<f64>(std::max(n_crudes, 1));
    const f64 prod_cap = 4000.0 * static_cast<f64>(std::max(n_crudes, 1));
    const f64 cdu_cap = crude_cap;
    for (int c = 0; c < n_crudes; ++c) {
        crude_cost[c] = cost_d(rng);
        crude_s[c] = sulf_c(rng);
        p.c[static_cast<std::size_t>(c)] = -crude_cost[c];  // maximize → negative cost
        p.col_names[static_cast<std::size_t>(c)] = "CRUDE" + std::to_string(c + 1);
        p.col_hi[static_cast<std::size_t>(c)] = crude_cap;
    }
    for (int pr = 0; pr < n_products; ++pr) {
        const Index j = static_cast<Index>(n_crudes + pr);
        p.c[static_cast<std::size_t>(j)] = price_d(rng);
        p.col_names[static_cast<std::size_t>(j)] = "PROD" + std::to_string(pr + 1);
        p.col_hi[static_cast<std::size_t>(j)] = prod_cap;
    }

    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    std::vector<f64> row_lo, row_hi;
    std::vector<std::string> row_names;
    auto add_row = [&](const std::string& name, f64 lo, f64 hi) {
        row_names.push_back(name);
        row_lo.push_back(lo);
        row_hi.push_back(hi);
        return static_cast<Index>(row_names.size() - 1);
    };

    // Yield balance: sum_c y_{c,p} crude - prod_p >= 0
    std::uniform_real_distribution<f64> yield_d(0.15, 0.55);
    for (int pr = 0; pr < n_products; ++pr) {
        const Index r = add_row("YLD" + std::to_string(pr + 1), 0.0, kInf);
        for (int c = 0; c < n_crudes; ++c) {
            rows.push_back(r); cols.push_back(c); vals.push_back(yield_d(rng));
        }
        rows.push_back(r); cols.push_back(n_crudes + pr); vals.push_back(-1.0);
    }
    // CDU capacity
    {
        const Index r = add_row("CAP", -kInf, cdu_cap);
        for (int c = 0; c < n_crudes; ++c) {
            rows.push_back(r); cols.push_back(c); vals.push_back(1.0);
        }
    }
    // Demand (scaled so large product counts remain feasible under CDU cap)
    const f64 dem_scale = cdu_cap / (4.0 * static_cast<f64>(std::max(n_products, 1)));
    for (int pr = 0; pr < n_products; ++pr) {
        const f64 dem = std::min(dem_d(rng), dem_scale);
        const Index r = add_row("DEM" + std::to_string(pr + 1), dem, kInf);
        rows.push_back(r); cols.push_back(n_crudes + pr); vals.push_back(1.0);
    }
    // Sulfur: sum_c (s_c - limit) crude_c <= 0
    {
        const Index r = add_row("SULF", -kInf, 0.0);
        const f64 limit = sulf_lim(rng);
        for (int c = 0; c < n_crudes; ++c) {
            rows.push_back(r); cols.push_back(c); vals.push_back(crude_s[c] - limit);
        }
    }

    p.row_lo = std::move(row_lo);
    p.row_hi = std::move(row_hi);
    p.row_names = std::move(row_names);
    p.A = sor::sparse::from_triplets(static_cast<Index>(p.row_lo.size()), n,
                                     rows, cols, vals);
    p.validate();
    return p;
}

// ---- refinery scheduling MILP --------------------------------------------
// periods x units, binary on/off, min up, demand coverage, cost minimize.
LpProblem make_schedule(std::uint32_t seed, int periods, int units) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f64> cost_d(10.0, 40.0);
    std::uniform_real_distribution<f64> dem_d(1.0, static_cast<f64>(units) * 0.7);

    // vars: x[t,u] binary run; s[t,u] binary startup
    const Index T = periods, U = units;
    const Index n_x = T * U;
    const Index n = n_x + n_x;  // run + startup
    LpProblem p;
    p.name = "SCHED_S" + std::to_string(seed);
    p.maximize = false;
    p.c.assign(static_cast<std::size_t>(n), 0.0);
    p.col_lo.assign(static_cast<std::size_t>(n), 0.0);
    p.col_hi.assign(static_cast<std::size_t>(n), 1.0);
    p.is_integer.assign(static_cast<std::size_t>(n), true);
    p.col_names.resize(static_cast<std::size_t>(n));

    auto xid = [&](int t, int u) { return static_cast<Index>(t * U + u); };
    auto sid = [&](int t, int u) { return static_cast<Index>(n_x + t * U + u); };

    for (int t = 0; t < T; ++t)
        for (int u = 0; u < U; ++u) {
            p.col_names[sz(xid(t, u))] =
                "X_T" + std::to_string(t + 1) + "U" + std::to_string(u + 1);
            p.col_names[sz(sid(t, u))] =
                "S_T" + std::to_string(t + 1) + "U" + std::to_string(u + 1);
            p.c[sz(xid(t, u))] = cost_d(rng);           // running cost
            p.c[sz(sid(t, u))] = 5.0 * cost_d(rng);     // startup cost
        }

    std::vector<Index> rows, cols;
    std::vector<f64> vals;
    std::vector<f64> row_lo, row_hi;
    std::vector<std::string> row_names;

    auto add_row = [&](const std::string& name, f64 lo, f64 hi) {
        row_names.push_back(name);
        row_lo.push_back(lo);
        row_hi.push_back(hi);
        return static_cast<Index>(row_names.size() - 1);
    };

    // Demand: sum_u x[t,u] >= dem[t]
    for (int t = 0; t < T; ++t) {
        const Index r = add_row("DEM_T" + std::to_string(t + 1), dem_d(rng), kInf);
        for (int u = 0; u < U; ++u) {
            rows.push_back(r);
            cols.push_back(xid(t, u));
            vals.push_back(1.0);
        }
    }
    // Startup linking: s[t,u] >= x[t,u] - x[t-1,u]
    for (int t = 0; t < T; ++t)
        for (int u = 0; u < U; ++u) {
            const Index r = add_row("SU_T" + std::to_string(t + 1) + "U" +
                                        std::to_string(u + 1),
                                    -kInf, 0.0);
            rows.push_back(r); cols.push_back(sid(t, u)); vals.push_back(-1.0);
            rows.push_back(r); cols.push_back(xid(t, u)); vals.push_back(1.0);
            if (t > 0) {
                rows.push_back(r);
                cols.push_back(xid(t - 1, u));
                vals.push_back(-1.0);
            }
        }

    p.row_lo = std::move(row_lo);
    p.row_hi = std::move(row_hi);
    p.row_names = std::move(row_names);
    p.A = sor::sparse::from_triplets(static_cast<Index>(p.row_lo.size()), n,
                                     rows, cols, vals);
    p.validate();
    return p;
}

// ---- power dispatch QP ---------------------------------------------------
void make_dispatch(std::uint32_t seed, int n_gens,
                   LpProblem& lp, std::vector<f64>& q_diag) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<f64> a_d(0.05, 0.4);   // quadratic
    std::uniform_real_distribution<f64> b_d(2.0, 20.0);   // linear
    std::uniform_real_distribution<f64> pmax_d(50.0, 200.0);

    const Index n = n_gens;
    lp = LpProblem{};
    lp.name = "DISPATCH_S" + std::to_string(seed);
    lp.maximize = false;
    lp.c.assign(sz(n), 0.0);
    q_diag.assign(sz(n), 0.0);
    lp.col_lo.assign(sz(n), 0.0);
    lp.col_hi.assign(sz(n), 0.0);
    lp.col_names.resize(sz(n));
    lp.is_integer.assign(sz(n), false);

    f64 demand = 0.0;
    for (Index j = 0; j < n; ++j) {
        const f64 pmax = pmax_d(rng);
        lp.col_hi[sz(j)] = pmax;
        lp.col_lo[sz(j)] = 0.1 * pmax;
        lp.c[sz(j)] = b_d(rng);
        q_diag[sz(j)] = a_d(rng);  // Q_jj for 1/2 x'Qx
        lp.col_names[sz(j)] = "G" + std::to_string(j + 1);
        demand += 0.55 * pmax;
    }
    // Equality: sum p = demand
    lp.row_lo = {demand};
    lp.row_hi = {demand};
    lp.row_names = {"DEMAND"};
    std::vector<Index> rows(sz(n), 0), cols(sz(n));
    std::vector<f64> vals(sz(n), 1.0);
    for (Index j = 0; j < n; ++j) cols[sz(j)] = j;
    lp.A = sor::sparse::from_triplets(1, n, rows, cols, vals);
    lp.validate();
}

void write_manifest_line(std::ostream& man, const std::string& kind,
                         std::uint32_t seed, const std::string& path,
                         const std::string& extra, bool last = false) {
    man << "  {\"kind\": \"" << kind << "\", \"seed\": " << seed
        << ", \"path\": \"" << path << "\"" << extra << "}"
        << (last ? "\n" : ",\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    const std::string cmd = argv[1];
    std::uint32_t seed = 42;
    std::string out_path, outdir = "examples";
    int n_crudes = 4, n_products = 3, periods = 6, units = 2, n_gens = 4;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* w) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s needs a value\n", w);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--seed") seed = parse_u32(next("--seed"), "seed");
        else if (a == "-o") out_path = next("-o");
        else if (a == "--outdir") outdir = next("--outdir");
        else if (a == "--crudes") n_crudes = static_cast<int>(parse_u32(next("--crudes"), "crudes"));
        else if (a == "--products") n_products = static_cast<int>(parse_u32(next("--products"), "products"));
        else if (a == "--periods") periods = static_cast<int>(parse_u32(next("--periods"), "periods"));
        else if (a == "--units") units = static_cast<int>(parse_u32(next("--units"), "units"));
        else if (a == "--gens") n_gens = static_cast<int>(parse_u32(next("--gens"), "gens"));
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else {
            std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            return 2;
        }
    }

    auto ensure_dir = [](const std::string& d) {
        std::filesystem::create_directories(d);
    };

    try {
        if (cmd == "blend") {
            if (out_path.empty()) out_path = "blend_s" + std::to_string(seed) + ".mps";
            auto p = make_blend(seed, n_crudes, n_products);
            sor::io::write_mps_file(out_path, p);
            std::printf("wrote %s  (%d x %d)\n", out_path.c_str(),
                        p.n_rows(), p.n_cols());
            return 0;
        }
        if (cmd == "schedule") {
            if (out_path.empty()) out_path = "schedule_s" + std::to_string(seed) + ".mps";
            auto p = make_schedule(seed, periods, units);
            sor::io::write_mps_file(out_path, p);
            std::printf("wrote %s  (%d x %d, %zu integer)\n", out_path.c_str(),
                        p.n_rows(), p.n_cols(), p.n_integer());
            return 0;
        }
        if (cmd == "dispatch") {
            if (out_path.empty()) out_path = "dispatch_s" + std::to_string(seed) + ".qps";
            LpProblem lp;
            std::vector<f64> q;
            make_dispatch(seed, n_gens, lp, q);
            sor::io::write_qps_file(out_path, lp, q);
            std::printf("wrote %s  (%d gens)\n", out_path.c_str(), n_gens);
            return 0;
        }
        if (cmd == "all") {
            ensure_dir(outdir);
            ensure_dir(outdir + "/crude_blending");
            ensure_dir(outdir + "/scheduling");
            ensure_dir(outdir + "/dispatch");
            const std::string man_path = outdir + "/MANIFEST.json";
            std::ofstream man(man_path);
            man << "{\n  \"generator\": \"sor_gen\",\n  \"seed\": " << seed
                << ",\n  \"instances\": [\n";

            {
                auto p = make_blend(seed, n_crudes, n_products);
                const std::string path =
                    outdir + "/crude_blending/blend_s" + std::to_string(seed) + ".mps";
                sor::io::write_mps_file(path, p);
                write_manifest_line(man, "blend_lp", seed, path,
                                    ", \"crudes\": " + std::to_string(n_crudes) +
                                        ", \"products\": " + std::to_string(n_products));
                std::printf("wrote %s\n", path.c_str());
            }
            {
                auto p = make_schedule(seed, periods, units);
                const std::string path =
                    outdir + "/scheduling/schedule_s" + std::to_string(seed) + ".mps";
                sor::io::write_mps_file(path, p);
                write_manifest_line(man, "schedule_milp", seed, path,
                                    ", \"periods\": " + std::to_string(periods) +
                                        ", \"units\": " + std::to_string(units));
                std::printf("wrote %s\n", path.c_str());
            }
            {
                LpProblem lp;
                std::vector<f64> q;
                make_dispatch(seed, n_gens, lp, q);
                const std::string path =
                    outdir + "/dispatch/dispatch_s" + std::to_string(seed) + ".qps";
                sor::io::write_qps_file(path, lp, q);
                write_manifest_line(man, "dispatch_qp", seed, path,
                                    ", \"gens\": " + std::to_string(n_gens),
                                    /*last=*/true);
                std::printf("wrote %s\n", path.c_str());
            }
            man << "  ]\n}\n";
            std::printf("wrote %s\n", man_path.c_str());
            return 0;
        }
        std::fprintf(stderr, "error: unknown command '%s'\n", cmd.c_str());
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
