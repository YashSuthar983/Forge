// sor_milp_train — bulk Sparse-SB / SC-MILP / PlanB&B label collection + fit.
//
// Runs strong-branch probes under milp.policy=latest with collect_labels on a
// list of MPS/LP paths, aggregates samples, fits via fit_sparse_sb_lasso /
// fit_sc_milp_contrastive / fit_planbb_paper, and writes models loadable by
// --sparse-sb-model / --sc-milp-model / --planbb-model.
#include "sor/io/mps.hpp"
#include "sor/search/bab.hpp"
#include "sor/search/dynsep.hpp"
#include "sor/search/lifted_branch.hpp"
#include "sor/search/planbb.hpp"
#include "sor/search/sc_milp_branch.hpp"
#include "sor/search/sparse_sb.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage() {
    std::fputs(
        "usage: sor_milp_train [options] MODEL.mps [MODEL2.mps ...]\n"
        "       sor_milp_train [options] --list PATHS.txt\n"
        "\n"
        "Collect strong-branch / separator labels across instances; fit Sparse-SB,\n"
        "SC-MILP, Lifted, PlanB&B, and/or DynSep GNN models for sor_solve.\n"
        "\n"
        "  --list PATH          text file: one MPS/LP path per line (# comments ok)\n"
        "  --out-dir DIR        directory for model files (default .)\n"
        "  --sparse-sb-out PATH write Sparse-SB model (default DIR/sparse_sb.model)\n"
        "  --sc-milp-out PATH   write SC-MILP model (default DIR/sc_milp.model)\n"
        "  --lifted-out PATH    write Lifted expert (default DIR/lifted_sb.model)\n"
        "  --planbb-out PATH    write PlanB&B paper model (default DIR/planbb.model)\n"
        "  --dynsep-out PATH    write DynSep GNN (default DIR/dynsep.model)\n"
        "  --only sparse-sb|sc-milp|lifted|planbb|dynsep|both|all\n"
        "  --sb-loss ranking|lasso   Sparse-SB fit loss (default ranking)\n"
        "  --time-limit S       per-instance wall limit (default 5)\n"
        "  --max-nodes N        per-instance node cap (default 500)\n"
        "  --max-samples N      global sample cap (default 50000)\n"
        "  --strong-cands K     strong-branch candidates per node (default 8)\n"
        "  --basis-update NAME  product | ft  (node LP; default product)\n"
        "  --verbose\n",
        stderr);
}

[[noreturn]] void die(const char* msg) {
    std::fprintf(stderr, "error: %s\n", msg);
    std::exit(2);
}

double parse_real(const std::string& s, const char* what) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0' || !(v == v) || v < 0.0)
        die((std::string(what) + " expects a non-negative real").c_str());
    return v;
}

std::uint64_t parse_u64(const std::string& s, const char* what) {
    char* end = nullptr;
    const auto v = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0')
        die((std::string(what) + " expects a non-negative integer").c_str());
    return static_cast<std::uint64_t>(v);
}

std::vector<std::string> read_list(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open --list file");
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t' ||
                                 line.back() == '\r'))
            line.pop_back();
        std::size_t i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i < line.size()) out.push_back(line.substr(i));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> models;
    std::string list_path;
    std::string out_dir = ".";
    std::string sparse_out;
    std::string sc_out;
    std::string lifted_out;
    std::string planbb_out;
    std::string dynsep_out;
    std::string only = "both";
    std::string sb_loss = "ranking";
    double time_limit = 5.0;
    std::uint64_t max_nodes = 500;
    std::uint64_t max_samples = 50000;
    int strong_cands = 8;
    bool verbose = false;
    sor::la::UpdateMethod update = sor::la::UpdateMethod::ProductForm;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) die((std::string(flag) + " needs a value").c_str());
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a == "--list") {
            list_path = next("--list");
        } else if (a == "--out-dir") {
            out_dir = next("--out-dir");
        } else if (a == "--sparse-sb-out") {
            sparse_out = next("--sparse-sb-out");
        } else if (a == "--sc-milp-out") {
            sc_out = next("--sc-milp-out");
        } else if (a == "--lifted-out") {
            lifted_out = next("--lifted-out");
        } else if (a == "--planbb-out") {
            planbb_out = next("--planbb-out");
        } else if (a == "--dynsep-out") {
            dynsep_out = next("--dynsep-out");
        } else if (a == "--only") {
            only = next("--only");
        } else if (a == "--sb-loss") {
            sb_loss = next("--sb-loss");
        } else if (a == "--time-limit") {
            time_limit = parse_real(next("--time-limit"), "--time-limit");
        } else if (a == "--max-nodes") {
            max_nodes = parse_u64(next("--max-nodes"), "--max-nodes");
        } else if (a == "--max-samples") {
            max_samples = parse_u64(next("--max-samples"), "--max-samples");
        } else if (a == "--strong-cands") {
            strong_cands = static_cast<int>(
                parse_u64(next("--strong-cands"), "--strong-cands"));
        } else if (a == "--basis-update") {
            const std::string m = next("--basis-update");
            if (m == "product")
                update = sor::la::UpdateMethod::ProductForm;
            else if (m == "ft")
                update = sor::la::UpdateMethod::ForrestTomlin;
            else
                die("--basis-update wants product|ft");
        } else if (a == "--verbose") {
            verbose = true;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "error: unknown flag '%s'\n", a.c_str());
            usage();
            return 2;
        } else {
            models.push_back(a);
        }
    }

    if (!list_path.empty()) {
        auto more = read_list(list_path);
        models.insert(models.end(), more.begin(), more.end());
    }
    if (models.empty()) {
        usage();
        return 2;
    }

    const bool want_sb = (only == "both" || only == "all" || only == "sparse-sb" ||
                          only == "sparsesb" || only == "sb");
    const bool want_sc = (only == "both" || only == "all" || only == "sc-milp" ||
                          only == "scmilp" || only == "sc");
    const bool want_lifted =
        (only == "all" || only == "lifted" || only == "lifted-branch" ||
         only == "lifted_branch");
    const bool want_planbb =
        (only == "all" || only == "planbb" || only == "plan-bb" ||
         only == "plan_bb");
    const bool want_dynsep =
        (only == "all" || only == "dynsep" || only == "dyn-sep");
    if (!want_sb && !want_sc && !want_lifted && !want_planbb && !want_dynsep) {
        die("--only wants sparse-sb|sc-milp|lifted|planbb|dynsep|both|all");
    }
    sor::search::SparseSbLoss sparse_loss =
        sor::search::SparseSbLoss::Ranking;
    if (sb_loss == "lasso")
        sparse_loss = sor::search::SparseSbLoss::Lasso;
    else if (sb_loss != "ranking" && sb_loss != "rank")
        die("--sb-loss wants ranking|lasso");

    if (sparse_out.empty()) sparse_out = out_dir + "/sparse_sb.model";
    if (sc_out.empty()) sc_out = out_dir + "/sc_milp.model";
    if (lifted_out.empty()) lifted_out = out_dir + "/lifted_sb.model";
    if (planbb_out.empty()) planbb_out = out_dir + "/planbb.model";
    if (dynsep_out.empty()) dynsep_out = out_dir + "/dynsep.model";

    sor::search::SparseSbCollector sb_pool;
    sb_pool.max_samples = max_samples;
    sor::search::ScMilpCollector sc_pool;
    sc_pool.max_samples = max_samples;
    sor::search::LiftedSbCollector lifted_pool;
    lifted_pool.max_samples = max_samples;
    sor::search::PlanBbCollector planbb_pool;
    planbb_pool.max_samples = max_samples;
    sor::search::DynSepCollector dynsep_pool;
    dynsep_pool.max_samples = max_samples;

    std::size_t ok_instances = 0;
    for (const auto& path : models) {
        if (sb_pool.samples.size() >= max_samples &&
            sc_pool.samples.size() >= max_samples &&
            lifted_pool.samples.size() >= max_samples &&
            planbb_pool.samples.size() >= max_samples &&
            dynsep_pool.samples.size() >= max_samples)
            break;

        sor::io::MpsReadReport rep;
        sor::model::LpProblem problem;
        try {
            problem = sor::io::read_mps_file_auto(path, rep);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "skip %s: %s\n", path.c_str(), e.what());
            continue;
        }
        if (rep.n_integer == 0) {
            std::fprintf(stderr, "skip %s: no integer columns\n", path.c_str());
            continue;
        }

        sor::search::BabOptions bab;
        bab.policy = sor::search::MilpPolicy::Latest;
        bab.time_limit_s = time_limit;
        bab.max_nodes = max_nodes;
        bab.reliability_branching = true;
        bab.strong_branch_candidates = strong_cands;
        bab.lp.update_method = update;
        bab.feasibility_jump = false;
        bab.sub_mip_lns = false;
        bab.balans.enabled = false;
        bab.kernel_pump.enabled = false;
        bab.mrens.enabled = false;
        bab.cuts_enabled = want_dynsep;
        bab.tree_cut.enabled = want_dynsep;
        bab.probing = false;
        bab.mip_presolve = false;
        bab.symmetry = false;
        bab.dynsep.enabled = want_dynsep;
        bab.dynsep.collect_labels = want_dynsep;
        bab.dynsep.backend = sor::search::DynSepBackend::Ucb;  // expert labels
        bab.dynsep_collect_out = want_dynsep ? &dynsep_pool : nullptr;
        bab.conflict_cut.enabled = false;

        bab.sparse_sb.enabled = true;
        bab.sparse_sb.collect_labels = want_sb;
        bab.sparse_sb.collect_max_samples = max_samples;
        bab.sparse_sb_collect_out = want_sb ? &sb_pool : nullptr;

        bab.sc_milp.enabled = true;
        bab.sc_milp.collect_labels = want_sc;
        bab.sc_milp.collect_max_samples = max_samples;
        bab.sc_milp_collect_out = want_sc ? &sc_pool : nullptr;

        bab.lifted.enabled = want_lifted;
        bab.lifted_collect_out = want_lifted ? &lifted_pool : nullptr;

        bab.planbb.enabled = true;
        bab.planbb.collect_labels = want_planbb;
        bab.planbb.collect_max_samples = max_samples;
        bab.planbb_collect_out = want_planbb ? &planbb_pool : nullptr;
        if (want_planbb)
            bab.branch_strategy = sor::search::BranchStrategy::PlanBb;
        else if (want_lifted && !want_sb && !want_sc)
            bab.branch_strategy = sor::search::BranchStrategy::Lifted;
        else
            bab.branch_strategy = sor::search::BranchStrategy::Auto;

        bab.sparse_sb.model_path.clear();
        bab.sc_milp.model_path.clear();
        bab.planbb.model_path.clear();
        bab.planbb.policy_path.clear();

        sor::search::BabDiagnostics diag;
        if (verbose)
            std::fprintf(stderr, "collect %s ...\n", path.c_str());
        (void)sor::search::solve_milp(problem, bab, diag);
        ++ok_instances;
        if (verbose) {
            std::fprintf(stderr,
                         "  %s: nodes=%llu sb+=%llu sc+=%llu planbb+=%llu "
                         "dynsep+=%llu\n",
                         path.c_str(),
                         static_cast<unsigned long long>(diag.nodes),
                         static_cast<unsigned long long>(diag.sparse_sb_samples),
                         static_cast<unsigned long long>(diag.sc_milp_samples),
                         static_cast<unsigned long long>(diag.planbb_samples),
                         static_cast<unsigned long long>(diag.dynsep_samples));
        }
    }

    std::printf("instances_ok:      %zu / %zu\n", ok_instances, models.size());
    std::printf("sparse_sb_samples: %zu\n", sb_pool.samples.size());
    std::printf("sc_milp_samples:   %zu\n", sc_pool.samples.size());
    std::printf("lifted_samples:    %zu\n", lifted_pool.samples.size());
    std::printf("planbb_samples:    %zu\n", planbb_pool.samples.size());
    std::printf("dynsep_samples:    %zu\n", dynsep_pool.samples.size());

    int rc = 0;
    if (want_sb) {
        if (sb_pool.samples.size() < 4) {
            std::fprintf(stderr,
                         "warn: too few Sparse-SB samples (%zu); skip fit\n",
                         sb_pool.samples.size());
            rc = 1;
        } else {
            sor::search::SparseSbFitOptions fit;
            fit.use_quadratic = true;
            fit.lasso_lambda = 1e-3;
            fit.max_iter = 200;
            fit.max_nonzero = 500;
            fit.loss = sparse_loss;
            fit.ranking_epochs = 40;
            auto model = sor::search::fit_sparse_sb(sb_pool.samples, fit);
            if (!model.loaded ||
                !sor::search::save_sparse_sb_model(sparse_out, model)) {
                std::fprintf(stderr, "error: Sparse-SB fit/save failed\n");
                rc = 1;
            } else {
                std::printf("sparse_sb_model:   %s  (terms=%zu loss=%s)\n",
                            sparse_out.c_str(), model.terms.size(),
                            model.loss == sor::search::SparseSbLoss::Ranking
                                ? "ranking"
                                : "lasso");
            }
        }
    }

    if (want_sc) {
        if (sc_pool.samples.size() < 2) {
            std::fprintf(stderr,
                         "warn: too few SC-MILP samples (%zu); skip fit\n",
                         sc_pool.samples.size());
            rc = 1;
        } else {
            sor::search::ScMilpFitOptions fit;
            fit.lr = 0.05;
            fit.epochs = 40;
            fit.contrastive_weight = 0.5;
            auto model =
                sor::search::fit_sc_milp_contrastive(sc_pool.samples, fit);
            if (!model.loaded ||
                !sor::search::save_sc_milp_model(sc_out, model)) {
                std::fprintf(stderr, "error: SC-MILP fit/save failed\n");
                rc = 1;
            } else {
                std::printf("sc_milp_model:     %s\n", sc_out.c_str());
            }
        }
    }

    if (want_lifted) {
        if (lifted_pool.samples.size() < 4) {
            std::fprintf(stderr,
                         "warn: too few Lifted samples (%zu); skip fit\n",
                         lifted_pool.samples.size());
            rc = 1;
        } else {
            sor::search::LiftedBranchOptions lopts;
            lopts.use_quadratic = false;
            lopts.loss = sor::search::SparseSbLoss::Ranking;
            lopts.ranking_epochs = 40;
            lopts.max_nonzero = 200;
            auto model = sor::search::fit_lifted_sb(lifted_pool.samples, lopts);
            if (!model.loaded ||
                !sor::search::save_lifted_sb_model(lifted_out, model)) {
                std::fprintf(stderr, "error: Lifted fit/save failed\n");
                rc = 1;
            } else {
                std::printf("lifted_model:      %s  (terms=%zu)\n",
                            lifted_out.c_str(), model.terms.size());
            }
        }
    }

    if (want_planbb) {
        if (planbb_pool.samples.size() < 4) {
            std::fprintf(stderr,
                         "warn: too few PlanB&B samples (%zu); skip fit\n",
                         planbb_pool.samples.size());
            rc = 1;
        } else {
            sor::search::PlanBbPaperFitOptions fit;
            fit.lr = 0.02;
            fit.epochs = 40;
            auto model =
                sor::search::fit_planbb_paper(planbb_pool.samples, fit);
            if (!model.ready() ||
                !sor::search::save_planbb_model(planbb_out, model)) {
                std::fprintf(stderr, "error: PlanB&B fit/save failed\n");
                rc = 1;
            } else {
                std::printf("planbb_model:      %s\n", planbb_out.c_str());
            }
        }
    }

    if (want_dynsep) {
        if (dynsep_pool.samples.size() < 2) {
            std::fprintf(stderr,
                         "warn: too few DynSep samples (%zu); skip fit\n",
                         dynsep_pool.samples.size());
            rc = 1;
        } else {
            sor::search::DynSepFitOptions fit;
            fit.emb_dim = sor::search::kDynSepEmbDim;
            fit.n_msg_layers = sor::search::kDynSepMsgLayers;
            fit.sgd_epochs = 40;
            fit.sgd_lr = 0.05;
            auto model = sor::search::fit_dynsep_imitation(dynsep_pool, fit);
            if (!model.loaded ||
                !sor::search::save_dynsep_model(dynsep_out, model)) {
                std::fprintf(stderr, "error: DynSep fit/save failed\n");
                rc = 1;
            } else {
                std::printf("dynsep_model:      %s  (samples=%zu)\n",
                            dynsep_out.c_str(), dynsep_pool.samples.size());
            }
        }
    }

    return rc;
}
