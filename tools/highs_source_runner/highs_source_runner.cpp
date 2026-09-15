#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "Highs.h"

namespace {

constexpr const char* kSchema = "sor-highs-source-runner-v1";

const char* status_name(const HighsStatus status) {
  switch (status) {
    case HighsStatus::kOk: return "kOk";
    case HighsStatus::kWarning: return "kWarning";
    case HighsStatus::kError: return "kError";
  }
  return "kUnknown";
}

struct Args {
  std::string model;
  double time_limit = 0.0;
  double tolerance = 0.0;
  int seed = 0;
  bool relax_integrality = false;
  bool has_small_matrix_value = false;
  double small_matrix_value = 0.0;
  bool identity = false;
};

double parse_positive(const char* name, const char* text) {
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (!end || *end != '\0' || !std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(std::string(name) + " must be finite and positive");
  }
  return value;
}

Args parse_args(const int argc, char** argv) {
  Args args;
  if (argc == 2 && std::string(argv[1]) == "--identity") {
    args.identity = true;
    return args;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " needs a value");
      return argv[++i];
    };
    if (arg == "--model") {
      args.model = value("--model");
    } else if (arg == "--time-limit") {
      args.time_limit = parse_positive("--time-limit", value("--time-limit"));
    } else if (arg == "--tol") {
      args.tolerance = parse_positive("--tol", value("--tol"));
    } else if (arg == "--seed") {
      char* end = nullptr;
      const long parsed = std::strtol(value("--seed"), &end, 10);
      if (!end || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
          parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error("--seed must be an integer");
      }
      args.seed = static_cast<int>(parsed);
    } else if (arg == "--relax-integrality") {
      args.relax_integrality = true;
    } else if (arg == "--small-matrix-value") {
      args.small_matrix_value = parse_positive(
          "--small-matrix-value", value("--small-matrix-value"));
      args.has_small_matrix_value = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (args.model.empty() || args.time_limit <= 0.0 || args.tolerance <= 0.0) {
    throw std::runtime_error("required: --model FILE --time-limit S --tol T --seed N");
  }
  return args;
}

nlohmann::json build_identity(const Highs& highs) {
  return nlohmann::json{
      {"kind", "sor-highs-source-api-runner"},
      {"runner_version", 1},
      {"build_type", "Release"},
      {"optimization", "-O3"},
      {"ndebug", true},
      {"native_arch", true},
      {"compiler", SOR_RUNNER_COMPILER},
      {"compiler_id", SOR_RUNNER_COMPILER_ID},
      {"compiler_version", SOR_RUNNER_COMPILER_VERSION},
      {"highs_source_dir", SOR_HIGHS_SOURCE_DIR},
      {"highs_build_dir", SOR_HIGHS_BUILD_DIR},
      {"highs_source_commit", SOR_HIGHS_SOURCE_COMMIT},
      {"highs_source_dirty", false},
      {"highs_githash", highs.githash()},
  };
}

template <typename T>
bool set_required_option(Highs& highs, const char* name, const T& value,
                         std::string& rejected) {
  const HighsStatus status = highs.setOptionValue(name, value);
  if (status == HighsStatus::kOk) return true;
  if (!rejected.empty()) rejected += "; ";
  rejected += std::string(name) + "=" + status_name(status);
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    Highs highs;
    if (args.identity) {
      nlohmann::json out = {
          {"schema", kSchema},
          {"kind", "identity"},
          {"highs_version", highs.version()},
          {"build_identity", build_identity(highs)},
      };
      std::cout << out.dump() << '\n';
      return 0;
    }

    std::string rejected;
    bool options_ok = true;
    options_ok &= set_required_option(highs, "output_flag", false, rejected);
    options_ok &= set_required_option(highs, "log_to_console", false, rejected);
    options_ok &= set_required_option(highs, "threads", HighsInt{1}, rejected);
    options_ok &= set_required_option(highs, "parallel", std::string("off"), rejected);
    options_ok &= set_required_option(highs, "time_limit", args.time_limit, rejected);
    options_ok &= set_required_option(
        highs, "primal_feasibility_tolerance", args.tolerance, rejected);
    options_ok &= set_required_option(
        highs, "dual_feasibility_tolerance", args.tolerance, rejected);
    options_ok &= set_required_option(highs, "random_seed", HighsInt{args.seed}, rejected);
    options_ok &= set_required_option(highs, "solver", std::string("choose"), rejected);
    options_ok &= set_required_option(highs, "presolve", std::string("choose"), rejected);
    options_ok &= set_required_option(highs, "run_crossover", std::string("on"), rejected);
    options_ok &= set_required_option(
        highs, "solve_relaxation", args.relax_integrality, rejected);
    if (args.has_small_matrix_value) {
      options_ok &= set_required_option(
          highs, "small_matrix_value", args.small_matrix_value, rejected);
    }
    if (!options_ok) throw std::runtime_error("required option rejected: " + rejected);

    // Model import is explicitly outside both measured solve intervals.
    const HighsStatus read_status = highs.readModel(args.model);
    if (read_status != HighsStatus::kOk) {
      throw std::runtime_error(
          std::string("readModel failed: ") + status_name(read_status));
    }
    const double highs_before = highs.getRunTime();
    const auto wall_before = std::chrono::steady_clock::now();
    const HighsStatus run_status = highs.run();
    const auto wall_after = std::chrono::steady_clock::now();
    const double solve_seconds = highs.getRunTime() - highs_before;
    const double measured_wall =
        std::chrono::duration<double>(wall_after - wall_before).count();

    nlohmann::json configuration = {
        {"output_flag", false},
        {"log_to_console", false},
        {"threads", 1},
        {"parallel", "off"},
        {"time_limit", args.time_limit},
        {"primal_feasibility_tolerance", args.tolerance},
        {"dual_feasibility_tolerance", args.tolerance},
        {"random_seed", args.seed},
        {"solver", "choose"},
        {"presolve", "choose"},
        {"run_crossover", "on"},
        {"solve_relaxation", args.relax_integrality},
        {"small_matrix_value",
         args.has_small_matrix_value ? nlohmann::json(args.small_matrix_value)
                                     : nlohmann::json(nullptr)},
    };

    nlohmann::json out = {
        {"schema", kSchema},
        {"kind", "solve_result"},
        {"read_status", status_name(read_status)},
        {"run_status", status_name(run_status)},
        {"options_applied", true},
        {"model_status", highs.modelStatusToString(highs.getModelStatus())},
        {"objective", highs.getObjectiveValue()},
        {"solve_seconds", solve_seconds},
        {"measured_wall_seconds", measured_wall},
        {"simplex_iterations", highs.getInfo().simplex_iteration_count},
        {"highs_version", highs.version()},
        {"configuration", std::move(configuration)},
        {"build_identity", build_identity(highs)},
    };
    std::cout << out.dump() << '\n';
    return run_status == HighsStatus::kOk ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "sor_highs_source_runner: " << error.what() << '\n';
    return 2;
  }
}
