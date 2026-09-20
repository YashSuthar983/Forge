module SovereignSolver

# C++-port modules first (L0–L7), then the original solver_accl engines
# (one-sided LPProblem + KA PDHG). Include order is a dependency DAG.

include("core/Result.jl")
include("sparse/Sparse.jl")
include("model/Lp.jl")
include("certify/Finalize.jl")
include("io/Mps.jl")
include("backend/Device.jl")
include("presolve/Presolve.jl")
include("engines/Farkas.jl")
include("engines/Pdhg.jl")
include("engines/Hpr.jl")
include("engines/Qp.jl")
include("types.jl")
include("linalg/LinAlgCore.jl")
include("linalg/SparseLU.jl")
include("engines/Simplex.jl")
include("engines/QpActiveSet.jl")
include("engines/QpInteriorPoint.jl")
include("engines/QpAuto.jl")
include("simplex/SimplexCore.jl")
include("milp/BranchAndBound.jl")
include("gpu/DeviceUtils.jl")
include("gpu/PDHGCore.jl")
include("gpu/BatchedPDHGCore.jl")
include("gpu/CostModel.jl")
include("gpu/QpHprCore.jl")
include("gpu/BatchedQpHprCore.jl")
include("search/Propagate.jl")
include("search/Cuts.jl")
include("search/Bab.jl")
include("search/MiqpBab.jl")
include("cli/solve.jl")
include("cli/check.jl")
include("cli/gen.jl")
include("api/Registry.jl")
include("api/Convert.jl")
include("api/Gpu.jl")
include("api/Server.jl")

using .SorCore
using .SorSparse
using .SorModel
using .SorCertify
using .SorIO
using .SorBackend: make_cpu_lp_device, make_cpu_backend
using .SorPresolve
using .SorFarkas
using .SorPdhg: solve_pdhg, pdhg_evidence, PdhgOptions
using .SorHpr: solve_hpr, hpr_evidence, HprOptions
using .SorQp
using .CoreTypes
using .LinAlgCore
using .SparseLU
using .SorSimplex: solve_simplex, simplex_evidence, SimplexOptions, SimplexDiagnostics, Primal, Dual, Auto
using .SorQpActiveSet
using .SorQpIpm
using .SorQpAuto
using .SimplexCore
using .BranchAndBoundCore: solve_milp, MILPResult
using .GPUDeviceUtils
using .FirstOrderCore
using .BatchedFirstOrderCore
using .SorCostModel
using .QpHprCore
using .BatchedQpHprCore
using .SorPropagate
using .SorSearch: BabOptions, BabDiagnostics, milp_evidence, solve_milp as solve_mip
using .SorSearchQp: MiqpOptions, MiqpDiagnostics, MiqpResult, solve_miqp
using .SorCLI: sor_solve_main
using .SorCheck: sor_check_main
using .SorGen: sor_gen_main
using .SorApi
using .SorApiConvert: to_one_sided, recover_solution, OneSidedMap
using .SorApiGpu: detect_gpu, gpu_info, GpuProbe
using .SorApiServer: serve, handle_line, PROTOCOL_VERSION

export Status, ProofLevel, to_string, human_line, ProofEvidence, RawResult, SolveResult
export NotSolved, Optimal, Infeasible, Unbounded, InfeasibleOrUnbounded
export Feasible, NoSolutionFound, Interrupted, NumericalFailure, Unsupported
export None, BoundOnly, FeasibleOnly, FeasibleWithGap, ProvedKKT
export ProvedGlobalEpsilon, ProvedOptimalFP, ProvedOptimalExact, ProvedOptimalCertified
export finalize_result
export SparsePattern, CsrMatrix, CscMatrix, from_triplets, to_csc
export LpProblem, kInf, validate!, objective, max_row_violation, max_bound_violation, n_integer
export LPProblem, MILPProblem, Sense, LE, GE, EQ, LPStatus, OPTIMAL, INFEASIBLE, UNBOUNDED, LPResult
export solve_linear, lu_factorize, lu_solve, lu_solve_transpose, LUFactorization, SingularMatrixError
export BasisFactorization, ftran, btran, refactorize!
export LuOptions, LuStats, SparseBasisFactor, factorize!, ftran!, btran!, update!, needs_refactor
export solve_lp
export solve_milp, MILPResult
export solve_lp_firstorder, FirstOrderResult
export solve_lp_firstorder_batched, BatchedFirstOrderResult
export solve_simplex, simplex_evidence, SimplexOptions, SimplexDiagnostics, Primal, Dual, Auto
export solve_pdhg, pdhg_evidence, PdhgOptions
export solve_hpr, hpr_evidence, HprOptions
export read_mps_file, read_mps_file_auto, write_mps_file
export make_cpu_lp_device, make_cpu_backend
export sor_solve_main, sor_check_main, sor_gen_main
export BabOptions, BabDiagnostics, milp_evidence, solve_mip
export EngineKind, CPU_ENGINE, GPU_ENGINE, ProblemClass, CLASS_LP, CLASS_MILP, CLASS_QP, CLASS_MIQP
export EngineSpec, ENGINES, engines, find_engine, engine_names, kind_string, class_string
export to_one_sided, recover_solution, OneSidedMap
export detect_gpu, gpu_info, GpuProbe
export serve, handle_line, PROTOCOL_VERSION

end # module
