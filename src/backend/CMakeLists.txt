# ---- Vulkan SPIR-V shaders (optional) -------------------------------------
# The .spv outputs stay under the TOP-LEVEL binary dir so the runtime lookup
# path (build/shaders) is unchanged by this module owning its own CMakeLists.
set(SOR_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders)
set(SOR_SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)
set(SOR_SHADER_NAMES
  spmv_csr spmv_csc primal_step dual_step halpern_mix halpern_fixed avg_update
  step_reduce kkt_cols kkt_rows kkt_final
  qp_x_step qp_v_build qp_y_step qp_w_halpern
  vec_axpby vec_reduce qp_kkt_rows qp_kkt_cols
  pdhcg_init pdhcg_diag_prox pdhcg_grad pdhcg_trial pdhcg_bb pdhcg_advance_x pdhcg_advance_y
  pdhcg_eval_cols pdhcg_eval_rows
  bq_gain bq_select bq_apply bq_save_best bq_perturb bq_rebuild_h bq_rebuild_r bq_rescore
  spmvb_csr spmvb_csc vec_copy_lanes vec_reduce_lanes
  pdhcgb_diag_prox pdhcgb_grad pdhcgb_trial pdhcgb_bb pdhcgb_advance_x pdhcgb_advance_y
  pdhcgb_eval_cols pdhcgb_eval_rows
  bench_fma_f32 bench_fma_f64)   # sor_gpu_bench only

if(SOR_ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
  file(MAKE_DIRECTORY ${SOR_SHADER_BINARY_DIR})
  set(SOR_SPV_OUTPUTS)
  foreach(s ${SOR_SHADER_NAMES})
    set(_src ${SOR_SHADER_SOURCE_DIR}/${s}.comp)
    set(_spv ${SOR_SHADER_BINARY_DIR}/${s}.spv)
    add_custom_command(
      OUTPUT ${_spv}
      COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.2 -o ${_spv} ${_src}
      DEPENDS ${_src}
      COMMENT "SPIR-V ${s}.comp"
      VERBATIM)
    list(APPEND SOR_SPV_OUTPUTS ${_spv})
  endforeach()
  add_custom_target(sor_shaders ALL DEPENDS ${SOR_SPV_OUTPUTS})
endif()

set(SOR_BACKEND_SOURCES
  src/cpu_backend.cpp
  src/backend_factory.cpp
  src/cpu_lp_device.cpp
  src/cpu_qp_device.cpp
  src/cpu_pdhcg_device.cpp
  src/pdhcg_lp_diagnostics.cpp
  src/cpu_binquad_device.cpp
  src/lanes_pdhcg_device.cpp
  src/lp_device_batch.cpp)
if(SOR_ENABLE_VULKAN)
  list(APPEND SOR_BACKEND_SOURCES src/vulkan/vk_context.cpp src/vulkan/vk_compute.cpp src/vulkan/vk_profiler.cpp
                                  src/vulkan/vk_lp_device.cpp
                                  src/vulkan/vk_qp_device.cpp
                                  src/vulkan/vk_pdhcg_device.cpp
                                  src/vulkan/vk_binquad_device.cpp
                                  src/vulkan/vk_batched_pdhcg_device.cpp)
else()
  list(APPEND SOR_BACKEND_SOURCES src/vulkan_stub.cpp)
endif()
add_library(sor_backend ${SOR_BACKEND_SOURCES})
add_library(sor::backend ALIAS sor_backend)
target_include_directories(sor_backend PUBLIC include)
target_link_libraries(sor_backend PUBLIC sor_sparse PRIVATE sor_flags)

if(SOR_ENABLE_VULKAN)
  target_link_libraries(sor_backend PUBLIC Vulkan::Vulkan)
  target_compile_definitions(sor_backend PUBLIC SOR_ENABLE_VULKAN=1)
  target_compile_definitions(sor_backend PRIVATE
    SOR_SHADER_DIR=\"${SOR_SHADER_BINARY_DIR}\")
  add_dependencies(sor_backend sor_shaders)
endif()
