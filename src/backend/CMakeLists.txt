# ---- Vulkan SPIR-V shaders (optional) -------------------------------------
# The .spv outputs stay under the TOP-LEVEL binary dir so the runtime lookup
# path (build/shaders) is unchanged by this module owning its own CMakeLists.
set(SOR_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders)
set(SOR_SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)
set(SOR_SHADER_NAMES
  spmv_csr spmv_csc primal_step dual_step halpern_mix avg_update
  step_reduce kkt_cols kkt_rows kkt_final
  )
# The QP shaders (qp_x_step, qp_v_build, qp_y_step, qp_w_step, qp_halpern,
# qp_axpby, qp_dot, qp_boxres, qp_absmax, qp_fold) are named here in the
# commit that adds the Vulkan QpDevice. Listing them before they exist makes
# the shader target unbuildable, and make_vulkan_qp_device() returns nullptr
# until then, so nothing can reach them anyway.

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
  src/cpu_qp_device.cpp)
if(SOR_ENABLE_VULKAN)
  list(APPEND SOR_BACKEND_SOURCES src/vulkan/vk_context.cpp src/vulkan/vk_lp_device.cpp
                                  src/vulkan/vk_qp_device.cpp)
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
