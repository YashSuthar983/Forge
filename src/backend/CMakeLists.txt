# ---- Vulkan SPIR-V shaders (optional) -------------------------------------
# The .spv outputs stay under the TOP-LEVEL binary dir so the runtime lookup
# path (build/shaders) is unchanged by this module owning its own CMakeLists.
set(FORGE_SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders)
set(FORGE_SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)
set(FORGE_SHADER_NAMES
  spmv_csr spmv_csc primal_step dual_step halpern_mix avg_update)

if(FORGE_ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
  file(MAKE_DIRECTORY ${FORGE_SHADER_BINARY_DIR})
  set(FORGE_SPV_OUTPUTS)
  foreach(s ${FORGE_SHADER_NAMES})
    set(_src ${FORGE_SHADER_SOURCE_DIR}/${s}.comp)
    set(_spv ${FORGE_SHADER_BINARY_DIR}/${s}.spv)
    add_custom_command(
      OUTPUT ${_spv}
      COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.2 -o ${_spv} ${_src}
      DEPENDS ${_src}
      COMMENT "SPIR-V ${s}.comp"
      VERBATIM)
    list(APPEND FORGE_SPV_OUTPUTS ${_spv})
  endforeach()
  add_custom_target(forge_shaders ALL DEPENDS ${FORGE_SPV_OUTPUTS})
endif()

set(FORGE_BACKEND_SOURCES
  src/cpu_backend.cpp
  src/backend_factory.cpp
  src/cpu_lp_device.cpp)
if(FORGE_ENABLE_VULKAN)
  list(APPEND FORGE_BACKEND_SOURCES src/vulkan/vk_context.cpp src/vulkan/vk_lp_device.cpp)
else()
  list(APPEND FORGE_BACKEND_SOURCES src/vulkan_stub.cpp)
endif()
add_library(forge_backend ${FORGE_BACKEND_SOURCES})
add_library(forge::backend ALIAS forge_backend)
target_include_directories(forge_backend PUBLIC include)
target_link_libraries(forge_backend PUBLIC forge_sparse PRIVATE forge_flags)

if(FORGE_ENABLE_VULKAN)
  target_link_libraries(forge_backend PUBLIC Vulkan::Vulkan)
  target_compile_definitions(forge_backend PUBLIC FORGE_ENABLE_VULKAN=1)
  target_compile_definitions(forge_backend PRIVATE
    FORGE_SHADER_DIR=\"${FORGE_SHADER_BINARY_DIR}\")
  add_dependencies(forge_backend forge_shaders)
endif()
