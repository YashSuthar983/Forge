add_library(forge_certify src/finalize.cpp src/basis_certificate.cpp)
add_library(forge::certify ALIAS forge_certify)
target_include_directories(forge_certify PUBLIC include)
target_link_libraries(forge_certify PUBLIC forge_core forge_model PRIVATE forge_flags)
