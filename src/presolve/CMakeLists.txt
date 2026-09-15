add_library(sor_presolve src/presolve.cpp src/presolve_live.cpp src/presolve_recovery.cpp)
add_library(sor::presolve ALIAS sor_presolve)
target_include_directories(sor_presolve PUBLIC include)
target_link_libraries(sor_presolve PUBLIC sor_model PRIVATE sor_flags sor_certify)
