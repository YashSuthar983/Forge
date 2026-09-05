add_library(sor_presolve src/presolve.cpp)
add_library(sor::presolve ALIAS sor_presolve)
target_include_directories(sor_presolve PUBLIC include)
target_link_libraries(sor_presolve PUBLIC sor_model PRIVATE sor_flags)
