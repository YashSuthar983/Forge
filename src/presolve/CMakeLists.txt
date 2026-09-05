add_library(forge_presolve src/presolve.cpp)
add_library(forge::presolve ALIAS forge_presolve)
target_include_directories(forge_presolve PUBLIC include)
target_link_libraries(forge_presolve PUBLIC forge_model PRIVATE forge_flags)
