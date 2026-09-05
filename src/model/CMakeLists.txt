add_library(forge_model src/lp.cpp)
add_library(forge::model ALIAS forge_model)
target_include_directories(forge_model PUBLIC include)
target_link_libraries(forge_model PUBLIC forge_sparse PRIVATE forge_flags)
