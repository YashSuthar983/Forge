add_library(sor_model src/lp.cpp)
add_library(sor::model ALIAS sor_model)
target_include_directories(sor_model PUBLIC include)
target_link_libraries(sor_model PUBLIC sor_sparse PRIVATE sor_flags)
