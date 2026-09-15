add_executable(forge_solve forge_solve.cpp)
target_link_libraries(forge_solve PRIVATE
  forge_io forge_engines forge_search forge_certify forge_backend forge_json forge_flags)

add_executable(forge_milp_train forge_milp_train.cpp)
target_link_libraries(forge_milp_train PRIVATE
  forge_io forge_engines forge_search forge_flags)

add_executable(forge_gen forge_gen.cpp)
target_link_libraries(forge_gen PRIVATE forge_io forge_flags)

add_executable(forge_check forge_check.cpp)
target_link_libraries(forge_check PRIVATE forge_io forge_engines forge_certify forge_flags)

add_executable(forge_ext_demo forge_ext_demo.cpp)
target_link_libraries(forge_ext_demo PRIVATE forge_search forge_certify forge_flags)
