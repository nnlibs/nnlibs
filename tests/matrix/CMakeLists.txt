add_executable(MatrixTest ${CMAKE_CURRENT_LIST_DIR}/test.cpp)
target_link_libraries(MatrixTest PRIVATE nn_lib_test openblas)
