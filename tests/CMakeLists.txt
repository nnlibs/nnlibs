# include_directories(${CMAKE_CURRENT_LIST_DIR}/utils)

add_executable(TensorTest ${CMAKE_CURRENT_LIST_DIR}/test_tensor.cpp)
target_link_libraries(
    TensorTest PRIVATE nn_lib_test
)
add_test(NAME TensorTest COMMAND TensorTest)

   
