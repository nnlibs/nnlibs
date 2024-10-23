# include_directories(${CMAKE_CURRENT_LIST_DIR}/utils)

add_executable(TensorTest ${CMAKE_CURRENT_LIST_DIR}/test_tensor.cpp)
target_link_libraries(
    TensorTest PRIVATE nn_lib_test
)
add_test(NAME TensorTest COMMAND TensorTest)

add_executable(ActivationTest ${CMAKE_CURRENT_LIST_DIR}/ops/test_activation.cpp)
target_link_libraries(
    ActivationTest PRIVATE nn_lib_test ops
)
add_test(NAME ActivationTest COMMAND ActivationTest)

add_executable(ConvTest ${CMAKE_CURRENT_LIST_DIR}/ops/test_convolution.cpp)
target_link_libraries(
    ConvTest PRIVATE nn_lib_test ops
)
add_test(NAME ConvTest COMMAND ConvTest)

add_executable(AttentionTest ${CMAKE_CURRENT_LIST_DIR}/ops/test_attention.cpp)
target_link_libraries(
    AttentionTest PRIVATE nn_lib_test ops
)
add_test(NAME AttentionTest COMMAND AttentionTest)

add_executable(NetTest ${CMAKE_CURRENT_LIST_DIR}/e2e_test.cpp)
target_link_libraries(
    NetTest PRIVATE nn_lib_test ops
)
add_test(NAME NetTest COMMAND NetTest)