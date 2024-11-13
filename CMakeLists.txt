cmake_minimum_required(VERSION 3.12)
project(nn_lib)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/nn_lib)

add_subdirectory(3rdparty/Catch2)
add_subdirectory(3rdparty/OpenBLAS)
include_directories(3rdparty/OpenBLAS)
include_directories(${CMAKE_BINARY_DIR}/3rdparty/OpenBLAS)

add_library(nn_lib_test STATIC tests/test_main.cpp)
target_link_libraries(nn_lib_test PUBLIC Catch2::Catch2)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(tests)

add_subdirectory(ops)
