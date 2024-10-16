cmake_minimum_required(VERSION 3.12)
project(nn_lib)

set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/nn_lib)
# option(USE_MLX "Use mlx interface instead of vsx interface" ON)

add_subdirectory(3rdparty/Catch2)
add_library(nn_lib_test STATIC tests/test_main.cpp)
target_link_libraries(nn_lib_test PUBLIC Catch2::Catch2)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(tests)

# add_subdirectory(ops)
