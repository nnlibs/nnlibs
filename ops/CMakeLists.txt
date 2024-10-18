# aux_source_directory(. SRC_LIST)
file(GLOB_RECURSE OPS_SRC "./*.cpp" "./*.h")
message(${OPS_SRC})

add_library(ops SHARED ${OPS_SRC})
target_include_directories(ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(ops ${LIB_LIST})