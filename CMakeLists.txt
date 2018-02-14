cmake_minimum_required(VERSION 3.3)
project(Scientific_Visualization)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_custom_target(Scientific_Visualization COMMAND make -C ${Scientific_Visualization_SOURCE_DIR}
        CLION_EXE_DIR=${PROJECT_BINARY_DIR})

set(TESTS_RESOURCES tests.c)
add_executable(tests ${TESTS_RESOURCES})
target_link_libraries(tests glut GL GLU m)