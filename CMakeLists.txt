cmake_minimum_required(VERSION 3.3)
project(Scientific_Visualization)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET_SOURCE_FILES_PROPERTIES(fluids.c PROPERTIES LANGUAGE CXX )
#add_custom_target(Scientific_Visualization COMMAND make -C ${Scientific_Visualization_SOURCE_DIR}
#        CLION_EXE_DIR=${PROJECT_BINARY_DIR})


# Add common dependencies for macOS and Linux ##########################################################################
include_directories(${PROJECT_SOURCE_DIR}/fftw-2.1.5/include $ENV{HOME}/.local/include)
link_directories(${PROJECT_SOURCE_DIR}/fftw-2.1.5/lib $ENV{HOME}/.local/lib)

file(GLOB CPP_FILES src/*.cpp)
file(GLOB HEADER_FILES src/*.h)
set(SOURCES ${CPP_FILES} ${HEADER_FILES})

message("INCLUDE DIRECTORIES: ${PROJECT_SOURCE_DIR}/fftw-2.1.5/include")

## OS-specific dependencies ############################################################################################
if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif ()

if (LINUX)
    message(STATUS "Linux environment detected")
    set(LIBS glui glut GL GLU m rfftw fftw)
    set(CMAKE_C_FLAGS "-O3 -march=x86-64 -mtune=generic -DNDEBUG -mfpmath=sse -ffast-math -Wall -pipe")
    add_definitions(-DLINUX)
elseif(APPLE)
    message(STATUS "macOs environment detected")
    find_package(GLUT REQUIRED)
    find_package(OpenGL REQUIRED)
    include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
    set(LIBS rfftw fftw glui ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
    set(CMAKE_C_FLAGS "-O0 -Wall -pipe")
    add_definitions(-DMACOS)
else()
    message(FATAL_ERROR "Your OS is not compatible with this project. This project can only be built on LINUX or APPLE environments")
endif()

message(STATUS "The following libraries will be linked ${LIBS}")
message("Linking directories: ")

## smoke target ########################################################################################################
set(SOURCES_SMOKE fluids.c)
add_executable(smoke ${SOURCES_SMOKE})
target_link_libraries(smoke ${LIBS})

## Test target #########################################################################################################
set(TESTS_SOURCES tests.c)
add_executable(tests ${TESTS_SOURCES})
target_link_libraries(tests ${LIBS})

## new architecture ####################################################################################################
set(SOURCES_NEW simulation.cpp visualization.cpp interaction.cpp visualization.h simulation.h)
add_executable(new_arch ${SOURCES_NEW})
target_link_libraries(new_arch ${LIBS})
