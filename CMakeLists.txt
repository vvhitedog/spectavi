cmake_minimum_required(VERSION 3.1)
project(spectavi)

set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 1)
set(PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH} )
set(PROJECT_AUTHOR Matt Gara )
set(PROJECT_AUTHOR_EMAIL gara.matt@gmail.com )
set(PROJECT_DESCRIPTION "A minimalistic multi-view stereo and geometry library.")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE on)

find_package(Eigen3 REQUIRED)

set(CNDARRAY_ROOT  ctypes_ndarray )
set(CNDARRAY_INCLUDE_DIR ${CNDARRAY_ROOT}/src/ )

set(SRCS src/Spectavi.cpp)

add_library(spectavi SHARED ${SRCS})
target_include_directories(spectavi PRIVATE ${EIGEN3_INCLUDE_DIR} ${CNDARRAY_INCLUDE_DIR})

