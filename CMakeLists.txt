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

# Setup variables for the cndarray project
set(CNDARRAY_ROOT  ctypes_ndarray )
set(CNDARRAY_INCLUDE_DIR ${CNDARRAY_ROOT}/src/ )

# Setup variables for the vlfeat project
# Note: minimum number of compilation units are chosen to get sift working
set(VLFEAT_ROOT  vlfeat )
set(VLFEAT_INCLUDE_DIR ${VLFEAT_ROOT} )
set(VLFEAT_SRCS ${VLFEAT_ROOT}/vl/sift.c ${VLFEAT_ROOT}/vl/generic.c
  ${VLFEAT_ROOT}/vl/imopv.c ${VLFEAT_ROOT}/vl/imopv_sse2.c
  ${VLFEAT_ROOT}/vl/host.c ${VLFEAT_ROOT}/vl/random.c)

# Setup the sources that should be compiled
set(SRCS src/Spectavi.cpp)
set(SRCS ${SRCS} ${VLFEAT_SRCS} )

add_library(spectavi SHARED ${SRCS})
target_include_directories(spectavi PRIVATE ${EIGEN3_INCLUDE_DIR}
  ${CNDARRAY_INCLUDE_DIR} ${VLFEAT_INCLUDE_DIR})

