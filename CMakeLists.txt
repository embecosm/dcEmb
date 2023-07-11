#
# Copyright (C) 2022 Embecosm Limited
#
# Contributor William Jones <william.jones@embecosm.com>
# Contributor Elliot Stein <E.Stein@soton.ac.uk>
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

cmake_minimum_required(VERSION 3.16)

project(dcEmb LANGUAGES CXX VERSION 1.0.1 DESCRIPTION "dcEmb, fast Dynamic Causal Modeling")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(UseSphinxDoc)
include(FindSphinx)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-std=c++14 -O3 -ffast-math -fno-unsafe-math-optimizations -flto -march=native\
 -D_GLIBCXX_PARALLEL")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}-O3")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}-O3")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenMP)
find_package(GTest REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

# add_subdirectory(${CMAKE_SOURCE_DIR}/lib/googletest)
# add_subdirectory(${CMAKE_SOURCE_DIR}/lib/eigen)
enable_testing()

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/3body
  ${CMAKE_SOURCE_DIR}/include/tests
  # ${EIGEN3_INCLUDE_DIRS}
  ${gtest_SOURCE_DIR}
  ${gtest_SOURCE_DIR}/include
)

set(SOURCES_DCM
  src/dynamic_model.cc
  src/utility.cc
)

set(SOURCES_TESTS
  tests/utility_test.cc
  src/3body/dynamic_3body_model.cc
  tests/dynamic_3body_model_test.cc
)

# look for Doxygen package
find_package(Doxygen)

if(DOXYGEN_FOUND)
  # set input and output files
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/source/doxygen/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.out)
  message(${DOXYGEN_IN})

  # request to configure the file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  message("Doxygen build started")

  # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
  add_custom_target(doxy
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
else(DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif(DOXYGEN_FOUND)

add_sphinx_doc(
  SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/docs/source/sphinx/source
  BUILD_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/docs/
  CACHE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/docs/.doctrees
  HTML_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/docs/
  CONF_FILE
  ${CMAKE_CURRENT_SOURCE_DIR}/docs/source/sphinx/source/conf.py
  TARGET_NAME
  sphinx
  COMMENT
  "HTML documentation"
)

# Documentation
add_custom_target(docs)
add_dependencies(docs doxy sphinx)

# Add dcm as library
add_library(dcEmb SHARED
  src/dynamic_model.cc
  src/utility.cc
)
target_link_libraries(dcEmb PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen)

set_target_properties(dcEmb PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER
  "include/dynamic_model.hh;include/bma_model.hh;include/bmr_model.hh;include/peb_model.hh"
)

# Run short form tests
add_executable(run_tests ${SOURCES_TESTS})

if(OpenMP_FOUND)
  target_link_libraries(run_tests PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

target_link_libraries(run_tests PUBLIC gtest gtest_main dcEmb Eigen3::Eigen)
set_target_properties(run_tests PROPERTIES COMPILE_FLAGS "")

install(TARGETS dcEmb
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_file(dcEmb.pc.in dcEmb.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dcEmb.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Embecosm") # required
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "1")

INCLUDE(CPack)