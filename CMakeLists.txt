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

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings using cython" ON)
option(BUILD_DOCS "Build docs using Sphinx and Doxygen" ON)
option(BUILD_TESTS "Build tests using googletest" ON)
option(STATIC_BUILD "Build a static library" OFF)
option(USE_OPENMP "Build with OpenMP for Eigen parallelisation" ON)
option(OPTIMIZE_FASTMATH_NATIVE "Build with -march=native and -ffast-math" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Basic CMake setup
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

if(OPTIMIZE_FASTMATH_NATIVE)
  set(FAST_MATH_FLAGS -ffast-math -fno-unsafe-math-optimizations)
  CHECK_CXX_COMPILER_FLAG("${FAST_MATH_FLAGS}" COMPILER_SUPPORTS_FAST_MATH)
  CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)

  if (COMPILER_SUPPORTS_FAST_MATH)
    add_compile_options(${FAST_MATH_FLAGS})
  endif()

  if (COMPILER_SUPPORTS_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# Configure for OpenMP
if(USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
    add_compile_definitions(_GLIBCXX_PARALLEL)
  endif()
endif()

find_package(Eigen3 3.4 REQUIRED)

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/3body
  ${CMAKE_SOURCE_DIR}/include/tests
  ${gtest_SOURCE_DIR}
  ${gtest_SOURCE_DIR}/include
)

# Library sources
set(SOURCES_DCM
  src/dynamic_model.cc
  src/utility.cc
)

# Test sources
set(SOURCES_TESTS
  tests/utility_test.cc
  src/3body/dynamic_3body_model.cc
  tests/dynamic_3body_model_test.cc
)

# Library config
if(STATIC_BUILD)
  add_library(dcEmb STATIC ${SOURCES_DCM})
else()
  add_library(dcEmb SHARED ${SOURCES_DCM})
endif()

target_link_libraries(dcEmb PUBLIC Eigen3::Eigen)
target_include_directories(dcEmb
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
set_target_properties(dcEmb PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

# Tests config
if(BUILD_TESTS)
  find_package(GTest REQUIRED)
  enable_testing()

  # Run short form tests
  add_executable(run_tests ${SOURCES_TESTS})

  # OpenMP is already added earlier if enabled
  target_link_libraries(run_tests PUBLIC gtest gtest_main dcEmb Eigen3::Eigen)
endif()

# Installation config
install(TARGETS dcEmb EXPORT dcEmb
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES
  include/dynamic_model.hh
  include/bma_model.hh
  include/bmr_model.hh
  include/peb_model.hh
  include/utility.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dcEmb
)

install(EXPORT dcEmb
  DESTINATION cmake
  FILE dcEmbConfig.cmake
)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python.stamp
    COMMAND ${Python3_EXECUTABLE} setup.py build_ext
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/python.stamp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
    COMMENT "Building Python bindings to 'dcEmb' library"
  )
  add_custom_target(dcEmb-python-bindings ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python.stamp)
endif()

if(BUILD_DOCS)
  # look for Doxygen package
  find_package(Doxygen)

  if (DOXYGEN_FOUND)
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
  else()
    message("Doxygen need to be installed to generate the doxygen documentation")
  endif()

  include(UseSphinxDoc)
  include(FindSphinx)

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
endif()

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Embecosm") # required
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "1")

INCLUDE(CPack)
