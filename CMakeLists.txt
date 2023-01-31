#
# Copyright (C) 2022 Embecosm Limited
#
# Contributor William Jones <william.jones@embecosm.com>
# Contributor Elliot Stein <E.Stein@soton.ac.uk>
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

cmake_minimum_required(VERSION 3.16)

project(dcm_demo LANGUAGES CXX)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}-std=c++14 -O3 -ffast-math -fno-unsafe-math-optimizations -flto -march=native\
 -D_GLIBCXX_PARALLEL")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}-O3 -g")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}-O3 -g")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

SET(EIGEN)
add_subdirectory(${CMAKE_SOURCE_DIR}/lib/googletest)
add_subdirectory(${CMAKE_SOURCE_DIR}/lib/eigen)
enable_testing()

include_directories(
    ${CMAKE_SOURCE_DIR}/lib/cereal/include
    ${CMAKE_SOURCE_DIR}/lib/eigen
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/COVID
    ${CMAKE_SOURCE_DIR}/include/3body
    ${CMAKE_SOURCE_DIR}/include/tests
    ${gtest_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}
)

set(SOURCES_COVID
    src/dynamic_model.cc
    src/COVID/run_COVID_dcm.cc
    src/COVID/DEM_COVID.cc
    src/COVID/import_COVID.cc
    src/COVID/dynamic_COVID_model.cc
    src/utility.cc
)

set(SOURCES_3BODY
    src/dynamic_model.cc
    src/3body/run_3body_dcm.cc
    src/3body/DEM_3body.cc
    src/3body/dynamic_3body_model.cc
    src/utility.cc
)

set(SOURCES_TESTS
    src/COVID/import_COVID.cc
    tests/import_COVID_test.cc
    src/utility.cc
    tests/utility_test.cc
    src/dynamic_model.cc
    src/3body/dynamic_3body_model.cc
    tests/dynamic_3body_model_test.cc
)

set(SOURCES_TESTS_LONG
    src/utility.cc
    src/dynamic_model.cc
    src/COVID/import_COVID.cc
    src/COVID/dynamic_COVID_model.cc
    tests/dynamic_COVID_model_test.cc
)

set(SOURCES_SERIALIZATION_TESTS
    src/dynamic_model.cc
    src/3body/dynamic_3body_model.cc
    src/COVID/import_COVID.cc
    src/COVID/dynamic_COVID_model.cc
    src/utility.cc
    tests/serialization_test.cc
)

# add_executable(dcm_covid ${SOURCES_COVID})

# if(OpenMP_FOUND)
#     target_link_libraries(dcm_covid PUBLIC OpenMP::OpenMP_CXX)
# endif(OpenMP_FOUND)

# set_target_properties(dcm_covid PROPERTIES COMPILE_FLAGS "")

# add_executable(dcm_3body ${SOURCES_3BODY})

# if(OpenMP_FOUND)
#     target_link_libraries(dcm_3body PUBLIC OpenMP::OpenMP_CXX)
# endif(OpenMP_FOUND)

# set_target_properties(dcm_3body PROPERTIES COMPILE_FLAGS "")

add_executable(run_tests ${SOURCES_TESTS})

if(OpenMP_FOUND)
    target_link_libraries(run_tests PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

target_link_libraries(run_tests PUBLIC gtest_main)
set_target_properties(run_tests PROPERTIES COMPILE_FLAGS "")

add_executable(run_tests_long ${SOURCES_TESTS_LONG})

if(OpenMP_FOUND)
    target_link_libraries(run_tests_long PUBLIC OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

target_link_libraries(run_tests_long PUBLIC gtest_main)
set_target_properties(run_tests_long PROPERTIES COMPILE_FLAGS "")

# add_executable(run_serialization_tests ${SOURCES_SERIALIZATION_TESTS})

# if(OpenMP_FOUND)
#     target_link_libraries(run_serialization_tests PUBLIC OpenMP::OpenMP_CXX)
# endif(OpenMP_FOUND)

# target_link_libraries(run_serialization_tests PUBLIC gtest_main)
# set_target_properties(run_serialization_tests PROPERTIES COMPILE_FLAGS "")