# 
# Copyright (C) 2022 Embecosm Limited
# 
# Contributor William Jones <william.jones@embecosm.com>
# Contributor Elliot Stein <E.Stein@soton.ac.uk>
# 
# SPDX-License-Identifier: GPL-3.0-or-later 
# 

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(dcm_demo LANGUAGES CXX)

add_subdirectory(${CMAKE_SOURCE_DIR}/lib/googletest)
add_subdirectory(${CMAKE_SOURCE_DIR}/lib/eigen)
enable_testing()

include_directories(  
    ${CMAKE_SOURCE_DIR}/lib/cereal/include
    ${CMAKE_SOURCE_DIR}/lib/eigen
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/COVID
    ${CMAKE_SOURCE_DIR}/include/3body
    ${CMAKE_SOURCE_DIR}/include/music
    ${CMAKE_SOURCE_DIR}/include/tests
    ${gtest_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}
)

set(SOURCES_COVID
    src/dynamic_model.cc
    src/generative_function.cc
    src/feature_selection_function.cc
    src/COVID/run_COVID_dcm.cc
    src/COVID/DEM_COVID.cc
    src/COVID/import_COVID.cc
    src/COVID/dynamic_COVID_model.cc
    src/COVID/generative_COVID.cc
    src/COVID/feature_selection_COVID.cc
    src/utility.cc
)

set(SOURCES_3BODY
    src/dynamic_model.cc
    src/generative_function.cc
    src/feature_selection_function.cc
    src/3body/run_3body_dcm.cc
    src/3body/DEM_3body.cc
    src/3body/dynamic_3body_model.cc
    src/3body/generative_3body.cc
    src/3body/feature_selection_3body.cc
    src/utility.cc
)

set(SOURCES_MUSIC
    src/dynamic_model.cc
    src/generative_function.cc
    src/music/run_music_dcm.cc
    src/music/DEM_music.cc
    src/music/dynamic_music_model.cc
    src/music/generative_music.cc
    src/utility.cc
)

set(SOURCES_TESTS
    src/COVID/import_COVID.cc
    tests/import_COVID_test.cc
    src/feature_selection_function.cc
    src/COVID/feature_selection_COVID.cc
    tests/feature_selection_COVID_test.cc
    src/generative_function.cc
    src/COVID/generative_COVID.cc
    tests/generative_COVID_test.cc
    src/utility.cc
    tests/utility_test.cc
    src/dynamic_model.cc
    src/generative_function.cc
    src/feature_selection_function.cc
    src/3body/dynamic_3body_model.cc
    src/3body/generative_3body.cc
    src/3body/feature_selection_3body.cc
    tests/dynamic_3body_model_test.cc
    
)

set(SOURCES_SERIALIZATION_TESTS
    src/dynamic_model.cc
    src/generative_function.cc
    src/feature_selection_function.cc
    src/3body/dynamic_3body_model.cc
    src/3body/generative_3body.cc
    src/3body/feature_selection_3body.cc
    src/COVID/import_COVID.cc
    src/COVID/dynamic_COVID_model.cc
    src/COVID/generative_COVID.cc
    src/COVID/feature_selection_COVID.cc    
    src/utility.cc
    tests/serialization_test.cc
)

set(SOURCES_TESTS_ELLIOT
    src/COVID/import_COVID.cc
    src/feature_selection_function.cc
    src/COVID/feature_selection_COVID.cc
    src/generative_function.cc
    src/COVID/generative_COVID.cc
    tests/generative_COVID_test.cc
    src/utility.cc
    src/dynamic_model.cc
    src/generative_function.cc
    src/log_likelihood_function.cc
    src/feature_selection_function.cc
    src/3body/dynamic_3body_model.cc
    src/3body/generative_3body.cc
    src/3body/log_likelihood_3body.cc
    src/3body/feature_selection_3body.cc
)

find_package(OpenMP)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}-O3")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}-O3")

add_executable(dcm_covid ${SOURCES_COVID})
target_link_libraries(dcm_covid PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(dcm_covid PROPERTIES LDFLAGS "-fopenmp -march=native -O3")
set_target_properties(dcm_covid PROPERTIES COMPILE_FLAGS "-fopenmp -lpthread -march=native -O3")

add_executable(dcm_3body ${SOURCES_3BODY})
target_link_libraries(dcm_3body PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(dcm_3body PROPERTIES COMPILE_FLAGS "-g -fopenmp -lpthread -march=native -O3")

add_executable(run_tests ${SOURCES_TESTS})
target_link_libraries(run_tests PUBLIC gtest_main PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(run_tests PROPERTIES COMPILE_FLAGS "-g -fopenmp -lpthread -march=native -O3")

add_executable(run_serialization_tests ${SOURCES_SERIALIZATION_TESTS})
target_link_libraries(run_serialization_tests PUBLIC gtest_main PUBLIC OpenMP::OpenMP_CXX PUBLIC)
set_target_properties(run_serialization_tests PROPERTIES COMPILE_FLAGS "-g -fopenmp -lpthread -march=native -O3")