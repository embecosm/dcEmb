.. _install:

Installation
============

The software has been designed to have no requirements to build beyond a modern 
C++ compiler and (currently) CMake, and should be fairly straightforward to 
build on any modern platform.

Dependencies
############
- Cmake
- C++ 14 compatible compiler
- Git

The following packages:
- Cython3
- Eigen3 library
- GTest library
- Numpy
- Sphinx
- Sphinx breathe (Doxygen integration)

Platforms
###########

Linux
++++++

1. Clone the repository

2. Initialise git submodules (git submodule update \-\-init)

3. Create and navigate into a build directory (mkdir build && cd build)

4. Run CMake (cmake .. && cmake \-\-build .)

5. Run the executable of your choice (e.g, ./dcm_3body)

Mac
+++
1. Clone the repository

2. Initialise git submodules (git submodule update \-\-init)

3. Create and navigate into a build directory (mkdir build && cd build)

4. Run CMake (cmake .. && cmake \-\-build .)

5. Run the executable of your choice (e.g, ./dcm_3body)

Windows
+++++++
WIP

