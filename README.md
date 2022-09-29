# dcEmb
An open source implementation of the Dynamic Causal Modelling framework in C++
by Embecosm. 

# Minimum Requirements
- Cmake
- C++ 14 compatible compiler

# Build Instructions

## Ubuntu
1. Clone the repository

2. Initialise git submodules (git submodule update --init)

3. Create and navigate into a build directory (mkdir build && mv build)

4. Run CMake (cmake .. && cmake --build)

5. Run the executable of your choice (e.g, ./dcm_3body)

## Windows
WIP

## Mac
WIP

# Manual and Documentation
Documentation is provided with doxygen and sphinx. Online versions of
documentation can be found [here](https://embecosm.github.io/dcEmb_docs/),
though is currently not as nicely rendered. To build the Doxygen 
documentation in ubuntu, run (from the root directory):

    doxygen /doc/Doxyfile.in

To build the Sphinx manual, run (from the root directory):

    sphinx-build -b html man/source man/build

# Changelog
A log of changes made to the code can be found in markdown format in the NEWS
file.

# Licensing
This code is available under a GPL 3.0 license. The code makes use of third
party library code that is licensed under MPL 2.0 and BSD 3-clause licenses. 



