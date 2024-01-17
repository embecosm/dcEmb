# dcEmb
An open source implementation of the Dynamic Causal Modelling library in C++
by Embecosm. 

# Minimum Requirements
- Cmake
- C++ 14 compatible compiler
- OpenMP
- Eigen >=3.4

# Documentation
This codebase is documented with Doxygen and Sphinx. Online versions of
documentation can be found [here](https://embecosm.github.io/dcEmb_docs/),
and also come attached to the repository. 

To build sphinx documentation:
1) sphinx-build -b html man/source man/build
2) doxygen doc/Doxyfile.in

# Getting Started
See the online manual [here](https://embecosm.github.io/dcEmb_docs/).

See a set of worked examples [here](https://github.com/embecosm/dcEmb-examples)

The codebase is undergoing a substantial set of changes. Some parts of the
online manual may be out of date

## Python
A recent version update has added a Python integration to dcEmb. This feature is
currently experimental.

# Changelog
A log of changes made to the code can be found in markdown format in the NEWS
file.

# Licensing
This code is available under a GPL 3.0 license. The code makes use of third
party library code that is licensed under MPL 2.0 and BSD 3-clause licenses. 



