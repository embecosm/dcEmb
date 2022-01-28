.. dcEmb documentation master file, created by
   sphinx-quickstart on Wed Sep 28 15:17:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dcEmb's documentation
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Foreword
#########################
This project is a work in progress, and most of the current documentation exists
in Doxygen. Refer to the README to find out how to build/view this
documentation.

Installation Instructions
#########################
Installation should have no requirements beyond cmake and a modern C++ compiler,
and should be fairly straightforward on any modern platform. That said, we have
currently only tested this extensively on Ubuntu.

Dependencies
++++++++++++
- Cmake
- C++ 14 compatible compiler

Ubuntu
++++++

1. Clone the repository

2. Initialise git submodules (git submodule update --init)

3. Create and navigate into a build directory (mkdir build && mv build)

4. Run CMake (cmake .. && cmake --build)

5. Run the executable of your choice (e.g, ./dcm_3body)

Windows
+++++++
WIP

Mac
+++
WIP

Implemented models
##################

The current repository does not yet implement the more complicated generative
models used in the neuroscience domain. Currently, we implement two
examples:

1. dcm_covid, a reimplementation of the Dynamic Causal Modelling for COVID-19 
code with a few optimisations.

2. dcm_3body, a classical physics example that serves as a sanity check for 
our implementation. Uses the DCM framework to recover a known stable
figure-of-8 orbit of 3 planetary bodies.

Implemented tests
#################

We also implement two testing suites (using GTest) that automatically test and 
verify the functionality of our code. These are:

1. ./run_tests, the main test suite that verifies all our core functionality

2. ./run_serialization_tests, an experimental test suite that verifies the
functionality of our serialization code.

Writing your own Dynamic Causal Model: Short Version
####################################################

The current DCM code makes extensive use of templates, and will likely transiton
fully to a header-only templated library in future. The current instructions are
likely to change significantly:

The existing implementation of the variational laplace inversion scheme is in
the dynamic_model class. It is intended that a user who wishes to implement
an application specific DCM will create a class that inherits from this. 

The dynamic_model class containts two virtual functions that must be implemented
in order a) for the inheriting class to be valid and b) for the model inversion
to work correctly. These are:

dynamic_model::forward_model, a function that takes a vector of parameters, and 
returns a timeseries vector of outcomes (the "generative model")

get_forward_model, a function that returns a wrapped function pointer to the
forward model function 

get_observed_outcomes, a function that returns the observed outcomes from the
model. Useful if you wish to implement a transform on the models output.

With these functions implemented:

1. create an instance of your class

2. populate its prior_parameter_expectations, prior_parameter_covariances,

prior_hyper_expectations and prior_hyper_covariances values

3. If needed, populate the num_samples and num_response_vars variables

4. Invert your model using mode.invert()

5. Collect your posterior estimates from conditional_parameter_expectations, 
conditional_parameter_covariances, conditional_hyper_expectations.


Contributing
============
Style Guide
+++++++++++
This project follows Google’s C++ style guide, as specified in the clang-format
file in the root directory.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
