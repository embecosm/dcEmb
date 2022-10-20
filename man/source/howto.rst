.. _howto:

Building Your Own model
=======================

Foreword
########
This section contains guidance for writing a Dynamic Causal Model for your 
own application. Be warned that the code is currently under development,
and that these instructions are likely to change significantly as time goes on.

Overview
########

Worked Example: dcm_3body
#########################

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