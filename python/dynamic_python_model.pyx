# distutils: language = c++

"""@package docstring
Cython class that wraps the C++ dynamic_python_model class in a Python
compatible format.

The goal of this is to allow the C++ Dynamic Causal Modeling package dcEmb to be
used from Python, allowing inversion of forward model that are also written in
Python. 

"""

from dynamic_python_model cimport dynamic_python_model
from mv_to_eigen cimport mv_to_vectorxd, mv_to_matrixxd, mv_to_vectorxi
from eigen_to_mv cimport vectorxd_to_mv, matrixxd_to_mv
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from vectorxdpy cimport VectorXdPy
from matrixxdpy cimport MatrixXdPy
from vectorxipy cimport VectorXiPy
from cpython.ref cimport PyObject
from cython.view cimport array as cvarray

cdef class PyDynamicModel:
    """Python wrapper class for the C++ dynamic_python_model class"""
    cdef dynamic_python_model c_dynamic_python_model
    cdef object external_generative_model
    cdef double[::1] prior_parameter_expectations
    cdef double[::,::] prior_parameter_covariances
    cdef double[::1] prior_hyper_expectations
    cdef double[::,::] prior_hyper_covariances
    cdef double[::1] conditional_parameter_expectations
    cdef double[::,::] conditional_parameter_covariances
    cdef double[::1] conditional_hyper_expectations
    cdef int[::1] select_response_vars
    cdef int num_samples
    cdef int num_response_vars
    cdef double[::,::] response_vars


    def invert_model(self):
        """Populate the C++ c_dynamic_python_model class with the respective
        parameters from Python, run model inversion, store the results in the
        respective Python variables.
        """
        self.c_dynamic_python_model.prior_parameter_expectations = \
            mv_to_vectorxd(
                &(self.prior_parameter_expectations[0]),
                len(self.prior_parameter_expectations)
            )
        self.c_dynamic_python_model.prior_parameter_covariances = \
            mv_to_matrixxd(
                &(self.prior_parameter_covariances[0,0]),
                len(self.prior_parameter_expectations),
                len(self.prior_parameter_expectations)
            )
        self.c_dynamic_python_model.prior_hyper_expectations = \
            mv_to_vectorxd(
                &(self.prior_hyper_expectations[0]),
                len(self.prior_hyper_expectations)
            )
        self.c_dynamic_python_model.prior_hyper_covariances = \
            mv_to_matrixxd(
                &(self.prior_hyper_covariances[0,0]),
                len(self.prior_hyper_expectations),
                len(self.prior_hyper_expectations)
            )
        self.c_dynamic_python_model.external_generative_model = \
            <PyObject*> self.external_generative_model
        self.c_dynamic_python_model.select_response_vars = \
            mv_to_vectorxi(
                &(self.select_response_vars[0]),
                len(self.select_response_vars)
            )
        self.c_dynamic_python_model.num_samples = \
            self.num_samples
        self.c_dynamic_python_model.num_response_vars = \
            self.num_response_vars
        self.c_dynamic_python_model.response_vars = \
            mv_to_matrixxd(
                &(self.response_vars[0,0]),
                (self.response_vars).shape[0],
                (self.response_vars).shape[1]
            )
        self.c_dynamic_python_model.invert_model()
        self.conditional_parameter_expectations = \
            vectorxd_to_mv(
                self.c_dynamic_python_model.conditional_parameter_expectations
            )
        self.conditional_parameter_covariances = \
            matrixxd_to_mv(
                self.c_dynamic_python_model.conditional_parameter_covariances
            )
        self.conditional_hyper_expectations = \
            vectorxd_to_mv(
                self.c_dynamic_python_model.conditional_hyper_expectations
            )


    def forward_model(self, double[::1] parameters, const int timeseries_length,    
        int[::1] select_response_vars, object func
    ):
        """Run the forward model given by the parameter func as the forward model 
        and return the result.

        It's unlikely you'll need to run this in isolation, but it provides a
        useful tool for debugging problems with model inversion.
        """
        cdef PyObject* func_obj = <PyObject*> func
        cdef double* output_ptr = <double *> malloc(
        timeseries_length * len(select_response_vars) * sizeof(double))
        (
            self.c_dynamic_python_model.forward_model_fun(    
                mv_to_vectorxd(&parameters[0], len(parameters)),
                timeseries_length,
                mv_to_vectorxi(
                    &select_response_vars[0],
                    len(select_response_vars)
                ),
                func_obj,
                output_ptr
            )
        )
        return <double[:timeseries_length*len(select_response_vars):1]>(output_ptr)


    def forward_model(self, double[::1] parameters, const int timeseries_length, 
        int[::1] select_response_vars
    ):
        """Run the forward model given by the self.external_generative_model

        It's unlikely you'll need to run this in isolation, but it provides a
        useful tool for debugging problems with model inversion.
        """
        return self.forward_model(
            self,
            parameters,
            timeseries_length,
            select_response_vars,
            self.external_generative_model
        )
    
    

    @property
    def external_generative_model(self):
        return self.external_generative_model
    
    @external_generative_model.setter
    def external_generative_model(self, value):
        self.external_generative_model = value

    @property
    def prior_parameter_expectations(self):
        return self.prior_parameter_expectations
    
    @prior_parameter_expectations.setter
    def prior_parameter_expectations(self, value):
        self.prior_parameter_expectations = value

    @property
    def prior_parameter_covariances(self):
        return self.prior_parameter_covariances
    
    @prior_parameter_covariances.setter
    def prior_parameter_covariances(self, value):
        self.prior_parameter_covariances = value

    @property
    def prior_hyper_expectations(self):
        return self.prior_hyper_expectations
    
    @prior_hyper_expectations.setter
    def prior_hyper_expectations(self, value):
        self.prior_hyper_expectations = value

    @property
    def prior_hyper_covariances(self):
        return self.prior_hyper_covariances
    
    @prior_hyper_covariances.setter
    def prior_hyper_covariances(self, value):
        self.prior_hyper_covariances = value

    @property
    def conditional_parameter_expectations(self):
        return self.conditional_parameter_expectations
    
    @conditional_parameter_expectations.setter
    def conditional_parameter_expectations(self, value):
        self.conditional_parameter_expectations = value

    @property
    def conditional_parameter_covariances(self):
        return self.conditional_parameter_covariances
    
    @conditional_parameter_covariances.setter
    def conditional_parameter_covariances(self, value):
        self.conditional_parameter_covariances = value

    @property
    def conditional_hyper_expectations(self):
        return self.conditional_hyper_expectations
    
    @conditional_hyper_expectations.setter
    def conditional_hyper_expectations(self, value):
        self.conditional_hyper_expectations = value

    @property
    def select_response_vars(self):
        return self.select_response_vars
    
    @select_response_vars.setter
    def select_response_vars(self, value):
        self.select_response_vars = value

    @property
    def num_samples(self):
        return self.num_samples
    
    @num_samples.setter
    def num_samples(self, value):
        self.num_samples = value

    @property
    def num_response_vars(self):
        return self.num_response_vars
    
    @num_response_vars.setter
    def num_response_vars(self, value):
        self.num_response_vars = value

    @property
    def response_vars(self):
        return self.response_vars
    
    @response_vars.setter
    def response_vars(self, value):
        self.response_vars = value

