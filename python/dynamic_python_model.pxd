
# distutils: language = c++

from matrixxdpy cimport MatrixXdPy
from matrixxipy cimport MatrixXiPy
from vectorxdpy cimport VectorXdPy
from vectorxipy cimport VectorXiPy
from libcpp.string cimport string
from cpython.ref cimport PyObject

cdef extern from "cpp/dynamic_python_model.cc":
    pass

# Declare the class with cdef
cdef extern from "cpp/dynamic_python_model.hh":
    cdef cppclass dynamic_python_model:
        int intermediate_outputs_to_file;
        string intermediate_expectations_filename;
        string intermediate_covariances_filename;
        double converge_crit;
        int max_invert_it;
        int performed_it;
        VectorXdPy conditional_parameter_expectations;
        MatrixXdPy conditional_parameter_covariances;
        VectorXdPy conditional_hyper_expectations;
        double free_energy;
        VectorXdPy prior_parameter_expectations;
        MatrixXdPy prior_parameter_covariances;
        VectorXdPy prior_hyper_expectations;
        MatrixXdPy prior_hyper_covariances;
        int num_samples;
        int num_response_vars;
        VectorXiPy select_response_vars;
        MatrixXdPy response_vars;
        VectorXdPy posterior_over_parameters;
        void invert_model();
        VectorXdPy get_observed_outcomes();
        PyObject* external_generative_model;
        void forward_model_fun(
            const VectorXdPy& parameters,
            const int& timeseries_length,
            const VectorXiPy& select_response_vars,
            PyObject* func,
            double* output_ptr);
