# distutils: language = c++


from vectorxdpy cimport VectorXdPy
from matrixxdpy cimport MatrixXdPy
from vectorxipy cimport VectorXiPy
from cpython.ref cimport PyObject

cdef extern from "cpp/eigen_to_mv.cc":
    pass

# Declare the class with cdef
cdef extern from "cpp/eigen_to_mv.hh":
    object vectorxd_to_mv(VectorXdPy& mat);
    object matrixxd_to_mv(MatrixXdPy& mat);
    object vectorxi_to_mv(VectorXiPy& mat);