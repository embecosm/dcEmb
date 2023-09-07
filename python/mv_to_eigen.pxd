# distutils: language = c++


from vectorxdpy cimport VectorXdPy
from matrixxdpy cimport MatrixXdPy
from vectorxipy cimport VectorXiPy
from cpython.ref cimport PyObject

cdef extern from "cpp/mv_to_eigen.cc":
    pass

# Declare the class with cdef
cdef extern from "cpp/mv_to_eigen.hh":
    const VectorXdPy mv_to_vectorxd(const double *mv, int len);
    const MatrixXdPy mv_to_matrixxd(const double *mv, int len1, int len2);
    const VectorXiPy mv_to_vectorxi(const int *mv, int len);