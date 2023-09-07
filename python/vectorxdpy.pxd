# distutils: language = c++

cdef extern from "cpp_vectorxdpy.hh":
    cdef cppclass VectorXdPy:
        VectorXdPy()
        VectorXdPy(int d1)
        VectorXdPy(VectorXdPy other)
        int rows()
        int cols()
        int coeff(int)
        double* data()