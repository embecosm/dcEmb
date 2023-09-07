# distutils: language = c++

cdef extern from "cpp_vectorxipy.hh":
    cdef cppclass VectorXiPy:
        VectorXiPy()
        VectorXiPy(int d1)
        VectorXiPy(VectorXiPy other)
        int rows()
        int cols()
        int coeff(int)
        int* data()