# distutils: language = c++

cdef extern from "cpp_matrixxipy.hh":
    cdef cppclass MatrixXiPy:
        MatrixXiPy()
        MatrixXiPy(int d1, int d2)
        MatrixXiPy(MatrixXiPy other)
        int rows()
        int cols()
        int coeff(int, int)