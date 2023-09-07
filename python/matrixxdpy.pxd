# distutils: language = c++

cdef extern from "cpp_matrixxdpy.hh":
    cdef cppclass MatrixXdPy:
        MatrixXdPy()
        MatrixXdPy(int d1, int d2)
        MatrixXdPy(MatrixXdPy other)
        int rows()
        int cols()
        double coeff(int, int)