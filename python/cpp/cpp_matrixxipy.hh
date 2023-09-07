#ifndef MATRIXXiPY_H
#define MATRIXXiPY_H
#include <Eigen/Dense>
using namespace Eigen;
#pragma once

 
class MatrixXiPy : public MatrixXi { 
    public: 
        MatrixXiPy() : MatrixXi() { }
        MatrixXiPy(int rows,int cols) : MatrixXi(rows,cols) { }
        MatrixXiPy(const MatrixXi& other) : MatrixXi(other) { } 
};
#endif