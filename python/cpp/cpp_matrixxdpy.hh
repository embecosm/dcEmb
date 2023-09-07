#ifndef MATRIXXdPY_H
#define MATRIXXdPY_H
#include <Eigen/Dense>
using namespace Eigen;
#pragma once

 
class MatrixXdPy : public MatrixXd { 
    public: 
        MatrixXdPy() : MatrixXd() { }
        MatrixXdPy(int rows,int cols) : MatrixXd(rows,cols) { }
        MatrixXdPy(const MatrixXd& other) : MatrixXd(other) { } 
};
#endif