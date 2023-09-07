#ifndef VECTORXiPY_H
#define VECTORXiPY_H
#include <Eigen/Dense>
using namespace Eigen;
#pragma once

 
class VectorXiPy : public VectorXi { 
    public: 
        VectorXiPy() : VectorXi() { }
        VectorXiPy(int size) : VectorXi(size) { }
        VectorXiPy(VectorXi other) : VectorXi(other) { } 
};
#endif