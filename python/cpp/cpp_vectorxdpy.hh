#ifndef VECTORXdPY_H
#define VECTORXdPY_H
#include <eigen3/Eigen/Dense>
#include <iostream>

using namespace Eigen;
#pragma once

 
class VectorXdPy : public VectorXd { 
    public: 
        VectorXdPy() : VectorXd() { }
        VectorXdPy(int size) : VectorXd(size) {  }
        VectorXdPy(VectorXd other) : VectorXd(other) { };
};
#endif