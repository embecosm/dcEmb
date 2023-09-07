#include <Eigen/Dense>
#pragma once


Eigen::VectorXd mv_to_vectorxd(double *mv, const int& len);
Eigen::MatrixXd mv_to_matrixxd(double *mv, const int& len1, const int& len2);
Eigen::VectorXi mv_to_vectorxi(int *mv, const int& len);