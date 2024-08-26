#include <eigen3/Eigen/Dense>
#include <Python.h>
#pragma once


PyObject* vectorxd_to_mv(const Eigen::VectorXd& mat);
PyObject* matrixxd_to_mv(const Eigen::MatrixXd& mat);
PyObject* vectorxi_to_mv(const Eigen::VectorXd& mat);
