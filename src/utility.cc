/**
 * A set of utility functions for the dcEmb package
 *
 * Copyright (C) 2022 Embecosm Limited
 *
 * Contributor William Jones <william.jones@embecosm.com>
 * Contributor Elliot Stein <E.Stein@soton.ac.uk>
 *
 * This file is part of the dcEmb package
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "utility.hh"
#include <omp.h>
#include <bitset>
#include <functional>
#include <iostream>
#include <sstream>
#include <unsupported/Eigen/MatrixFunctions>
#include "Eigen/Core"

#define SparseMD Eigen::SparseMatrix<double>
#define SparseVD Eigen::SparseVector<double>
#define SparseMDrow Eigen::SparseMatrix<double, Eigen::RowMajor>
#define SparseMDcol Eigen::SparseMatrix<double, Eigen::ColMajor>
#define DEBUG(x) std::cout << #x << "= " << x << std::endl;

/**
 * Output an Eigen Matrix or Vector to a CSV file at full precision
 */
Eigen::VectorXd utility::dx(const Eigen::MatrixXd& dfdx,
                            const Eigen::VectorXd& f, const double& t_in) {
  int f_len = f.size();
  double t = exp(t_in - logdet(dfdx) / f_len);
  int row_sz = std::max(f.rows(), dfdx.rows()) + 1;
  int col_sz = f.cols() + dfdx.cols();
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(row_sz, col_sz);
  jacobian(Eigen::seq(Eigen::last + 1 - f.rows(), Eigen::last),
           Eigen::seq(0, f.cols() - 1)) = f * t;
  jacobian(Eigen::seq(Eigen::last + 1 - dfdx.rows(), Eigen::last),
           Eigen::seq(Eigen::last + 1 - dfdx.cols(), Eigen::last)) = dfdx * t;
  Eigen::MatrixXd jacobian_exp = jacobian.exp();
  return jacobian_exp(Eigen::seq(1, Eigen::last), 0);
}

/**
 * Calculate a fast approximation to the log-determinant of a positive-semi
 * definite square matrix
 */
double utility::logdet(const Eigen::MatrixXd& mat) {
  double tol = 1e-16;
  if (mat.cols() != mat.rows()) {
    std::cout << "Error: Expected Square Matrix in logdet";
    throw;
  }
  if (mat.isDiagonal()) {
    Eigen::VectorXd eig_values = mat.diagonal();
    Eigen::VectorXd pos_eig_values = eig_values.unaryExpr([tol](double x) {
      return ((x > tol) && (x < 1 / tol)) ? log(x) : 0.0;
    });
    return pos_eig_values.sum();
  } else {
    Eigen::BDCSVD<Eigen::MatrixXd> svd;
    svd.compute(mat, Eigen::HouseholderQRPreconditioner | Eigen::ComputeThinV |
                         Eigen::ComputeThinU);
    Eigen::VectorXd singular_values = svd.singularValues();
    Eigen::VectorXd log_singular_values =
        singular_values.unaryExpr([tol](double x) {
          return ((x > tol) && (x < 1 / tol)) ? log(x) : 0.0;
        });
    return log_singular_values.sum();
  }
}

/**
 * Calculate a runge-kutta step for a given function
 */
Eigen::VectorXd utility::rungekutta(
    std::function<Eigen::VectorXd(Eigen::VectorXd)> func, Eigen::VectorXd& vars,
    double& h) {
  Eigen::VectorXd k1 = func(vars);
  Eigen::VectorXd k2 = func(vars + (h * k1 / 2));
  Eigen::VectorXd k3 = func(vars + (h * k2 / 2));
  Eigen::VectorXd k4 = func(vars + (h * k3));
  return h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

/*
 * Recursive Gram-Schmit orthagonalisation of basis functions.
 */
Eigen::MatrixXd utility::orth(const Eigen::MatrixXd& mat) {
  int num_rows = mat.rows();
  int num_cols = mat.cols();
  Eigen::BDCSVD<Eigen::MatrixXd> svd;
  svd.compute(mat, Eigen::ComputeThinV | Eigen::ComputeThinU);
  int rank = svd.rank();

  Eigen::MatrixXd u = Eigen::MatrixXd::Zero(rank, rank);
  u.col(0) = mat.col(0);
  int j_it = 0;
  for (int i = 1; i < rank; i++) {
    Eigen::MatrixXd pinv = u.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::VectorXd u_n = mat.col(i);
    u_n = u_n - (u * pinv * u_n);
    if (!(u_n.norm() > exp(-32))) {
      throw std::runtime_error(
          "Currently unspecified behaviour in utility::orth - norm out of "
          "range");
      return Eigen::MatrixXd::Zero(1, 1);
    }
    u.col(i) = u_n;
  }
  return u;
}

/*
 * Calculate and return the log-evidence and posteriors of a reduced DCM that
 * is nested within a larger DCM.
 */
std::vector<Eigen::MatrixXd> utility::reduced_log_evidence(
    const Eigen::VectorXd& conditional_parameter_expectations_in,
    const Eigen::MatrixXd& conditional_parameter_covariances_in,
    const Eigen::VectorXd& prior_parameter_expectations_in,
    const Eigen::MatrixXd& prior_parameter_covariances_in,
    const Eigen::VectorXd& reduced_parameter_expectations_in,
    const Eigen::MatrixXd& reduced_parameter_covariances_in) {
  double tol = 1e-8;
  Eigen::VectorXd RE = reduced_parameter_expectations_in;
  Eigen::VectorXd SE = conditional_parameter_expectations_in;

  Eigen::BDCSVD<Eigen::MatrixXd> svd;
  svd.setThreshold(1e-6);
  svd.compute(prior_parameter_covariances_in,
              Eigen::ComputeThinV | Eigen::ComputeFullU);
  Eigen::MatrixXd singular_vec = svd.matrixU();
  Eigen::VectorXd conditional_parameter_expectations =
      singular_vec.transpose() * conditional_parameter_expectations_in;
  Eigen::MatrixXd conditional_parameter_covariances =
      singular_vec.transpose() * conditional_parameter_covariances_in *
      singular_vec;
  Eigen::VectorXd prior_parameter_expectations =
      singular_vec.transpose() * prior_parameter_expectations_in;
  Eigen::MatrixXd prior_parameter_covariances =
      singular_vec.transpose() * prior_parameter_covariances_in * singular_vec;
  Eigen::VectorXd reduced_parameter_expectations =
      singular_vec.transpose() * reduced_parameter_expectations_in;
  Eigen::MatrixXd reduced_parameter_covariances =
      singular_vec.transpose() * reduced_parameter_covariances_in *
      singular_vec;

  Eigen::MatrixXd conditional_parameter_covariances_inv =
      utility::inverse_tol(conditional_parameter_covariances, tol);
  Eigen::MatrixXd prior_parameter_covariances_inv =
      utility::inverse_tol(prior_parameter_covariances, tol);
  Eigen::MatrixXd reduced_parameter_covariances_inv =
      utility::inverse_tol(reduced_parameter_covariances, tol);

  Eigen::MatrixXd out_parameter_covariances_inv =
      conditional_parameter_covariances_inv +
      reduced_parameter_covariances_inv - prior_parameter_covariances_inv;
  Eigen::MatrixXd out_parameter_covariances =
      utility::inverse_tol(out_parameter_covariances_inv, tol);
  prior_parameter_covariances =
      utility::inverse_tol(prior_parameter_covariances_inv, tol);

  Eigen::VectorXd out_parameter_expectations_c =
      (conditional_parameter_covariances_inv *
       conditional_parameter_expectations);
  Eigen::VectorXd out_parameter_expectations_r =
      (reduced_parameter_covariances_inv * reduced_parameter_expectations);
  Eigen::VectorXd out_parameter_expectations_p =
      (prior_parameter_covariances_inv * prior_parameter_expectations);
  Eigen::VectorXd out_parameter_expectations = out_parameter_expectations_c +
                                               out_parameter_expectations_r -
                                               out_parameter_expectations_p;

  Eigen::MatrixXd F1 = Eigen::MatrixXd(1, 1);
  F1 << utility::logdet(reduced_parameter_covariances_inv *
                        conditional_parameter_covariances_inv *
                        out_parameter_covariances *
                        prior_parameter_covariances);

  Eigen::MatrixXd F2_c = (conditional_parameter_expectations.transpose() *
                          conditional_parameter_covariances_inv *
                          conditional_parameter_expectations);
  Eigen::MatrixXd F2_r =
      (reduced_parameter_expectations.transpose() *
       reduced_parameter_covariances_inv * reduced_parameter_expectations);
  Eigen::MatrixXd F2_p =
      (prior_parameter_expectations.transpose() *
       prior_parameter_covariances_inv * prior_parameter_expectations);
  Eigen::MatrixXd F2_o =
      (out_parameter_expectations.transpose() * out_parameter_covariances *
       out_parameter_expectations);
  Eigen::MatrixXd F2 = F2_c + F2_r - F2_p - F2_o;
  Eigen::MatrixXd free_energy = (F1 - F2) / 2;

  Eigen::VectorXd reduced_prior_expectations_tmp =
      out_parameter_covariances * out_parameter_expectations;
  Eigen::VectorXd conditional_reduced_expectations =
      (singular_vec * reduced_prior_expectations_tmp) +
      (reduced_parameter_expectations_in) -
      (singular_vec * singular_vec.transpose() *
       reduced_parameter_expectations_in);
  Eigen::MatrixXd conditional_reduced_covariances =
      singular_vec * out_parameter_covariances * singular_vec.transpose();
  std::vector<Eigen::MatrixXd> return_vector = std::vector<Eigen::MatrixXd>(3);
  return_vector[0] = free_energy;
  return_vector[1] = conditional_reduced_expectations;
  return_vector[2] = conditional_reduced_covariances;
  return return_vector;
}

/*
 * Matrix inverse with tolerance
 */
Eigen::MatrixXd utility::inverse_tol(const Eigen::MatrixXd& mat) {
  double norm = mat.lpNorm<Eigen::Infinity>();
  int max_dim = std::max(mat.rows(), mat.cols());
  double tol = std::max(DBL_EPSILON * norm * max_dim, exp(-32));
  return utility::inverse_tol(mat, tol);
}
Eigen::MatrixXd utility::inverse_tol(const Eigen::MatrixXd& mat,
                                     const double& tol) {
  Eigen::MatrixXd diag = Eigen::MatrixXd::Identity(mat.rows(), mat.cols());
  diag = diag * tol;

  Eigen::MatrixXd ret_mat = (mat + diag);
  if (ret_mat.isDiagonal()) {
    ret_mat.diagonal() =
        ret_mat.diagonal().unaryExpr([](double x) { return (1 / x); });
  } else {
    ret_mat = ret_mat.inverse().eval();
  }
  return ret_mat;
}

/*
 * Calculate the trace for a Sparse matrix
 */
double utility::SparseTrace(const SparseMD& A) {
  double trace = 0;
  // check square
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("Matrix non-square");
  }
  for (int i = 0; i < A.rows(); i++) {
    trace += A.coeff(i, i);
  }
  return trace;
}

/*
 * Calculate the trace for a Sparse matrix product
 */
double utility::SparseProductTrace(const SparseMD& in1, const SparseMD& in2) {
  // Check dimensions
  if (in1.rows() != in1.cols() || in2.rows() != in2.cols() ||
      in1.cols() != in2.rows()) {
    throw std::invalid_argument("Dimension mismatch");
  }

  double trace = 0;
  SparseMDrow A = in1;
  SparseMDcol B = in2;

  for (int i = 0; i < A.rows(); i++) {
    trace += A.row(i).dot(B.col(i));
  }
  return trace;
}

