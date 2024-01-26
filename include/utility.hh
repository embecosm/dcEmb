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

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#define SparseMD Eigen::SparseMatrix<double>
#define SparseVD Eigen::SparseVector<double>
#pragma once

namespace utility {

using DiagM = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

double logdet(const Eigen::MatrixXd& mat);
double logdet(const DiagM& mat);

Eigen::VectorXd dx(const Eigen::MatrixXd& dfdx, const Eigen::VectorXd& f,
                   const double& t_in);
Eigen::VectorXd rungekutta(std::function<Eigen::VectorXd(Eigen::VectorXd)> func,
                           Eigen::VectorXd& vars, double& h);

SparseMD permute_kron_matrix(const SparseMD& matrix,
                             const Eigen::VectorXi& new_order,
                             const Eigen::VectorXi& cur_order_size);

SparseMD calc_permuted_kron_identity_product(
    const int& id_size, const SparseMD& matrix,
    const Eigen::VectorXi& new_order, const Eigen::VectorXi& cur_order_size);
/*
 * Given a multidimensional matrix represented in block matrix form, give the
 * position in the block matrix from a multidimeional matrix coordinate
 */
inline int find_kron_position(const Eigen::VectorXi& idx1,
                              const Eigen::VectorXi& sizes) {
  Eigen::VectorXi pos_vector = Eigen::VectorXi::Ones(sizes.size());

  int size_iterator = 1;
  // Calculate the values of the vectors converting position to co-ordinates
  for (int i = 1; i < sizes.size(); i++) {
    pos_vector(i) = sizes(i - 1) * size_iterator;
    size_iterator = pos_vector(i);
  }
  return idx1.transpose() * pos_vector;
}
Eigen::VectorXd calculate_marginal_vector(
    const Eigen::VectorXd& ensemble_density, const Eigen::VectorXi& size_order,
    const int& index);
Eigen::MatrixXd calculate_marginal_vector(
    const Eigen::VectorXd& ensemble_density, const Eigen::VectorXi& size_order,
    const int& index1, const int& index2);

Eigen::MatrixXd orth(const Eigen::MatrixXd& mat);

std::vector<Eigen::MatrixXd> reduced_log_evidence(
    const Eigen::VectorXd& conditional_parameter_expectations,
    const Eigen::MatrixXd& conditional_parameter_covariances,
    const Eigen::VectorXd& prior_parameter_expectations,
    const Eigen::MatrixXd& prior_parameter_covariances,
    const Eigen::VectorXd& reduced_parameter_expectations,
    const Eigen::MatrixXd& reduced_parameter_covariances);
Eigen::MatrixXd inverse_tol(const Eigen::MatrixXd& mat);
Eigen::MatrixXd inverse_tol(const Eigen::MatrixXd& mat, const double& tol);
DiagM inverse_tol(const DiagM& mat);
DiagM inverse_tol(const DiagM& mat, const double tol);
double phi(const double& x);
double sigma(const double& x, const double& thresh, const double& sens);
double sigma(const double& x, const double& thresh);

/**
 * Returns a softmax of the elements in an Eigen Matrix or Vector
 */
template <typename Derived, typename T = typename Derived::Scalar>
Derived softmax(const Eigen::MatrixBase<Derived>& mat) {
  T max = mat.maxCoeff();
  Derived mat_norm = mat.unaryExpr([max](T x) { return (x - max); });
  T exp_mat_sum = mat_norm.array().exp().sum();
  if (exp_mat_sum == INFINITY) {
    throw std::runtime_error("vector values too large");
    return Derived::Zero(1);
  }
  return mat_norm.unaryExpr(
      [exp_mat_sum](T x) { return (exp(x) / exp_mat_sum); });
}

/**
 * Output an Eigen Matrix or Vector to a CSV file at full precision
 */
template <class derived>
void print_matrix(const std::string& name,
                  const Eigen::MatrixBase<derived>& mat) {
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                         Eigen::DontAlignCols, ",", "\n");
  std::ofstream file(name.c_str());
  file << mat.format(CSVFormat) << '\n';
  file.close();
  return;
}

/**
 * Read a CSV file to an Eigen Matrix. Typename must be specified in
 * function call, e.g.:
 * read_matrix<Eigen::MatrixXd>(mat)
 */
template <typename M>
M read_matrix(const std::string& name) {
  std::string line;
  std::vector<double> values;
  std::ifstream file(name.c_str());
  if (!file.good()) {
    throw std::runtime_error("Error: File not read");
  }
  uint rows = 0;
  while (std::getline(file, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    rows++;
  }

  return Eigen::Map<
      const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, Eigen::RowMajor>>(
      values.data(), rows, values.size() / rows);
}

/**
 * Generate a vector of normally distributed random numbers with mean
 * mu, and covariance matrix with eigenvalues eval, and eigenvectors evec.
 */
template <typename Derived0, typename T0 = typename Derived0::Scalar,
          typename Derived1, typename T1 = typename Derived1::Scalar>
Derived0 normrnd(const Eigen::MatrixBase<Derived0>& mu,
                 const Eigen::MatrixBase<Derived0>& eval,
                 const Eigen::MatrixBase<Derived1>& evec) {
  if ((eval.array() < 0).any()) {
    throw std::runtime_error(
        "error: matrix not"
        "positive semi-definite");
    return Derived0::Zero(1);
  }
  if ((mu.rows() != 1) && (mu.cols() != 1)) {
    throw std::runtime_error("error: mu must be a vector of numbers");
    return Derived0::Zero(1);
  }

  Derived0 rnd_matrix = Derived0(mu.size(), 1);
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<T0> d{(T0)0, (T0)1};
  for (int i = 0; i < mu.size(); i++) {
    rnd_matrix(i) = d(gen);
  }
  Derived0 mat1 = (rnd_matrix.transpose() *
                   eval.asDiagonal().toDenseMatrix().array().sqrt().matrix())
                      .transpose();
  return (evec * mat1) + mu;
}

/**
 * Sample a value from the given multinomial distribution
 */
template <typename Derived, typename T = typename Derived::Scalar>
int selrnd(const Eigen::MatrixBase<Derived>& dist) {
  if ((dist.rows() != 1) && (dist.cols() != 1)) {
    throw std::runtime_error("error: dist must be a vector of numbers");
    return -1;
  }

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_real_distribution<T> d{0, 1};
  T rnd = d(gen);

  T cum_sum = 0;

  for (int i = 1; i < dist.size(); i++) {
    cum_sum += dist(i - 1);
    if (cum_sum > rnd) {
      return i - 1;
    }
  }
  return dist.size() - 1;
}

/**
 * Calculate an estimate of the gradient of the given vector via linear
 * interpolation, according to:
 * y'(i) = 0.5 * (y(i-1)-y(i+1))
 */
template <typename Derived, typename T = typename Derived::Scalar>
Derived gradient(const Derived& vec) {
  if ((vec.rows() != 1) && (vec.cols() != 1)) {
    throw std::runtime_error("error: gradient must take a vector of numbers");
  }
  int sz = vec.size();
  Derived result_vector(sz);
  for (Eigen::Index i = 1; i < sz - 1; i++) {
    result_vector(i) = 0.5 * ((T)vec(i + 1) - (T)vec(i - 1));
  }
  result_vector(0) = vec(1) - vec(0);
  result_vector(sz - 1) = vec(sz - 1) - vec(sz - 2);
  return result_vector;
}

double SparseTrace(const SparseMD& A);
double SparseProductTrace(const SparseMD& A, const SparseMD& B);
void splitstr(std::vector<std::string>& vec, const std::string& str,
              const char& delim);

}  // namespace utility
