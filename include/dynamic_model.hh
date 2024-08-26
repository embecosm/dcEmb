/**
 * A base class for dynamic causal models within the dcEmb package
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

#include <eigen3/Eigen/Dense>
#pragma once

/**
 * Base class for all Dynamic Causal Models. Specific implementations of DCM
 * for a given purpose should inherit from this class, and overwrite the
 * get_observed_outcomes function, the get_forward_model function and the
 * forward_model function.
 */
class dynamic_model {
 public:
  /**
   * @brief Store intermediate posterior expectation and covariance values to
   * files.
   */
  int intermediate_outputs_to_file = 0;
  /**
   * @brief filename for storing intermediate posterior expectations
   */
  std::string intermediate_expectations_filename = "param_expecations.csv";
  /**
   * @brief filename for storing intermediate posterior covariances
   */
  std::string intermediate_covariances_filename = "param_covariances.csv";
  double converge_crit = 1e-1;
  /**
   * @brief Maximum number of EM steps attempted in inversion
   */
  int max_invert_it = 128;
  /**
   * @brief The number of EM steps performed by inversion
   */
  int performed_it = 0;
  /**
   * @brief Vector of posterior parameter expectations
   */
  Eigen::VectorXd conditional_parameter_expectations;
  /**
   * @brief Matrix of posterior parameter covariances
   */
  Eigen::MatrixXd conditional_parameter_covariances;
  /**
   * @brief Vector of posterior "hyper"-parameter expectations
   */
  Eigen::VectorXd conditional_hyper_expectations;
  /**
   * @brief Free Energy of final inverted model
   */
  double free_energy;
  /**
   * @brief Vector of prior parameter expectations
   */
  Eigen::VectorXd prior_parameter_expectations;
  /**
   * @brief Matrix of prior parameter expectations
   */
  Eigen::MatrixXd prior_parameter_covariances;
  /**
   * @brief Vector of prior "hyper"-parameter expectations
   */
  Eigen::VectorXd prior_hyper_expectations;
  /**
   * @brief Matrix of prior "hyper"-parameter covariances
   */
  Eigen::MatrixXd prior_hyper_covariances;
  /**
   * @brief Number of samples in each response variable of interest in
   * the generative model
   */
  int num_samples;
  /**
   * @brief Number of response variable of interest in the generative
   * model
   */
  int num_response_vars = 0;
  /**
   * @brief Vector of location of response variables of interest
   */
  Eigen::VectorXi select_response_vars;
  /**
   * @brief True response varibles to fit the generative model to
   */
  Eigen::MatrixXd response_vars;
  /**
   * @brief If inverting a group of DCMs, a posterior over all models
   * parameters
   */
  Eigen::VectorXd posterior_over_parameters;
  /**
   * Invert a Dynamic Causal Model using the variational laplace scheme.
   * At least the following must be defined be given values or
   * implementations by the inheriting class for this to be successful:
   *
   * - forward_model()
   * - get_forward_model_function()
   * - get_observed_outcomes()
   * - prior_parameter_expectations
   * - prior_parameter_covariances
   * - prior_hyper_expectations
   * - prior_hyper_covariances
   * - response_vars
   * - num_response_vars
   * - select_response_vars
   * - num_samples
   *
   * Once inverted, a models pa
   */
  void invert_model();
  /**
   * Getter function for the response variables that is a suitable place
   * to apply a transform
   */
  virtual Eigen::VectorXd get_observed_outcomes();
  /**
   * Wrapper for the forward_model_function. Must create a wrapped
   * fuction that takes a vector and returns a vector.
   */
  virtual std::function<Eigen::VectorXd(const Eigen::VectorXd&)>
  get_forward_model_function();
  /**
   * Forward model function for the Dynamic Causal Model. Takes a vector
   * of parameters and potentially other arguments, and returns time
   * series data in a vector
   */
  Eigen::VectorXd forward_model(const Eigen::VectorXd& parameters);
  /**
   * Empty constructor for the dynamic_model class
   */
  /**
   * Calculate the numerical derivative of a function. Function must be
   * parameterised by a single vector, and return a single vector.
   */

  template <typename Derived0, typename T0 = typename Derived0::Scalar,
            typename Derived1, typename T1 = typename Derived1::Scalar>
  Derived1 diff(
      std::function<Derived0(const Derived0&)>
          func,
      Eigen::MatrixBase<Derived0>& vars, Eigen::MatrixBase<Derived1>& transform) {
    Derived0 base = func(vars);
    Derived1 out_matrix = Derived1::Zero(base.size(), vars.size());
    T0 dx = exp(-8.0);
  #pragma omp parallel
    {
  #pragma omp for schedule(static)
      for (int i = 0; i < vars.size(); i++) {
        Derived0 vars_tmp = vars + (transform.col(i) * dx);
        Derived0 modified = func(vars_tmp);
        out_matrix.col(i) = (modified - base) / dx;
      }
    }
    return out_matrix;
  }

  dynamic_model();
};
