/**
 * The 3-body dynamic causal model class within the dcEmb package
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

#include "DEM_weather.hh"
#include <stdio.h>
#include <iostream>
#include <list>
#include <vector>
#include "Eigen/Dense"
#include "bmr_model.hh"
#include "country_data.hh"
#include "dynamic_weather_model.hh"
#include "species_struct.hh"
#include "utility.hh"

/**
 * Run the weather example
 */
int run_weather_test() {
  dynamic_weather_model model;

  return 0;
}

/**
 * "True" values that generate a stable system
 */
Eigen::VectorXd true_prior_expectations() {
  Eigen::MatrixXd default_prior_expectation = Eigen::MatrixXd::Zero(7, 3);
  default_prior_expectation.row(0) << 1, 1, 1;
  default_prior_expectation.row(1) << 0.97000436, -0.97000436, 0;
  default_prior_expectation.row(2) << -0.24308753, 0.24308753, 0;
  default_prior_expectation.row(3) << 0, 0, 0;
  default_prior_expectation.row(4) << 0.93240737 / 2, 0.93240737 / 2,
      -0.93240737;
  default_prior_expectation.row(5) << 0.86473146 / 2, 0.86473146 / 2,
      -0.86473146;
  default_prior_expectation.row(6) << 0, 0, 0;
  Eigen::Map<Eigen::VectorXd> return_default_prior_expectation(
      default_prior_expectation.data(),
      default_prior_expectation.rows() * default_prior_expectation.cols());
  return return_default_prior_expectation;
}

/**
 * Prior expectations on position
 */
Eigen::VectorXd default_prior_expectations() {
  Eigen::MatrixXd default_prior_expectation = Eigen::MatrixXd::Zero(7, 3);
  double x = 0.04;
  default_prior_expectation.row(0) << 1 - x, 1 + x, 1 + x;
  default_prior_expectation.row(1) << 0.97000436 + x, -0.97000436 - x, 0 + x;
  default_prior_expectation.row(2) << -0.24308753 + x, 0.24308753 + x, 0 - x;
  default_prior_expectation.row(3) << 0 + x, 0 + x, 0 - x;
  default_prior_expectation.row(4) << 0.93240737 / 2 + x, 0.93240737 / 2 - x,
      -0.93240737 + x;
  default_prior_expectation.row(5) << 0.86473146 / 2 + x, 0.86473146 / 2 - x,
      -0.86473146 - x;
  default_prior_expectation.row(6) << 0 + x, 0 - x, 0 + x;
  Eigen::Map<Eigen::VectorXd> return_default_prior_expectation(
      default_prior_expectation.data(),
      default_prior_expectation.rows() * default_prior_expectation.cols());
  return return_default_prior_expectation;
}

/**
 * Prior covariance matrix
 */
Eigen::MatrixXd default_prior_covariances() {
  double flat = 1.0;                    // flat priors
  double informative = 1 / (double)16;  // informative priors
  double precise = 1 / (double)256;     // precise priors
  double fixed = 1 / (double)2048;      // fixed priors
  Eigen::MatrixXd default_prior_covariance = Eigen::MatrixXd::Zero(7, 3);
  default_prior_covariance.row(0) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(1) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(2) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(3) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(4) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(5) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(6) = Eigen::VectorXd::Constant(3, informative);
  Eigen::Map<Eigen::VectorXd> default_prior_covariance_diag(
      default_prior_covariance.data(),
      default_prior_covariance.rows() * default_prior_covariance.cols());
  Eigen::MatrixXd return_default_prior_covariance =
      Eigen::MatrixXd::Zero(21, 21);
  return_default_prior_covariance.diagonal() = default_prior_covariance_diag;
  return return_default_prior_covariance;
}

parameter_location_weather default_parameter_locations() {
  parameter_location_weather parameter_locations;
  return parameter_locations;
}

/**
 * Prior hyperparameter expectation vector
 */
Eigen::VectorXd default_hyper_expectations() {
  Eigen::VectorXd default_hyper_expectation = Eigen::VectorXd::Zero(3);
  return default_hyper_expectation;
}
/**
 * Prior hyperparameter covariance matrix
 */
Eigen::MatrixXd default_hyper_covariances() {
  Eigen::MatrixXd default_hyper_covariance = Eigen::MatrixXd::Zero(3, 3);
  default_hyper_covariance.diagonal() << 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0;
  return default_hyper_covariance;
}
