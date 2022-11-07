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

#include "DEM_3body.hh"
#include <stdio.h>
#include <iostream>
#include <list>
#include <vector>
#include "Eigen/Dense"
#include "country_data.hh"
#include "dynamic_3body_model.hh"
#include "utility.hh"
#include "bmr_model.hh"

/**
 * Run the 3body example
 */
int run_3body_test() {
  dynamic_3body_model model;
  model.prior_parameter_expectations = default_prior_expectations();
  model.prior_parameter_covariances = default_prior_covariances();
  model.prior_hyper_expectations = default_hyper_expectations();
  model.prior_hyper_covariances = default_hyper_covariances();
  model.parameter_locations = default_parameter_locations();
  model.num_samples = 1000;
  model.num_response_vars = 3;

  Eigen::MatrixXd out1 =
      model.eval_generative(true_prior_expectations(),
                            model.parameter_locations, model.num_samples, 3);
  utility::print_matrix("../visualisation/true_generative.csv", out1);
  Eigen::MatrixXd response_vars =
      Eigen::MatrixXd::Zero(model.num_samples, model.num_response_vars);
  Eigen::VectorXi select_response_vars =
      Eigen::VectorXi::Zero(model.num_response_vars);
  select_response_vars << 1, 2, 3;
  response_vars = out1(Eigen::all, select_response_vars);
  model.select_response_vars = select_response_vars;
  model.response_vars = response_vars;
  model.num_bodies = 3;

  model.invert_model();
  Eigen::MatrixXd out2 =
      model.eval_generative(model.conditional_parameter_expectations,
                            model.parameter_locations, model.num_samples, 3);
  utility::print_matrix("../visualisation/deriv_generative.csv", out2);
  Eigen::MatrixXd out3 =
      model.eval_generative(default_prior_expectations(),
                            model.parameter_locations, model.num_samples, 3);
  utility::print_matrix("../visualisation/org_generative.csv", out3);

  bmr_model<dynamic_3body_model> BMR;
  BMR.DCM_in = model;
  BMR.reduce();
  Eigen::MatrixXd out4 =
      model.eval_generative(BMR.DCM_out.conditional_parameter_expectations,
                            BMR.DCM_out.parameter_locations, BMR.DCM_out.num_samples, 3);
  utility::print_matrix("../visualisation/deriv2_generative.csv", out4);
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
  default_prior_expectation.row(0) << 0.95, 1.05, 1.05;
  default_prior_expectation.row(1) << 0.97000436 + 0.05, -0.97000436 - 0.05, 0;
  default_prior_expectation.row(2) << -0.24308753 + 0.05, 0.24308753 + 0.05, 0;
  default_prior_expectation.row(3) << 0.05, 0.05, -0.05;
  default_prior_expectation.row(4) << 0.93240737 / 2 + 0.05,
      0.93240737 / 2 - 0.05, -0.93240737 + 0.05;
  default_prior_expectation.row(5) << 0.86473146 / 2 + 0.05,
      0.86473146 / 2 - 0.05, -0.86473146 - 0.05;
  default_prior_expectation.row(6) << 0.05, -0.05, 0.05;
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
  double fixed = 1 / (double)2048;      // precise priors
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

parameter_location_3body default_parameter_locations() {
  parameter_location_3body parameter_locations;
  parameter_locations.planet_masses = 0;
  parameter_locations.planet_coordsX = 0;
  parameter_locations.planet_coordsY = 1;
  parameter_locations.planet_coordsZ = 2;
  parameter_locations.planet_velocityX = 4;
  parameter_locations.planet_velocityY = 5;
  parameter_locations.planet_velocityZ = 6;
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
