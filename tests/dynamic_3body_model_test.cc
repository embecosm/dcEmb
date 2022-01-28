/**
 * Tests of 3-body functions for the dcEmb package
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

#include "3body/dynamic_3body_model.hh"
#include <gtest/gtest.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include "tests/dynamic_3body_model_test.hh"

TEST(dynamic_3body_model_test, system) {
  dynamic_3body_model model;
  model.prior_parameter_expectations = default_prior_expectations();
  model.prior_parameter_covariances = default_prior_covariances();
  model.prior_hyper_expectations = default_hyper_expectations();
  model.prior_hyper_covariances = default_hyper_covariances();
  model.parameter_locations = default_parameter_locations();
  model.num_samples = 1000;
  model.num_response_vars = 3;
  feature_selection_3body fs;
  model.fs_function = fs;
  generative_3body gen;
  model.gen_function = gen;
  gen.eval_generative(true_prior_expectations(), model.parameter_locations,
                      model.num_samples);
  Eigen::MatrixXd true_output = gen.get_output();
  Eigen::MatrixXd response_vars =
      Eigen::MatrixXd::Zero(model.num_samples, model.num_response_vars);
  Eigen::VectorXi select_response_vars =
      Eigen::VectorXi::Zero(model.num_response_vars);
  select_response_vars << 1, 2, 3;
  response_vars = gen.get_output_column(select_response_vars);
  model.select_response_vars = select_response_vars;
  model.response_vars = response_vars;

  model.invert_model();
  gen.eval_generative(model.conditional_parameter_expectations,
                      model.parameter_locations, model.num_samples);
  Eigen::MatrixXd deriv_output = gen.get_output();

  Eigen::MatrixXd diff_mat = true_output - deriv_output;
  // Accept test if mean absolute error is < 1.5%
  EXPECT_TRUE((diff_mat.array().abs().sum() /
               (diff_mat.rows() * diff_mat.cols())) < 0.015

  );
}

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

Eigen::VectorXd default_prior_expectations() {
  Eigen::MatrixXd default_prior_expectation = Eigen::MatrixXd::Zero(7, 3);
  default_prior_expectation.row(0) << 0.95, 1.05, 1.05;
  default_prior_expectation.row(1) << 0.97000436 + 0.05, -0.97000436 - 0.05, 0;
  default_prior_expectation.row(2) << -0.24308753 + 0.05, 0.24308753 + 0.05, 0;
  default_prior_expectation.row(3) << 0, 0, 0;
  default_prior_expectation.row(4) << 0.93240737 / 2 + 0.05,
      0.93240737 / 2 - 0.05, -0.93240737 + 0.05;
  default_prior_expectation.row(5) << 0.86473146 / 2 + 0.05,
      0.86473146 / 2 - 0.05, -0.86473146 - 0.05;
  default_prior_expectation.row(6) << 0, 0, 0;
  Eigen::Map<Eigen::VectorXd> return_default_prior_expectation(
      default_prior_expectation.data(),
      default_prior_expectation.rows() * default_prior_expectation.cols());
  return return_default_prior_expectation;
}

Eigen::MatrixXd default_prior_covariances() {
  double flat = 1.0;                    // flat priors
  double informative = 1 / (double)16;  // informative priors
  double precise = 1 / (double)256;     // precise priors
  double fixed = 1 / (double)2048;      // precise priors
  Eigen::MatrixXd default_prior_covariance = Eigen::MatrixXd::Zero(7, 3);
  default_prior_covariance.row(0) = Eigen::VectorXd::Constant(3, precise);
  default_prior_covariance.row(1) = Eigen::VectorXd::Constant(3, precise);
  default_prior_covariance.row(2) = Eigen::VectorXd::Constant(3, precise);
  default_prior_covariance.row(3) = Eigen::VectorXd::Constant(3, fixed);
  default_prior_covariance.row(4) = Eigen::VectorXd::Constant(3, precise);
  default_prior_covariance.row(5) = Eigen::VectorXd::Constant(3, precise);
  default_prior_covariance.row(6) = Eigen::VectorXd::Constant(3, fixed);
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

Eigen::VectorXd default_hyper_expectations() {
  Eigen::VectorXd default_hyper_expectation = Eigen::VectorXd::Zero(3);
  return default_hyper_expectation;
}
Eigen::MatrixXd default_hyper_covariances() {
  Eigen::MatrixXd default_hyper_covariance = Eigen::MatrixXd::Zero(3, 3);
  default_hyper_covariance.diagonal() << 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0;
  return default_hyper_covariance;
}