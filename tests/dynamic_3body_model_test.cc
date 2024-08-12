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

#include "dynamic_3body_model.hh"
#include <gtest/gtest.h>
#include <stdio.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include "bmr_model.hh"
#include "peb_model.hh"
#include "tests/dynamic_3body_model_test.hh"

/**
 * An simple test of model inversion. Based on a stable "figure of 8" orbit
 * of three planetary bodies.
 *
 * Tests that based on priors different from the true stable orbit, and the
 * position, mass and velocity of one of the planets, posteriors closely
 * reflecting the true stable orbit for the other two planets can be recovered.
 */
TEST(dynamic_3body_model_test, system) {
  dynamic_3body_model model;
  model.prior_parameter_expectations = default_prior_expectations();
  model.prior_parameter_covariances = default_prior_covariances();
  model.prior_hyper_expectations = default_hyper_expectations();
  model.prior_hyper_covariances = default_hyper_covariances();
  model.parameter_locations = default_parameter_locations();
  model.num_samples = 1000;
  model.num_response_vars = 3;

  Eigen::MatrixXd true_output =
      model.eval_generative(true_prior_expectations(1),
                            model.parameter_locations, model.num_samples, 3, 1);
  Eigen::MatrixXd response_vars =
      Eigen::MatrixXd::Zero(model.num_samples, model.num_response_vars);
  Eigen::VectorXi select_response_vars =
      Eigen::VectorXi::Zero(model.num_response_vars);
  select_response_vars << 1, 2, 3;
  response_vars = true_output(Eigen::all, select_response_vars);
  model.select_response_vars = select_response_vars;
  model.response_vars = response_vars;
  model.num_bodies = 3;

  model.invert_model();
  Eigen::MatrixXd deriv_output =
      model.eval_generative(model.conditional_parameter_expectations,
                            model.parameter_locations, model.num_samples, 3, 1);

  Eigen::MatrixXd diff_mat = true_output - deriv_output;
  // Accept test if mean absolute error is < 2%
  EXPECT_LT(
      (diff_mat.array().abs().sum() / (diff_mat.rows() * diff_mat.cols())),
      0.02);
}

/**
 * An simple test of BMR. Based on a stable "figure of 8" orbit
 * of three planetary bodies.
 *
 * The stable orbit in question occurs in a 2d plane, but we have parameterized
 * it in 3 dimensions. This test tests that the values of the spurious third
 * dimension, plus two other parameters that are uniformly zero, are correctly
 * identified and "removed" from the model by the BMR process
 * (by having priors set to zero).
 */
TEST(dynamic_3body_model_bmr_test, system) {
  dynamic_3body_model model;
  model.prior_parameter_expectations = default_prior_expectations();
  model.prior_parameter_covariances = default_prior_covariances();
  model.prior_hyper_expectations = default_hyper_expectations();
  model.prior_hyper_covariances = default_hyper_covariances();
  model.parameter_locations = default_parameter_locations();
  model.num_samples = 1000;
  model.num_response_vars = 3;

  Eigen::VectorXd t_prior = true_prior_expectations(1);
  Eigen::MatrixXd true_output = model.eval_generative(
      t_prior, model.parameter_locations, model.num_samples, 3, 1);
  Eigen::MatrixXd response_vars =
      Eigen::MatrixXd::Zero(model.num_samples, model.num_response_vars);
  Eigen::VectorXi select_response_vars =
      Eigen::VectorXi::Zero(model.num_response_vars);
  select_response_vars << 1, 2, 3;
  response_vars = true_output(Eigen::all, select_response_vars);
  model.select_response_vars = select_response_vars;
  model.response_vars = response_vars;
  model.num_bodies = 3;

  model.invert_model();
  Eigen::MatrixXd deriv_output =
      model.eval_generative(model.conditional_parameter_expectations,
                            model.parameter_locations, model.num_samples, 3, 1);

  bmr_model<dynamic_3body_model> BMR;
  BMR.DCM_in = model;
  BMR.reduce();
  Eigen::MatrixXd bmr_output = model.eval_generative(
      BMR.DCM_out.conditional_parameter_expectations,
      BMR.DCM_out.parameter_locations, BMR.DCM_out.num_samples, 3, 1);
      

  double diff_out = (true_output - deriv_output).array().abs().sum() /
                             (true_output.rows() * true_output.cols());
  double diff_bmr = (true_output - bmr_output).array().abs().sum() /
                             (true_output.rows() * true_output.cols());

  // Variables that are actually zero are correctly selected to be removed
  for (int i = 0; i < t_prior.size(); i++) {
    if (!t_prior(i)) {
      EXPECT_LT(BMR.DCM_out.conditional_parameter_expectations(i), 1e-5);
    }
  }
  // Total fit stays similar
  EXPECT_NEAR(diff_bmr, diff_out, 0.01);
}

/**
 * An simple test of BMR. Based on a stable "figure of 8" orbit
 * of three planetary bodies.
 *
 * In this setup, we have 3 different 3body systems, each of which has the mass
 * of the planets in question peturbed by a small amount.
 *
 * We test by creating a design matrix reflecting the magnitude of these changes
 * in these parameters, and running the PEB process to produce better empirical
 * estimates of these parameters. We verify that the effect of these new
 * empirical priors on the inversion process.
 */
TEST(dynamic_3body_model_peb_test, system) {
  std::vector<dynamic_3body_model> GCM(3);
  std::vector<double> diff(3);
  for (int i = 0; i < 3; i++) {
    dynamic_3body_model& model = GCM[i];
    model.prior_parameter_expectations = default_prior_expectations();
    model.prior_parameter_covariances = default_prior_covariances();
    model.prior_hyper_expectations = default_hyper_expectations();
    model.prior_hyper_covariances = default_hyper_covariances();
    model.parameter_locations = default_parameter_locations();
    model.num_samples = 1000;
    model.num_response_vars = 3;

    Eigen::VectorXd t_prior = true_prior_expectations(0.95 + i * 0.05);
    Eigen::MatrixXd true_output = model.eval_generative(
        t_prior, model.parameter_locations, model.num_samples, 3, 1);
    Eigen::MatrixXd response_vars =
        Eigen::MatrixXd::Zero(model.num_samples, model.num_response_vars);
    Eigen::VectorXi select_response_vars =
        Eigen::VectorXi::Zero(model.num_response_vars);
    select_response_vars << 1, 2, 3;
    response_vars = true_output(Eigen::all, select_response_vars);
    model.select_response_vars = select_response_vars;
    model.response_vars = response_vars;
    model.num_bodies = 3;

    model.invert_model();
    Eigen::MatrixXd org_output = model.eval_generative(
        model.prior_parameter_expectations, model.parameter_locations,
        model.num_samples, 3, 1);
    Eigen::MatrixXd deriv_output = model.eval_generative(
        model.conditional_parameter_expectations, model.parameter_locations,
        model.num_samples, 3, 1);
    Eigen::MatrixXd diff_mat = true_output - deriv_output;
    diff[i] =
        (diff_mat.array().abs().sum() / (diff_mat.rows() * diff_mat.cols()));
  }

  Eigen::VectorXd ones = Eigen::VectorXd::Ones(3);
  Eigen::VectorXd size = Eigen::VectorXd::Zero(3);
  for (int i = 0; i < 3; i++) {
    size(i) = 0.95 + i * 0.05;
  }
  Eigen::MatrixXd design_matrix_tmp = Eigen::MatrixXd(3, 2);
  design_matrix_tmp.col(0) = ones;
  design_matrix_tmp.col(1) = size;

  peb_model<dynamic_3body_model> PEB;
  // Current assumption: All models are using the same parameters in the same
  // Locations
  PEB.random_effects = default_random_effects();
  PEB.GCM = GCM;
  PEB.between_design_matrix = design_matrix_tmp;
  PEB.max_invert_it = 64;
  PEB.invert_model();

  std::vector<dynamic_3body_model> GCM_peb(3);
  std::vector<double> diff_peb(3);
  for (int i = 0; i < 3; i++) {
    dynamic_3body_model& model = GCM_peb[i];
    model.prior_parameter_expectations =
        PEB.empirical_GCM[i].prior_parameter_expectations;
    model.prior_parameter_covariances =
        PEB.empirical_GCM[i].prior_parameter_covariances;
    model.prior_hyper_expectations =
        PEB.empirical_GCM[i].prior_hyper_expectations;
    model.prior_hyper_covariances =
        PEB.empirical_GCM[i].prior_hyper_covariances;
    ;
    model.parameter_locations = default_parameter_locations();
    model.num_samples = 1000;
    model.num_response_vars = 3;

    Eigen::VectorXd t_prior = true_prior_expectations(0.95 + i * 0.05);
    Eigen::MatrixXd true_output = model.eval_generative(
        t_prior, model.parameter_locations, model.num_samples, 3, 1);
    Eigen::MatrixXd response_vars =
        Eigen::MatrixXd::Zero(model.num_samples, model.num_response_vars);
    Eigen::VectorXi select_response_vars =
        Eigen::VectorXi::Zero(model.num_response_vars);
    select_response_vars << 1, 2, 3;
    response_vars = true_output(Eigen::all, select_response_vars);
    model.select_response_vars = select_response_vars;
    model.response_vars = response_vars;
    model.num_bodies = 3;

    model.invert_model();
    Eigen::MatrixXd org_output = model.eval_generative(
        model.prior_parameter_expectations, model.parameter_locations,
        model.num_samples, 3, 1);
    Eigen::MatrixXd deriv_output = model.eval_generative(
        model.conditional_parameter_expectations, model.parameter_locations,
        model.num_samples, 3, 1);
    Eigen::MatrixXd diff_mat = true_output - deriv_output;
    diff_peb[i] =
        (diff_mat.array().abs().sum() / (diff_mat.rows() * diff_mat.cols()));
  }

  EXPECT_NEAR(diff[0], diff_peb[0],
              0.01);  // Best fit to slightly less good fit
  EXPECT_NEAR(diff_peb[0], diff[0], 0.05);  // Middle fit slightly improved
  EXPECT_LT(diff_peb[0] - diff[0], 0.2);    // Bad fit substantially better
}

Eigen::VectorXd true_prior_expectations(const double& x) {
  Eigen::MatrixXd default_prior_expectation = Eigen::MatrixXd::Zero(7, 3);
  default_prior_expectation.row(0) << 1 * x, 1 * x, 1 * x;
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

Eigen::VectorXd default_hyper_expectations() {
  Eigen::VectorXd default_hyper_expectation = Eigen::VectorXd::Zero(3);
  return default_hyper_expectation;
}
Eigen::MatrixXd default_hyper_covariances() {
  Eigen::MatrixXd default_hyper_covariance = Eigen::MatrixXd::Zero(3, 3);
  default_hyper_covariance.diagonal() << 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0;
  return default_hyper_covariance;
}

Eigen::VectorXi default_random_effects() {
  Eigen::VectorXi re = Eigen::VectorXi(3);
  re << 0, 7, 14;
  return re;
}