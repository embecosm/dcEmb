/**
 * A generative function (forward model) class for the COVID-19 dynamic causal
 * model within the dcEmb package
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

#include <gtest/gtest_prod.h>
#include <stdarg.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "generative_function.hh"
#include "parameter_location_COVID.hh"
#define SparseMD Eigen::SparseMatrix<double>
#define SparseVD Eigen::SparseVector<double>
#pragma once

/**
 * Generative class for the 3body problem.
 */
class generative_COVID : public generative_function {
 private:
  SparseMD eval_transition_probability_matrix(
      const Eigen::VectorXd& parameters_exp, const SparseMD& ensemble_density);
  SparseMD calc_location_transition_matrix(const Eigen::VectorXd& parameter_exp,
                                           const SparseMD& ensemble_density);
  SparseMD calc_infection_transition_matrix(
      const Eigen::VectorXd& parameter_exp, const SparseMD& ensemble_density);
  SparseMD calc_clinical_transition_matrix(
      const Eigen::VectorXd& parameter_exp);
  SparseMD calc_testing_transition_matrix(const Eigen::VectorXd& parameter_exp,
                                          const SparseMD& ensemble_density);
  FRIEND_TEST(generative_COVID_test, permute_matrix);
  FRIEND_TEST(generative_COVID_test, find_position);
  FRIEND_TEST(generative_COVID_test, calculate_marginal_matrix_simple);
  FRIEND_TEST(generative_COVID_test, calculate_marginal_matrix_recovery);
  FRIEND_TEST(generative_COVID_test, sigma);
  FRIEND_TEST(generative_COVID_test, phi);

 public:
  /**
   * @brief marginal probabilities at each time step for location
   */
  Eigen::MatrixXd marginal_location;
  /**
   * @brief marginal probabilities at each time step for infection
   */
  Eigen::MatrixXd marginal_infection;
  /**
   * @brief marginal probabilities at each time step for clinical
   */
  Eigen::MatrixXd marginal_clinical;
  /**
   * @brief marginal probabilities at each time step for testing
   */
  Eigen::MatrixXd marginal_testing;
  /**
   * @brief parameters for generative model
   */
  Eigen::VectorXd parameters;
  /**
   * @brief parameter locations for generative model
   */
  parameter_location_COVID parameter_locations;
  /**
   * @brief number of timesteps to generate
   */
  int timeseries_length;

  void eval_generative(const Eigen::VectorXd& parameters,
                       const parameter_location_COVID& parameter_locations,
                       const int& timeseries_length);
  void eval_generative();

  generative_COVID(const Eigen::VectorXd& parameters,
                   const parameter_location_COVID& parameter_locations,
                   const int& timeseries_length);
  generative_COVID();
};

/**
 * Evaluate equality between generative COVID model objects
 */
inline bool operator==(const generative_COVID& lhs,
                       const generative_COVID& rhs) {
  return lhs.output == rhs.output &
         lhs.marginal_location == rhs.marginal_location &
         lhs.marginal_infection == rhs.marginal_infection &
         lhs.marginal_clinical == rhs.marginal_clinical &
         lhs.marginal_testing == rhs.marginal_testing &
         lhs.parameters == rhs.parameters &
         lhs.parameter_locations == rhs.parameter_locations &
         lhs.timeseries_length == rhs.timeseries_length;
}
