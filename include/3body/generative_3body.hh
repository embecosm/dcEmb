/**
 * A generative function (forward model) class for the 3-body dynamic causal
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
#include "parameter_location_3body.hh"
#define SparseMD Eigen::SparseMatrix<double>
#define SparseVD Eigen::SparseVector<double>
#pragma once

/**
 * generative model class for the 3body problem.
 */
class generative_3body : public generative_function {
 public:
  /**
   * @brief parameters for generative model
   */
  Eigen::VectorXd parameters;
  /**
   * @brief parameters locations for generative model
   */
  parameter_location_3body parameter_locations;
  /**
   * @brief number of timesteps to generate
   */
  int timeseries_length;
  Eigen::MatrixXd get_joint_density();

  void eval_generative(const Eigen::VectorXd& parameters,
                       const parameter_location_3body& parameter_locations,
                       const int& timeseries_length);
  void eval_generative();
  Eigen::VectorXd differential_eq(const Eigen::VectorXd& state);
  generative_3body(const Eigen::VectorXd& parameters,
                   const parameter_location_3body& parameter_locations,
                   const int& timeseries_length);
  generative_3body();
};

/**
 * Equality between generative 3-body models
 */
inline bool operator==(const generative_3body& lhs,
                       const generative_3body& rhs) {
  return lhs.output == rhs.output & lhs.parameters == rhs.parameters &
         lhs.parameter_locations == rhs.parameter_locations &
         lhs.timeseries_length == rhs.timeseries_length;
}