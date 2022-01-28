/**
 * A feature selection class for the 3-body dynamic causal model within the
 * dcEmb package
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

#include <stdarg.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include "feature_selection_function.hh"
#pragma once

/**
 * Feature selection class for the 3body problem.
 */
class feature_selection_3body : public feature_selection_function {
 private:
 public:
  /**
   * @brief Feature selection output variable
   */
  Eigen::MatrixXd response_vars;
  void eval_features(Eigen::MatrixXd response_vars);
  void eval_features();
  feature_selection_3body(const Eigen::MatrixXd& response_vars);
  feature_selection_3body();
};

/**
 * Equality between feature selection 3-body functions
 */
inline bool operator==(const feature_selection_3body& lhs,
                       const feature_selection_3body& rhs) {
  return lhs.fs_response_vars == rhs.fs_response_vars &
         lhs.response_vars == rhs.response_vars;
}