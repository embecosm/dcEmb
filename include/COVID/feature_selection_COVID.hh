/**
 * The COVID-19 dynamic causal model class within the dcEmb package
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
#include <Eigen/Dense>
#include "feature_selection_function.hh"
#pragma once

/**
 * Feature Selection class for the COVID problem.
 */
class feature_selection_COVID : public feature_selection_function {
 private:
 public:
  /**
   * @brief Feature selection output variable
   */
  Eigen::MatrixXd response_vars;
  void eval_features(Eigen::MatrixXd response_vars);
  void eval_features();
  feature_selection_COVID(const Eigen::MatrixXd& response_vars);
  feature_selection_COVID();
};

/**
 * Equality between feature selection COVID functions
 */
inline bool operator==(const feature_selection_COVID& lhs,
                       const feature_selection_COVID& rhs) {
  return lhs.fs_response_vars == rhs.fs_response_vars &
         lhs.response_vars == rhs.response_vars;
}