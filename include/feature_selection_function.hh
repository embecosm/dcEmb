/**
 * A base class for feature selection functions within the dcEmb package
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
#pragma once

/**
 * Base class for Feature Selection Functions.
 */
class feature_selection_function {
 public:
  /**
   * @brief feature selection function output variable
   */
  Eigen::MatrixXd fs_response_vars;
  virtual void eval_features();
  Eigen::MatrixXd get_fs_response_vars();
  feature_selection_function();
};