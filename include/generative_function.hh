/**
 * A base class for generative (forward model) functions within the dcEmb
 * package
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
 * Base class for Generative Functions.
 */
class generative_function {
 public:
  /**
   * @brief Generative function output variable
   */
  Eigen::MatrixXd output;
  void eval_generative();
  Eigen::MatrixXd get_output();
  Eigen::MatrixXd get_output_column(const int& i);
  Eigen::MatrixXd get_output_column(const Eigen::VectorXi& i);
  Eigen::MatrixXd get_output_row(const int& i);
  Eigen::MatrixXd get_output_row(const Eigen::VectorXi& i);

  generative_function();
};