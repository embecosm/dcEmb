/**
 * A feature selection class for the COVID-19 dynamic causal model within the
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

#include "feature_selection_COVID.hh"
#include <stdarg.h>
#include <Eigen/Dense>
#include <iostream>
/**
 * Evaluate feature selection for the 3body problem
 */
void feature_selection_COVID::eval_features(Eigen::MatrixXd response_vars) {
  this->response_vars = response_vars;
  eval_features();
}
void feature_selection_COVID::eval_features() {
  Eigen::MatrixXd ones_mat = Eigen::MatrixXd::Ones(this->response_vars.rows(),
                                                   this->response_vars.cols());
  this->fs_response_vars =
      (this->response_vars).array().max(ones_mat.array()).sqrt();
  return;
}
/**
 * Constructor for feature selection 3body
 */
feature_selection_COVID::feature_selection_COVID(
    const Eigen::MatrixXd& response_vars) {
  this->response_vars = response_vars;
  return;
}
feature_selection_COVID::feature_selection_COVID() { return; }
