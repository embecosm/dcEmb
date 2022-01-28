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

#include "feature_selection_function.hh"
#include <stdarg.h>
#include <Eigen/Dense>
#include <iostream>

/*
 * Return feature selected response variables
 */
Eigen::MatrixXd feature_selection_function::get_fs_response_vars() {
  return this->fs_response_vars;
}
/*
 * Eval features function, to be overwritten in inheriting classes. If this
 * function is reached, something is wrong and an error is thrown.
 */
void feature_selection_function::eval_features() {
  throw std::runtime_error("feature selection eval_features not specified");
  return;
}

/*
 * feature selection function constructor
 */
feature_selection_function::feature_selection_function() { return; }
