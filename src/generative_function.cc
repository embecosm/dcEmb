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

#include "generative_function.hh"
#include <stdarg.h>
#include <Eigen/Dense>
#include <iostream>

/*
 * eval_generative function, to be overwritten in inheriting classes. If
 * this function is reached, something is wrong and an error is thrown.
 */
void generative_function::eval_generative() {
  throw std::runtime_error(
      "error: generative model eval_generative not specified");
  return;
}
/*
 * Return generative output
 */
Eigen::MatrixXd generative_function::get_output() { return this->output; }
/*
 * Return generative output at specific column
 */
Eigen::MatrixXd generative_function::get_output_column(const int& i) {
  return (this->output.col(i));
}
/*
 * Return generative output at specific columns
 */
Eigen::MatrixXd generative_function::get_output_column(
    const Eigen::VectorXi& i) {
  return (this->output(Eigen::all, i));
}
/*
 * Return generative output at specific row
 */
Eigen::MatrixXd generative_function::get_output_row(const int& i) {
  return (this->output.row(i));
}
/*
 * Return generative output at specific rows
 */
Eigen::MatrixXd generative_function::get_output_row(const Eigen::VectorXi& i) {
  return (this->output(i, Eigen::all));
}
/*
 * generative function constructor
 */
generative_function::generative_function() { return; }