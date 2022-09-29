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

#include "dynamic_COVID_model.hh"
#include "utility.hh"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/SVD"

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <vector>
#define FSGEN

/**
 * Observed outcomes for the COVID problem.
 */
Eigen::VectorXd dynamic_COVID_model::get_observed_outcomes() {
  this->fs_function.eval_features(this->response_vars);
  Eigen::MatrixXd fs_mat = fs_function.get_fs_response_vars();
  Eigen::Map<Eigen::VectorXd> observed_outcomes(fs_mat.data(),
                                                fs_mat.rows() * fs_mat.cols());
  return observed_outcomes;
}

/**
 * Return the wrapped forward model for the COVID problem
 */
std::function<Eigen::VectorXd(Eigen::VectorXd)>
dynamic_COVID_model::get_forward_model_function() {
  std::function<Eigen::VectorXd(Eigen::VectorXd)> forward_model = std::bind(
      &dynamic_COVID_model::forward_model, this, std::placeholders::_1,
      this->parameter_locations, this->num_samples, this->select_response_vars);
  return forward_model;
}

/**
 * Returns the forward model for the 3body problem
 */
Eigen::VectorXd dynamic_COVID_model::forward_model(
    const Eigen::VectorXd& parameters,
    const parameter_location_COVID& parameter_locations,
    const int& timeseries_length, const Eigen::VectorXi& select_response_vars) {
  this->gen_function.eval_generative(parameters, parameter_locations,
                                     timeseries_length);
  this->fs_function.eval_features(
      this->gen_function.get_output_column(select_response_vars));
  Eigen::MatrixXd fs_output = this->fs_function.get_fs_response_vars();
  Eigen::Map<Eigen::VectorXd> output(fs_output.data(),
                                     fs_output.rows() * fs_output.cols());
  return output;
}

/**
 * Dynamic Causal Model constructor for the COVID problem
 */
dynamic_COVID_model::dynamic_COVID_model() { return; }