/**
 * The 3-body dynamic causal model class within the dcEmb package
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

#include "dynamic_weather_model.hh"
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

#define PL parameter_locations
#define ZERO(a, b) Eigen::MatrixXd::Zero(a, b)

/**
 * Observed outcomes for the weather problem.
 */
Eigen::VectorXd dynamic_weather_model::get_observed_outcomes() {
  Eigen::Map<Eigen::VectorXd> observed_outcomes(
      this->response_vars.data(),
      this->response_vars.rows() * this->response_vars.cols());
  return observed_outcomes;
}

/**
 * Return the wrapped forward model for the weather problem
 */
std::function<Eigen::VectorXd(Eigen::VectorXd)>
dynamic_weather_model::get_forward_model_function() {
  std::function<Eigen::VectorXd(Eigen::VectorXd)> forward_model = std::bind(
      &dynamic_weather_model::forward_model, this, std::placeholders::_1,
      this->parameter_locations, this->num_samples, this->select_response_vars);
  return forward_model;
}

/**
 * Returns the forward model for the weather problem
 */
Eigen::VectorXd dynamic_weather_model::forward_model(
    const Eigen::VectorXd& parameters,
    const parameter_location_weather& parameter_locations,
    const int& timeseries_length, const Eigen::VectorXi& select_response_vars) {
  Eigen::MatrixXd gen = eval_generative(
      parameters, parameter_locations, timeseries_length, select_response_vars);
  Eigen::Map<Eigen::VectorXd> output(gen.data(), gen.rows() * gen.cols());
  return output;
}

/**
 * Evaluate the generative model for the weather problem, using the
 * runge-kutta method
 */
Eigen::MatrixXd dynamic_weather_model::eval_generative(
    const Eigen::VectorXd& parameters,
    const parameter_location_weather& parameter_locations,
    const int& timeseries_length) {
  return Eigen::MatrixXd::Zero(1, 1);
}
Eigen::MatrixXd dynamic_weather_model::eval_generative(
    const Eigen::VectorXd& parameters,
    const parameter_location_weather& parameter_locations,
    const int& timeseries_length, const Eigen::VectorXi& select_response_vars) {
  Eigen::MatrixXd output = eval_generative(parameters, parameter_locations,
                                           timeseries_length);

  return Eigen::MatrixXd::Zero(1, 1);
}

/**
 * Dynamic Causal Model constructor for the weather problem
 */
dynamic_weather_model::dynamic_weather_model() { return; }