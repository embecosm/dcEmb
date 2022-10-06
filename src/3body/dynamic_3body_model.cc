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

#include "dynamic_3body_model.hh"
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
 * Observed outcomes for the 3body problem.
 */
Eigen::VectorXd dynamic_3body_model::get_observed_outcomes() {
  Eigen::Map<Eigen::VectorXd> observed_outcomes(
      this->response_vars.data(),
      this->response_vars.rows() * this->response_vars.cols());
  return observed_outcomes;
}

/**
 * Return the wrapped forward model for the 3body problem
 */
std::function<Eigen::VectorXd(Eigen::VectorXd)>
dynamic_3body_model::get_forward_model_function() {
  std::function<Eigen::VectorXd(Eigen::VectorXd)> forward_model = std::bind(
      &dynamic_3body_model::forward_model, this, std::placeholders::_1,
      this->parameter_locations, this->num_samples, this->select_response_vars);
  return forward_model;
}

/**
 * Returns the forward model for the 3body problem
 */
Eigen::VectorXd dynamic_3body_model::forward_model(
    const Eigen::VectorXd& parameters,
    const parameter_location_3body& parameter_locations,
    const int& timeseries_length, const Eigen::VectorXi& select_response_vars) {
  Eigen::MatrixXd gen = eval_generative(
      parameters, parameter_locations, timeseries_length, select_response_vars);
  Eigen::Map<Eigen::VectorXd> output(gen.data(), gen.rows() * gen.cols());
  return output;
}

/**
 * Evaluate the generative model for the 3body problem, using the
 * runge-kutta method
 */
Eigen::MatrixXd dynamic_3body_model::eval_generative(
    const Eigen::VectorXd& parameters,
    const parameter_location_3body& parameter_locations,
    const int& timeseries_length) {
  Eigen::MatrixXd output =
      Eigen::MatrixXd::Zero(timeseries_length, parameters.size());
  double h = 0.001;
  Eigen::VectorXd state = parameters;
  output.row(0) = state;
  std::function<Eigen::VectorXd(Eigen::VectorXd)> dfdt = std::bind(
      &dynamic_3body_model::differential_eq, this, std::placeholders::_1);
  for (int i = 1; i < timeseries_length; i++) {
    for (int j = 0; j < 10; j++) {
      Eigen::VectorXd state_delta = utility::rungekutta(dfdt, state, h);
      state = state + state_delta;
    }
    output.row(i) = state;
  }
  return output;
}
Eigen::MatrixXd dynamic_3body_model::eval_generative(
    const Eigen::VectorXd& parameters,
    const parameter_location_3body& parameter_locations,
    const int& timeseries_length, const Eigen::VectorXi& select_response_vars) {
  Eigen::MatrixXd output =
      eval_generative(parameters, parameter_locations, timeseries_length);

  return output(Eigen::all, select_response_vars);
}

/**
 * Calculate an update of the existing positions, masses and velocities in the
 * 3body state
 */
Eigen::VectorXd dynamic_3body_model::differential_eq(
    const Eigen::VectorXd& state_in) {
  Eigen::VectorXd state_var = state_in;
  double G = 1;
  Eigen::Map<Eigen::MatrixXd> state(state_var.data(), 7, 3);
  Eigen::MatrixXd return_matrix =
      Eigen::MatrixXd::Zero(state.rows(), state.cols());
  for (int i = 0; i < state.cols(); i++) {
    return_matrix(1, i) = state(4, i);
    return_matrix(2, i) = state(5, i);
    return_matrix(3, i) = state(6, i);
    for (int j = 0; j < state.cols(); j++) {
      if (i == j) {
        continue;
      }
      double distancex = state(1, j) - state(1, i);
      double distancey = state(2, j) - state(2, i);
      double distancez = state(3, j) - state(3, i);
      double distance_euclidian =
          sqrt((distancex * distancex) + (distancey * distancey) +
               (distancez * distancez));
      return_matrix(4, i) +=
          (G * state(0, j) * distancex) / pow(distance_euclidian, 3);
      return_matrix(5, i) +=
          (G * state(0, j) * distancey) / pow(distance_euclidian, 3);
      return_matrix(6, i) +=
          (G * state(0, j) * distancez) / pow(distance_euclidian, 3);
    }
  }
  Eigen::Map<Eigen::VectorXd> return_state(
      return_matrix.data(), return_matrix.rows() * return_matrix.cols());
  return return_state;
}

/**
 * Dynamic Causal Model constructor for the 3body problem
 */
dynamic_3body_model::dynamic_3body_model() { return; }