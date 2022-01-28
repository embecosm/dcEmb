/**
 * A generative function (forward model) class for the 3-body dynamic causal
 * model within the dcEmb package
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

#include "generative_3body.hh"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <functional>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>
#include "utility.hh"

#define PL parameter_locations
#define ZERO(a, b) Eigen::MatrixXd::Zero(a, b)

/**
 * Evaluate the generative model for the 3body problem, using the
 * runge-kutta method
 */
void generative_3body::eval_generative(
    const Eigen::VectorXd& parameters,
    const parameter_location_3body& parameter_locations,
    const int& timeseries_length) {
  this->parameters = parameters;
  this->timeseries_length = timeseries_length;
  this->parameter_locations = parameter_locations;
  eval_generative();
}
void generative_3body::eval_generative() {
  this->output = Eigen::MatrixXd::Zero(timeseries_length, parameters.size());
  double h = 0.001;
  Eigen::VectorXd state = this->parameters;
  output.row(0) = state;
  std::function<Eigen::VectorXd(Eigen::VectorXd)> dfdt = std::bind(
      &generative_3body::differential_eq, this, std::placeholders::_1);
  for (int i = 1; i < this->timeseries_length; i++) {
    for (int j = 0; j < 10; j++) {
      Eigen::VectorXd state_delta = utility::rungekutta(dfdt, state, h);
      state = state + state_delta;
    }
    output.row(i) = state;
  }
  return;
}

/**
 * Calculate an update of the existing positions, masses and velocities in the
 * 3body state
 */
Eigen::VectorXd generative_3body::differential_eq(
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
 * Constructor for generative model for the 3body problem
 */
generative_3body::generative_3body(
    const Eigen::VectorXd& parameters,
    const parameter_location_3body& parameter_locations,
    const int& timeseries_length) {
  this->parameters = parameters;
  this->timeseries_length = timeseries_length;
  this->parameter_locations = parameter_locations;
  generative_3body();
  return;
}

// constructors
generative_3body::generative_3body() { return; }