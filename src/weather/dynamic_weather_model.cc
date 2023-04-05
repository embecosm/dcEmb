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
#define DEBUG(x) std::cout << #x << "= " << x << std::endl;
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
  Eigen::MatrixXd output =
      eval_generative(parameters, parameter_locations, timeseries_length);

  return Eigen::MatrixXd::Zero(1, 1);
}

Eigen::VectorXd dynamic_weather_model::meinshausen(
    const Eigen::VectorXd& concentration,
    const Eigen::VectorXd& reference_concentration,
    const Eigen::VectorXd& forcing_scaling,
    const Eigen::VectorXd& radiative_efficiency,
    const Eigen::VectorXi& co2_indices, const Eigen::VectorXi& ch4_indices,
    const Eigen::VectorXi& n2o_indices, const Eigen::VectorXi& other_indices) {
  double a1 = -2.4785e-07;
  double b1 = 0.00075906;
  double c1 = -0.0021492;
  double d1 = 5.2488;
  double a2 = -0.00034197;
  double b2 = 0.00025455;
  double c2 = -0.00024357;
  double d2 = 0.12173;
  double a3 = -8.9603e-05;
  double b3 = -0.00012462;
  double d3 = 0.045194;
  Eigen::VectorXd co2 = concentration(co2_indices);
  Eigen::VectorXd ch4 = concentration(ch4_indices);
  Eigen::VectorXd n2o = concentration(n2o_indices);
  Eigen::VectorXd co2_base = reference_concentration(co2_indices);
  Eigen::VectorXd ch4_base = reference_concentration(ch4_indices);
  Eigen::VectorXd n2o_base = reference_concentration(n2o_indices);

  Eigen::VectorXd erf_out = Eigen::VectorXd::Zero(concentration.size());
  Eigen::VectorXd ca_max = co2_base.array() - b1 / (2 * a1);

  std::vector<int> where_central_tmp;
  std::vector<int> where_low_tmp;
  std::vector<int> where_high_tmp;

  for (int i = 0; i < co2_base.size(); i++) {
    if ((co2_base(i) < co2(i)) & (co2(i) <= ca_max[i])) {
      where_central_tmp.push_back(i);
    }
    if (co2(i) <= co2_base(i)) {
      where_low_tmp.push_back(i);
    }
    if (co2(i) > ca_max[i]) {
      where_high_tmp.push_back(i);
    }
  }

  Eigen::VectorXi where_central = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
      where_central_tmp.data(), where_central_tmp.size());
  Eigen::VectorXi where_low = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
      where_low_tmp.data(), where_low_tmp.size());
  Eigen::VectorXi where_high = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
      where_high_tmp.data(), where_high_tmp.size());

  Eigen::VectorXd alpha_p = Eigen::VectorXd::Zero(co2.size());
  Eigen::VectorXd diff = (co2(where_central) - co2_base(where_central));
  alpha_p(where_central) = d1 + a1 * diff.array().square() + b1 * diff.array();
  alpha_p(where_low) = alpha_p(where_low).array() + d1;
  alpha_p(where_high) = alpha_p(where_high).array() + d1 - b1 * b1 / (4 * a1);
  Eigen::VectorXd alpha_n2o = c1 * n2o.array().sqrt();

  erf_out(co2_indices) = (alpha_p + alpha_n2o).array() *
                         (co2.array() / co2_base.array()).log() *
                         forcing_scaling(co2_indices).array();

  erf_out(ch4_indices) =
      ((a3 * ch4.array().sqrt() + b3 * n2o.array().sqrt() + d3) *
       (ch4.array().sqrt() - ch4_base.array().sqrt())) *
      forcing_scaling(ch4_indices).array();

  erf_out(n2o_indices) = ((a2 * co2.array().sqrt() + b2 * n2o.array().sqrt() +
                           c2 * ch4.array().sqrt() + d2) *
                          (n2o.array().sqrt() - n2o_base.array().sqrt())) *
                         forcing_scaling(n2o_indices).array();

  return erf_out;
}

Eigen::VectorXd dynamic_weather_model::calculate_alpha(
    const Eigen::VectorXd& airborne_emissions,
    const Eigen::VectorXd& cumulative_emissions, const Eigen::VectorXd& g0,
    const Eigen::VectorXd& g1, const Eigen::VectorXd& iirf_0,
    const Eigen::VectorXd& iirf_airborne,
    const Eigen::VectorXd& iirf_temperature, const Eigen::VectorXd& iirf_uptake,
    const Eigen::VectorXd& temperature, double iirf_max) {
  Eigen::VectorXd iirf_im1 =
      iirf_0.array() +
      iirf_uptake.array() *
          (cumulative_emissions - airborne_emissions).array() +
      iirf_temperature.array() * temperature.array() +
      iirf_airborne.array() * airborne_emissions.array();
  Eigen::VectorXd iirf_im2 = iirf_im1.unaryExpr(
      [iirf_max](double x) { return (x > iirf_max ? iirf_max : x); });
  Eigen::VectorXd alpha = g0.array() * (iirf_im2.array() / g1.array()).exp();
  // TODO: any that go nan -> 1
  return alpha;
}

std::vector<Eigen::MatrixXd> dynamic_weather_model::step_concentration(
    const Eigen::VectorXd& emissions, const Eigen::MatrixXd& gasboxes_old,
    const Eigen::VectorXd& airborne_emissions_old,
    const Eigen::VectorXd& alpha_lifetime,
    const Eigen::VectorXd& baseline_concentration,
    const Eigen::VectorXd& baseline_emissions,
    const Eigen::VectorXd& concentration_per_emission,
    const Eigen::MatrixXd& lifetime, const Eigen::MatrixXd& partition_fraction,
    int timestep) {
  Eigen::MatrixXd alpha_lifetime_array(lifetime.rows(), lifetime.cols());
  Eigen::MatrixXd emissions_array(lifetime.rows(), lifetime.cols());
  Eigen::MatrixXd baseline_emissions_array(lifetime.rows(), lifetime.cols());

  for (int i = 0; i < lifetime.rows(); i++) {
    alpha_lifetime_array.row(i) = alpha_lifetime;
    emissions_array.row(i) = emissions;
    baseline_emissions_array.row(i) = baseline_emissions;
  }

  Eigen::MatrixXd decay_rate =
      timestep / (alpha_lifetime_array.array() * lifetime.array());
  Eigen::MatrixXd decay_factor = (-(decay_rate.array())).exp();

  Eigen::MatrixXd gasboxes_new =
      (partition_fraction.array() *
           (emissions_array - baseline_emissions_array).array() * 1 /
           decay_rate.array() * (1 - decay_factor.array()) * timestep +
       gasboxes_old.array() * decay_factor.array());

  Eigen::VectorXd airborne_emissions_new = gasboxes_new.colwise().sum();

  Eigen::VectorXd concentration_out =
      (baseline_concentration.array() +
       concentration_per_emission.array() * airborne_emissions_new.array());
  std::vector<Eigen::MatrixXd> out_mat(3);
  out_mat[0] = (concentration_out);
  out_mat[1] = (gasboxes_new);
  out_mat[2] = (airborne_emissions_new);

  return out_mat;
}

/**
 * Dynamic Causal Model constructor for the weather problem
 */
dynamic_weather_model::dynamic_weather_model() { return; }