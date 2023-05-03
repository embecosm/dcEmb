/**
 * The weather dynamic causal model class within the dcEmb package
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

#include <vector>
#include "dynamic_model.hh"
#include "parameter_location_weather.hh"
#include "species_struct.hh"
#pragma once

/**
 * Dynamic Causal Model class for the weather problem.
 */
class dynamic_weather_model : public dynamic_model {
 public:
  /**
   * @brief metadata containing the names and indexes of parameters in the
   * parameter array, so parameters can be referenced by name for clarity
   */
  parameter_location_weather parameter_locations;
  species_struct species_list;
  Eigen::MatrixXd ecf;
  double airborne_emissions;
  double cumulative_emissions;
  Eigen::VectorXd get_observed_outcomes();
  std::function<Eigen::VectorXd(Eigen::VectorXd)> get_forward_model_function();
  Eigen::VectorXd forward_model(
      const Eigen::VectorXd& parameters,
      const parameter_location_weather& parameter_locations,
      const int& timeseries_length,
      const Eigen::VectorXi& select_response_vars);
  Eigen::MatrixXd eval_generative(
      const Eigen::VectorXd& parameters,
      const parameter_location_weather& parameter_locations,
      const int& timeseries_length);
  Eigen::MatrixXd eval_generative(
      const Eigen::VectorXd& parameters,
      const parameter_location_weather& parameter_locations,
      const int& timeseries_length,
      const Eigen::VectorXi& select_response_vars);
  Eigen::VectorXd meinshausen(
      const Eigen::VectorXd& concentration,
      const Eigen::VectorXd& reference_concentration,
      const Eigen::VectorXd& forcing_scaling,
      const Eigen::VectorXd& radiative_efficiency,
      const Eigen::VectorXi& co2_indices, const Eigen::VectorXi& ch4_indices,
      const Eigen::VectorXi& n2o_indices,
      const Eigen::VectorXi& minor_greenhouse_gas_indices);
  Eigen::VectorXd calculate_alpha(const Eigen::VectorXd& airborne_emissions,
                                  const Eigen::VectorXd& cumulative_emissions,
                                  const Eigen::VectorXd& g0,
                                  const Eigen::VectorXd& g1,
                                  const Eigen::VectorXd& iirf_0,
                                  const Eigen::VectorXd& iirf_airborne,
                                  const Eigen::VectorXd& iirf_temperature,
                                  const Eigen::VectorXd& iirf_uptake,
                                  const Eigen::VectorXd& cummins_state_array,
                                  double iirf_max);
  std::vector<Eigen::MatrixXd> step_concentration(
      const Eigen::VectorXd& emissions, const Eigen::MatrixXd& gasboxes_old,
      const Eigen::VectorXd& airborne_emissions_old,
      const Eigen::VectorXd& alpha_lifetime,
      const Eigen::VectorXd& baseline_concentration,
      const Eigen::VectorXd& baseline_emissions,
      const Eigen::VectorXd& concentration_per_emission,
      const Eigen::MatrixXd& lifetime,
      const Eigen::MatrixXd& partition_fraction, int timestep);
  std::vector<Eigen::MatrixXd> unstep_concentration(
      const Eigen::VectorXd& concentrations,
      const Eigen::MatrixXd& gasboxes_old,
      const Eigen::VectorXd& airborne_emissions_old,
      const Eigen::VectorXd& alpha_lifetime,
      const Eigen::VectorXd& baseline_concentration,
      const Eigen::VectorXd& baseline_emissions,
      const Eigen::VectorXd& concentration_per_emission,
      const Eigen::MatrixXd& lifetime,
      const Eigen::MatrixXd& partition_fraction, int timestep);
  dynamic_weather_model();
};

inline bool operator==(const dynamic_weather_model& lhs,
                       const dynamic_weather_model& rhs) {
  return lhs.parameter_locations == rhs.parameter_locations &
         lhs.max_invert_it == rhs.max_invert_it &
         lhs.conditional_parameter_expectations ==
             rhs.conditional_parameter_expectations &
         lhs.conditional_parameter_covariances ==
             rhs.conditional_parameter_covariances &
         lhs.conditional_hyper_expectations ==
             rhs.conditional_hyper_expectations &
         lhs.free_energy == rhs.free_energy &
         lhs.prior_parameter_expectations == rhs.prior_parameter_expectations &
         lhs.prior_parameter_covariances == rhs.prior_parameter_covariances &
         lhs.prior_hyper_expectations == rhs.prior_hyper_expectations &
         lhs.prior_hyper_covariances == rhs.prior_hyper_covariances &
         lhs.num_samples == rhs.num_samples &
         lhs.num_response_vars == rhs.num_response_vars &
         lhs.select_response_vars == rhs.select_response_vars &
         lhs.response_vars == rhs.response_vars;
}