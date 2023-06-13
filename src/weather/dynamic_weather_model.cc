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

#include <unsupported/Eigen/MatrixFunctions>
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
#define DEBUG(x) std::cout << #x << "= " << '\n' << x << std::endl;
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
    const Eigen::VectorXd& parameters, const parameter_location_weather& pl,
    const int& timeseries_length) {
  // TODO IF

  Eigen::MatrixXd emissions_array = this->emissions;
  Eigen::MatrixXd forcings_array = this->forcings;
  Eigen::MatrixXd concentrations_array = this->concentrations;
  Eigen::MatrixXd temperature_array = this->temperature;
  Eigen::MatrixXd airborne_emissions_array = this->airborne_emissions;
  Eigen::MatrixXd cumulative_emissions_array = this->cumulative_emissions;
  species_struct sl = this->species_list;

  emissions_array(0, Eigen::seq(250, Eigen::last)) = parameters(Eigen::seq(11, Eigen::last)).transpose(); 

  Eigen::MatrixXd eb_matrix = calculate_eb_matrix(
      pl.ocean_heat_transfer.size(), parameters(pl.deep_ocean_efficacy).value(),
      parameters(pl.gamma_autocorrelation).value(),
      parameters(pl.ocean_heat_transfer), parameters(pl.ocean_heat_capacity));

  Eigen::MatrixXd eb_matrix_d = eb_matrix.exp();

  Eigen::VectorXd forcing_vector = Eigen::VectorXd::Zero(4);
  forcing_vector(0) = parameters(pl.gamma_autocorrelation).value();

  Eigen::VectorXd forcing_vector_d = eb_matrix.colPivHouseholderQr().solve(
      (eb_matrix_d - Eigen::MatrixXd::Identity(4, 4)) * forcing_vector);

  emissions_array(species_list.co2_indices, Eigen::all) =
      emissions_array(species_list.co2_afolu_indices, Eigen::all) +
      emissions_array(species_list.co2_ffi_indices, Eigen::all);

  for (int i = 1; i < timeseries_length; i++) {
    cumulative_emissions_array(Eigen::all, i) =
        cumulative_emissions_array(Eigen::all, i - 1) +
        emissions_array(Eigen::all, i);
  }

  Eigen::VectorXd forcing_sum_array = Eigen::VectorXd::Zero(timeseries_length);
  forcing_sum_array(0) = forcings_array(Eigen::all, 0).array().sum();

  Eigen::VectorXd forcing_efficacy_sum_array =
      Eigen::VectorXd::Zero(timeseries_length);

  Eigen::VectorXd forcing_scale_array =
      sl.forcing_scale.array() * (sl.tropospheric_adjustment.array() + 1.0);

  Eigen::MatrixXd cummins_state_array =
      Eigen::MatrixXd::Zero(temperature_array.rows() + 1, timeseries_length);
  cummins_state_array(0, Eigen::all) = forcing_sum_array;
  cummins_state_array(Eigen::seqN(1, temperature_array.rows()), Eigen::all) =
      temperature_array;

  Eigen::MatrixXd alpha_lifetime_array =
      Eigen::MatrixXd::Zero(sl.name.size(), timeseries_length);

  Eigen::MatrixXd gas_partitions_array = Eigen::MatrixXd::Zero(8, 4);

  Eigen::VectorXd ghg_forcing_offset = meinshausen(
      sl.baseline_concentration, sl.forcing_reference_concentration,
      forcing_scale_array, sl.greenhouse_gas_radiative_efficiency,
      sl.co2_indices, sl.ch4_indices, sl.n2o_indices, sl.other_gh_indices);

  for (int i = 0; i < timeseries_length - 1; i++) {
    alpha_lifetime_array(sl.ghg_indices, i) = calculate_alpha(
        airborne_emissions_array((sl.ghg_indices), i),
        cumulative_emissions_array((sl.ghg_indices), i), sl.g0(sl.ghg_indices),
        sl.g1(sl.ghg_indices), sl.iirf_0(sl.ghg_indices),
        sl.iirf_airborne(sl.ghg_indices), sl.iirf_temperature(sl.ghg_indices),
        sl.iirf_uptake(sl.ghg_indices), cummins_state_array(1, i), 100);

    std::vector<Eigen::MatrixXd> con_step = step_concentration(
        emissions_array(sl.ghg_forward_indices, i),
        gas_partitions_array(sl.ghg_forward_indices, Eigen::all),
        airborne_emissions_array(sl.ghg_forward_indices, i + 1),
        alpha_lifetime_array(sl.ghg_forward_indices, i),
        sl.baseline_concentration(sl.ghg_forward_indices),
        sl.baseline_emissions(sl.ghg_forward_indices),
        sl.concentration_per_emission(sl.ghg_forward_indices),
        sl.unperturbed_lifetime(Eigen::all, sl.ghg_forward_indices),
        sl.partition_fraction(Eigen::all, sl.ghg_forward_indices), 1);

    concentrations_array(sl.ghg_forward_indices, i + 1) = con_step.at(0);
    gas_partitions_array(sl.ghg_forward_indices, Eigen::all) =
        con_step.at(1).transpose();
    airborne_emissions_array(sl.ghg_forward_indices, i + 1) = con_step.at(2);

    std::vector<Eigen::MatrixXd> em_step = unstep_concentration(
        concentrations_array(sl.ghg_inverse_indices, i + 1),
        gas_partitions_array(sl.ghg_inverse_indices, Eigen::all),
        airborne_emissions_array(sl.ghg_inverse_indices, i),
        alpha_lifetime_array(sl.ghg_inverse_indices, i),
        sl.baseline_concentration(sl.ghg_inverse_indices),
        sl.baseline_emissions(sl.ghg_inverse_indices),
        sl.concentration_per_emission(sl.ghg_inverse_indices),
        sl.unperturbed_lifetime(Eigen::all, sl.ghg_inverse_indices),
        sl.partition_fraction(Eigen::all, sl.ghg_inverse_indices), 1);

    emissions_array(sl.ghg_inverse_indices, i) = em_step.at(0);
    gas_partitions_array(sl.ghg_inverse_indices, Eigen::all) =
        em_step.at(1).transpose();
    airborne_emissions_array(sl.ghg_inverse_indices, i + 1) = em_step.at(2);

    cumulative_emissions_array(Eigen::all, i + 1) =
        cumulative_emissions_array(Eigen::all, i) +
        emissions_array(Eigen::all, i) * 1;

    // Need to step/unstep concentration first
    Eigen::VectorXd ghg_forcing_unoffset =
        meinshausen(concentrations_array(Eigen::all, i + 1),
                    sl.forcing_reference_concentration, forcing_scale_array,
                    sl.greenhouse_gas_radiative_efficiency, sl.co2_indices,
                    sl.ch4_indices, sl.n2o_indices, sl.other_gh_indices);

    forcings_array(sl.ghg_indices, i + 1) =
        (ghg_forcing_unoffset - ghg_forcing_offset)(sl.ghg_indices);

    // if sl.ari_indices is not empty
    if (sl.ari_indices.size()) {
      forcings_array(sl.ari_indices, i + 1) = calculate_erafi_forcing(
          emissions_array(Eigen::all, i),
          concentrations_array(Eigen::all, i + 1), sl.baseline_emissions,
          sl.baseline_concentration, forcing_scale_array,
          sl.erfari_radiative_efficiency,
          sl.aerosol_chemistry_from_emissions_indices,
          sl.aerosol_chemistry_from_concentration_indices);
    }
    // if sl.ari_indices is not empty
    if (sl.aci_indices.size()) {
      forcings_array(sl.aci_indices, i + 1) = calculate_eraci_forcing(
          emissions_array(Eigen::all, i),
          concentrations_array(Eigen::all, i + 1), sl.baseline_emissions,
          sl.baseline_concentration, forcing_scale_array, sl.aci_scale,
          sl.aci_shape, sl.aerosol_chemistry_from_emissions_indices,
          sl.aerosol_chemistry_from_concentration_indices);
    }
    Eigen::VectorXd forcing_feedback =
        Eigen::VectorXd::Ones(forcings_array.rows()) *
        cummins_state_array(1, i);
    forcings_array(Eigen::all, i + 1) =
        forcings_array(Eigen::all, i + 1).array() +
        (forcing_feedback.array() * sl.forcing_temperature_feedback.array());

    forcing_sum_array(i + 1) = forcings_array(Eigen::all, i + 1).array().sum();

    forcing_efficacy_sum_array(i + 1) =
        (forcings_array(Eigen::all, i + 1).array() *
         sl.forcing_efficacy.array())
            .array()
            .sum();

    cummins_state_array(Eigen::all, i + 1) =
        step_temperature(cummins_state_array(Eigen::all, i), eb_matrix_d,
                         forcing_vector_d, forcing_efficacy_sum_array(i + 1));
  }

  Eigen::MatrixXd cummins_state_array_t = cummins_state_array.transpose();

  Eigen::MatrixXd out_mat =
      Eigen::MatrixXd(emissions_array.rows() + concentrations_array.rows() +
                          forcings_array.rows() + cummins_state_array.rows() +
                          airborne_emissions_array.rows(),
                      timeseries_length);

  out_mat << emissions_array, concentrations_array, forcings_array,
      cummins_state_array, airborne_emissions_array;
  out_mat =
      out_mat.unaryExpr([](double x) { return (std::isnan(x)) ? 0.0 : x; });
  // consistency with other demo models - time in rows
  return out_mat.transpose();
}
Eigen::MatrixXd dynamic_weather_model::eval_generative(
    const Eigen::VectorXd& parameters,
    const parameter_location_weather& parameter_locations,
    const int& timeseries_length, const Eigen::VectorXi& select_response_vars) {
  Eigen::MatrixXd output =
      eval_generative(parameters, parameter_locations, timeseries_length);

  return output(Eigen::all, select_response_vars);
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
    const double& temperature, double iirf_max) {
  Eigen::VectorXd temperature_vec =
      Eigen::VectorXd::Ones(iirf_0.size()) * temperature;

  Eigen::VectorXd iirf_im1 =
      iirf_0.array() +
      iirf_uptake.array() *
          (cumulative_emissions - airborne_emissions).array() +
      iirf_temperature.array() * temperature_vec.array() +
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
       gasboxes_old.transpose().array() * decay_factor.array());

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

std::vector<Eigen::MatrixXd> dynamic_weather_model::unstep_concentration(
    const Eigen::VectorXd& concentrations, const Eigen::MatrixXd& gasboxes_old,
    const Eigen::VectorXd& airborne_emissions_old,
    const Eigen::VectorXd& alpha_lifetime,
    const Eigen::VectorXd& baseline_concentration,
    const Eigen::VectorXd& baseline_emissions,
    const Eigen::VectorXd& concentration_per_emission,
    const Eigen::MatrixXd& lifetime, const Eigen::MatrixXd& partition_fraction,
    int timestep) {
  Eigen::MatrixXd alpha_lifetime_array(lifetime.rows(), lifetime.cols());

  for (int i = 0; i < lifetime.rows(); i++) {
    alpha_lifetime_array.row(i) = alpha_lifetime;
  }

  Eigen::MatrixXd decay_rate =
      timestep / (alpha_lifetime_array.array() * lifetime.array());
  Eigen::MatrixXd decay_factor = (-(decay_rate.array())).exp();

  Eigen::VectorXd airborne_emissions_new =
      (concentrations - baseline_concentration).array() /
      concentration_per_emission.array();

  Eigen::VectorXd emissions_new =
      (airborne_emissions_new.array() -
       (gasboxes_old.array().transpose() * decay_factor.array())
           .colwise()
           .sum()
           .transpose()) /
      (partition_fraction.array() / decay_rate.array() *
       (1 - decay_factor.array()) * timestep)
          .colwise()
          .sum()
          .transpose();

  Eigen::MatrixXd emissions_array(lifetime.rows(), lifetime.cols());

  for (int i = 0; i < lifetime.rows(); i++) {
    emissions_array.row(i) = emissions_new;
  }

  Eigen::MatrixXd gasboxes_new =
      timestep * emissions_array.array() * partition_fraction.array() * 1 /
          decay_rate.array() * (1 - decay_factor.array()) +
      gasboxes_old.array().transpose() * decay_factor.array();

  std::vector<Eigen::MatrixXd> out_mat(3);

  out_mat[0] = (emissions_new + baseline_emissions);
  out_mat[1] = (gasboxes_new);
  out_mat[2] = (airborne_emissions_new);

  return out_mat;
}

Eigen::VectorXd dynamic_weather_model::calculate_erafi_forcing(
    const Eigen::VectorXd& emissions, const Eigen::VectorXd& concentrations,
    const Eigen::VectorXd& baseline_emissions,
    const Eigen::VectorXd& baseline_concentration,
    const Eigen::VectorXd& forcing_scale_array,
    const Eigen::VectorXd& radiative_efficiency,
    const Eigen::VectorXi& emissions_indices,
    const Eigen::VectorXi& concentrations_indices) {
  Eigen::VectorXd erf_out = Eigen::VectorXd::Zero(emissions.size());

  erf_out(emissions_indices) =
      ((emissions(emissions_indices) - baseline_emissions(emissions_indices))
           .array() *
       radiative_efficiency(emissions_indices).array()) *
      forcing_scale_array(emissions_indices).array();

  erf_out(concentrations_indices) =
      ((concentrations(concentrations_indices) -
        baseline_concentration(concentrations_indices))
           .array() *
       radiative_efficiency(concentrations_indices).array()) *
      forcing_scale_array(concentrations_indices).array();

  // Hack: currently we're summing outputs for simplicity. We may not in future!
  return Eigen::VectorXd::Ones(1) * erf_out.sum();
}

Eigen::VectorXd dynamic_weather_model::calculate_eraci_forcing(
    const Eigen::VectorXd& emissions, const Eigen::VectorXd& concentrations,
    const Eigen::VectorXd& baseline_emissions,
    const Eigen::VectorXd& baseline_concentration,
    const Eigen::VectorXd& forcing_scale_array, const Eigen::VectorXd& scale,
    const Eigen::VectorXd& sensitivity, const Eigen::VectorXi& slcf_indices,
    const Eigen::VectorXi& ghg_indices) {
  Eigen::VectorXd radiative_effect =
      scale(slcf_indices) *
      log(1 +
          (sensitivity(slcf_indices).array() * emissions(slcf_indices).array())
              .sum() +
          (sensitivity(ghg_indices).array() *
           concentrations(ghg_indices).array())
              .sum());

  Eigen::VectorXd baseline_radiative_effect =
      scale(slcf_indices) * log(1 +
                                (sensitivity(slcf_indices).array() *
                                 baseline_emissions(slcf_indices).array())
                                    .sum() +
                                (sensitivity(ghg_indices).array() *
                                 baseline_concentration(ghg_indices).array())
                                    .sum());

  Eigen::VectorXd erf_out =
      (radiative_effect - baseline_radiative_effect).array() *
      forcing_scale_array(slcf_indices).array();

  return erf_out;
}

Eigen::VectorXd dynamic_weather_model::step_temperature(
    const Eigen::VectorXd& state_old, const Eigen::MatrixXd& eb_matrix_d,
    const Eigen::VectorXd& forcing_vector_d, const double& forcing) {
  Eigen::VectorXd state_new =
      (eb_matrix_d * state_old).array() + (forcing_vector_d.array() * forcing);
  return state_new;
}

// n_box is currently hardcoded to be 3. TODO: 2 and 4.
Eigen::MatrixXd dynamic_weather_model::calculate_eb_matrix(
    const int& n_box, const double& deep_ocean_efficacy,
    const double& gamma_autocorrelation,
    const Eigen::VectorXd& ocean_heat_transfer,
    const Eigen::VectorXd& ocean_heat_capacity) {
  Eigen::MatrixXd eb_matrix = Eigen::MatrixXd::Zero(n_box + 1, n_box + 1);
  Eigen::VectorXd epsilon_array = Eigen::VectorXd::Ones(n_box);
  epsilon_array(n_box - 2) = deep_ocean_efficacy;

  eb_matrix(1, 1) =
      -(ocean_heat_transfer(0) + epsilon_array(0) * ocean_heat_transfer(1)) /
      ocean_heat_capacity(0);
  eb_matrix(1, 2) =
      epsilon_array(0) * ocean_heat_transfer(1) / ocean_heat_capacity(0);

  eb_matrix(2, 1) = ocean_heat_transfer(1) / ocean_heat_capacity(1);
  eb_matrix(2, 2) =
      -(ocean_heat_transfer(1) + epsilon_array(1) * ocean_heat_transfer(2)) /
      ocean_heat_capacity(1);
  eb_matrix(2, 3) =
      epsilon_array(1) * ocean_heat_transfer(2) / ocean_heat_capacity(1);

  eb_matrix(3, 2) = ocean_heat_transfer(2) / ocean_heat_capacity(2);
  eb_matrix(3, 3) = -eb_matrix(2, 1);

  eb_matrix(0, 0) = -gamma_autocorrelation;
  eb_matrix(1, 0) = 1 / ocean_heat_capacity(0);

  return eb_matrix;
}

/**
 * Dynamic Causal Model constructor for the weather problem
 */
dynamic_weather_model::dynamic_weather_model() { return; }