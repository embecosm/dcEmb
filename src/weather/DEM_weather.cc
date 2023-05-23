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

#include "DEM_weather.hh"
#include <stdio.h>
#include <iostream>
#include <list>
#include <vector>
#include "Eigen/Dense"
#include "bmr_model.hh"
#include "country_data.hh"
#include "dynamic_weather_model.hh"
#include "species_struct.hh"
#include "utility.hh"
#define DEBUG(x) std::cout << #x << "= " << '\n' << x << std::endl;

/**
 * Run the weather example
 */
int run_weather_test() {
  dynamic_weather_model model;
  model.prior_parameter_expectations = default_prior_expectations();
  model.prior_parameter_covariances = default_prior_covariances();
  model.prior_hyper_expectations = default_hyper_expectations();
  model.prior_hyper_covariances = default_hyper_covariances();
  model.parameter_locations = default_parameter_locations();

  int start_date = 2000;
  int end_date = 2050;
  int sz = end_date - start_date + 1;
  model.num_samples = sz;

  model.num_response_vars = 8;
  model.select_response_vars =
      (Eigen::VectorXi(8) << 0, 1, 2, 3, 4, 5, 6, 7).finished();
  std::vector<std::string> species_names(
      {"CO2 FFI", "CO2 AFOLU", "Sulfur", "CH4", "N2O", "CO2"});

  model.species_list = simple_species_struct(species_names);

  std::vector<Eigen::MatrixXd> ecf =
      simple_ecf(model.species_list, "ssp119", start_date, end_date);
  model.emissions = ecf.at(0);
  model.concentrations = ecf.at(1);
  model.forcings = ecf.at(2);

  model.emissions(0, Eigen::all) = Eigen::VectorXd::Ones(sz) * 38;
  model.emissions(1, Eigen::all) = Eigen::VectorXd::Ones(sz) * 3;
  model.emissions(2, Eigen::all) = Eigen::VectorXd::Ones(sz) * 100;
  model.concentrations(3, Eigen::all) = Eigen::VectorXd::Ones(sz) * 1800;
  model.concentrations(4, Eigen::all) = Eigen::VectorXd::Ones(sz) * 325;
  model.concentrations(5, 0) = 278.3;
  model.temperature = Eigen::MatrixXd::Zero(3, sz);
  model.airborne_emissions =
      Eigen::MatrixXd::Zero(model.species_list.name.size(), sz);
  model.cumulative_emissions =
      Eigen::MatrixXd::Zero(model.species_list.name.size(), sz);

  Eigen::MatrixXd true_out = model.eval_generative(
      true_prior_expectations(), default_parameter_locations(), sz);

  model.response_vars = true_out(Eigen::all, model.select_response_vars);

  model.invert_model();

  Eigen::MatrixXd prior_e_out = model.eval_generative(
      default_prior_expectations(), default_parameter_locations(), sz);

  Eigen::MatrixXd posterior_e_out =
      model.eval_generative(model.conditional_parameter_expectations,
                            default_parameter_locations(), sz);

  Eigen::MatrixXd prior_c_out_m = model.eval_generative(
      default_prior_expectations(), default_parameter_locations(), sz);
  Eigen::MatrixXd posterior_c_out_m =
      model.eval_generative(model.conditional_parameter_expectations,
                            default_parameter_locations(), sz);

  Eigen::MatrixXd prior_c_out_v =
      Eigen::MatrixXd::Zero(prior_c_out_m.rows(), prior_c_out_m.cols());
  Eigen::MatrixXd posterior_c_out_v =
      Eigen::MatrixXd::Zero(posterior_c_out_m.rows(), posterior_c_out_m.cols());

  std::default_random_engine rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dis(0, 1);

  int n = 1000;
  for (int i = 1; i < n; i++) {
    std::cout << "simulating variance: " << i << '\n';
    Eigen::VectorXd rand_param_prior =
        Eigen::VectorXd::Zero(model.prior_parameter_expectations.size())
            .unaryExpr([&](double dummy) { return dis(gen); });

    rand_param_prior = ((rand_param_prior.array() *
                         model.prior_parameter_covariances.diagonal().array()) +
                        model.prior_parameter_expectations.array())
                           .eval();
    Eigen::MatrixXd prior_tmp = model.eval_generative(
        rand_param_prior, default_parameter_locations(), sz);
    Eigen::MatrixXd prior_c_out_m_old = prior_c_out_m;
    prior_c_out_m = prior_c_out_m_old.array() +
                    ((prior_tmp - prior_c_out_m_old).array() / i);
    prior_c_out_v =
        (prior_c_out_v.array() + (prior_tmp - prior_c_out_m_old).array() *
                                     (prior_tmp - prior_c_out_m).array())
            .eval();

    Eigen::VectorXd rand_param_posterior =
        Eigen::VectorXd::Zero(model.conditional_parameter_expectations.size())
            .unaryExpr([&](double dummy) { return dis(gen); });

    rand_param_posterior =
        ((rand_param_posterior.array() *
          model.conditional_parameter_covariances.diagonal().array()) +
         model.conditional_parameter_expectations.array())
            .eval();
    Eigen::MatrixXd posterior_tmp = model.eval_generative(
        rand_param_posterior, default_parameter_locations(), sz);
    Eigen::MatrixXd posterior_c_out_m_old = posterior_c_out_m;
    posterior_c_out_m = posterior_c_out_m_old.array() +
                        ((posterior_tmp - posterior_c_out_m_old).array() / i);
    posterior_c_out_v = (posterior_c_out_v.array() +
                         (posterior_tmp - posterior_c_out_m_old).array() *
                             (posterior_tmp - posterior_c_out_m).array())
                            .eval();
  }

  Eigen::MatrixXd prior_c_out = prior_c_out_v.array() / n;
  Eigen::MatrixXd posterior_c_out = posterior_c_out_v.array() / n;

  // for (int i = i; i < n; i++) {

  // }

  utility::print_matrix("../visualisation/weather/true_generative.csv",
                        true_out);
  utility::print_matrix("../visualisation/weather/prior_generative.csv",
                        prior_e_out);
  utility::print_matrix("../visualisation/weather/prior_generative_var.csv",
                        prior_c_out);
  utility::print_matrix("../visualisation/weather/pos_generative.csv",
                        posterior_e_out);
  utility::print_matrix("../visualisation/weather/pos_generative_var.csv",
                        posterior_c_out);
  // std::cout << "temperature" << true_out(25, Eigen::all) << '\n';
  return 0;
}

// Fill emissions, concentrations and/or forcings
std::vector<Eigen::MatrixXd> simple_ecf(const species_struct& species,
                                        const std::string& scenario,
                                        const int& start_date,
                                        const int& end_date) {
  Eigen::MatrixXd e_matrix =
      Eigen::MatrixXd::Zero(species.name.size(), end_date - start_date + 1);
  Eigen::MatrixXd c_matrix =
      Eigen::MatrixXd::Zero(species.name.size(), end_date - start_date + 1);
  Eigen::MatrixXd f_matrix =
      Eigen::MatrixXd::Zero(species.name.size(), end_date - start_date + 1);
  std::string emissions_filename =
      "../src/weather/data/rcmip-emissions-annual-means-v5-1-0.csv";
  std::string concentrations_filename =
      "../src/weather/data/rcmip-concentrations-annual-means-v5-1-0.csv";
  std::string forcings_filename =
      "../src/weather/data/rcmip-radiative-forcing-annual-means-v5-1-0.csv";

  std::vector<std::string> species_names_rcmip = species.name;

  std::replace(species_names_rcmip.begin(), species_names_rcmip.end(),
               (std::string) "CO2 FFI",
               (std::string) "CO2|MAGICC Fossil and Industrial");
  std::replace(species_names_rcmip.begin(), species_names_rcmip.end(),
               (std::string) "CO2 AFOLU", (std::string) "CO2|MAGICC AFOLU");

  std::string emissions_line;
  std::ifstream emissions_file;
  emissions_file.open(emissions_filename);
  while (std::getline(emissions_file, emissions_line)) {
    std::vector<std::string> emissions_split;
    std::vector<std::string> variable_split;
    utility::splitstr(emissions_split, emissions_line, ',');
    utility::splitstr(variable_split, emissions_split.at(3), '|');
    // std::cout << emissions_line << '\n';
    if (emissions_split.at(2) != "World") {
      continue;
    }
    if (emissions_split.at(1) != scenario) {
      continue;
    }
    int name_idx = std::find(species_names_rcmip.begin(),
                             species_names_rcmip.end(), variable_split.back()) -
                   species_names_rcmip.begin();
    if (name_idx == species_names_rcmip.size()) {
      if (variable_split.size() == 1) {
        continue;
      }
      name_idx =
          std::find(species_names_rcmip.begin(), species_names_rcmip.end(),
                    (variable_split.at(variable_split.size() - 2) + "|" +
                     variable_split.back())) -
          species_names_rcmip.begin();
      if (name_idx == species_names_rcmip.size()) {
        continue;
      }
    }
    // There's probably a way to do this elegantly with Eigen::Map in a way that
    // reuses the memory
    if (species.input_mode.at(name_idx) == "emissions") {
      for (int i = 0; i < (end_date - start_date + 1); i++) {
        int pos = start_date - 1700 + 7 + i;
        if (pos < emissions_split.size()) {
          std::string s = emissions_split.at(pos);
          e_matrix(name_idx, i) =
              (s == "") ? 0 : std::stod(emissions_split.at(pos));
        }
      }
    }
  }

  std::string concentrations_line;
  std::ifstream concentrations_file;
  concentrations_file.open(concentrations_filename);
  while (std::getline(concentrations_file, concentrations_line)) {
    std::vector<std::string> concentrations_split;
    std::vector<std::string> variable_split;
    utility::splitstr(concentrations_split, concentrations_line, ',');
    utility::splitstr(variable_split, concentrations_split.at(3), '|');
    // std::cout << concentrations_line << '\n';
    if (concentrations_split.at(2) != "World") {
      continue;
    }
    if (concentrations_split.at(1) != scenario) {
      continue;
    }
    int name_idx = std::find(species_names_rcmip.begin(),
                             species_names_rcmip.end(), variable_split.back()) -
                   species_names_rcmip.begin();
    if (name_idx == species_names_rcmip.size()) {
      if (variable_split.size() == 1) {
        continue;
      }
      name_idx =
          std::find(species_names_rcmip.begin(), species_names_rcmip.end(),
                    (variable_split.at(variable_split.size() - 2) + "|" +
                     variable_split.back())) -
          species_names_rcmip.begin();
      if (name_idx == species_names_rcmip.size()) {
        continue;
      }
    }
    // There's probably a way to do this elegantly with Eigen::Map in a way that
    // reuses the memory
    if (species.input_mode.at(name_idx) == "concentrations") {
      for (int i = 0; i < (end_date - start_date + 1); i++) {
        int pos = start_date - 1700 + 7 + i;
        if (pos < concentrations_split.size()) {
          std::string s = concentrations_split.at(pos);
          c_matrix(name_idx, i) =
              (s == "") ? 0 : std::stod(concentrations_split.at(pos));
        }
      }
    }
  }

  std::string forcings_line;
  std::ifstream forcings_file;
  forcings_file.open(forcings_filename);
  while (std::getline(forcings_file, forcings_line)) {
    std::vector<std::string> forcings_split;
    std::vector<std::string> variable_split;
    utility::splitstr(forcings_split, forcings_line, ',');
    utility::splitstr(variable_split, forcings_split.at(3), '|');
    // std::cout << forcings_line << '\n';
    if (forcings_split.at(2) != "World") {
      continue;
    }
    if (forcings_split.at(1) != scenario) {
      continue;
    }
    int name_idx = std::find(species_names_rcmip.begin(),
                             species_names_rcmip.end(), variable_split.back()) -
                   species_names_rcmip.begin();
    if (name_idx == species_names_rcmip.size()) {
      if (variable_split.size() == 1) {
        continue;
      }
      name_idx =
          std::find(species_names_rcmip.begin(), species_names_rcmip.end(),
                    (variable_split.at(variable_split.size() - 2) + "|" +
                     variable_split.back())) -
          species_names_rcmip.begin();
      if (name_idx == species_names_rcmip.size()) {
        continue;
      }
    }
    // There's probably a way to do this elegantly with Eigen::Map in a way that
    // reuses the memory
    if (species.input_mode.at(name_idx) == "forcings") {
      for (int i = 0; i < (end_date - start_date + 1); i++) {
        int pos = start_date - 1700 + 7 + i;
        if (pos < forcings_split.size()) {
          std::string s = forcings_split.at(pos);
          f_matrix(name_idx, i) =
              (s == "") ? 0 : std::stod(forcings_split.at(pos));
        }
      }
    }
  }
  std::vector<Eigen::MatrixXd> out;
  out.push_back(e_matrix);
  out.push_back(c_matrix);
  out.push_back(f_matrix);
  return out;
}

species_struct simple_species_struct(
    const std::vector<std::string>& species_names) {
  std::string filename = "../src/weather/data/species_configs_properties.csv";
  species_struct species = utility::species_from_file(filename, species_names);

  species.input_mode.at(0) = "emissions";
  species.input_mode.at(1) = "emissions";
  species.input_mode.at(2) = "emissions";
  species.input_mode.at(3) = "concentrations";
  species.input_mode.at(4) = "concentrations";
  species.input_mode.at(5) = "calculated";

  species.greenhouse_gas(3) = 1;
  species.greenhouse_gas(4) = 1;

  species.baseline_emissions(2) = 0;
  species.aci_shape = Eigen::VectorXd::Zero(6);
  species.aci_shape(2) = 1 / 260.34644166;

  species_struct erfari(1);
  erfari.name.at(0) = "ERFari";
  erfari.type.at(0) = "ari";
  erfari.input_mode.at(0) = "calculated";
  erfari.greenhouse_gas << 0;
  erfari.aerosol_chemistry_from_emissions << 0;
  erfari.aerosol_chemistry_from_concentration << 0;
  erfari.forcing_efficacy << 1;

  species_struct erfaci(1);
  erfaci.name.at(0) = "ERFaci";
  erfaci.type.at(0) = "aci";
  erfaci.input_mode.at(0) = "calculated";
  erfaci.greenhouse_gas << 0;
  erfaci.aerosol_chemistry_from_emissions << 0;
  erfaci.aerosol_chemistry_from_concentration << 0;
  erfaci.forcing_efficacy << 1;
  species_struct species_out =
      append_species(species, append_species(erfari, erfaci));

  species_out.forcing_scale = Eigen::VectorXd::Ones(8);
  species_out.tropospheric_adjustment = Eigen::VectorXd::Zero(8);
  species_out.aci_scale = Eigen::VectorXd::Ones(8) * -2.09841432;
  utility::update_species_list_indicies(species_out);
  return species_out;
}

/**
 * "True" values that generate a stable system
 */
Eigen::VectorXd true_prior_expectations() {
  Eigen::VectorXd default_prior_expectation = Eigen::VectorXd::Zero(11);
  default_prior_expectation << 0.6, 1.3, 1, 5, 15, 80, 1.29, 0.5, 0.5, 2, 8;

  return default_prior_expectation;
}

parameter_location_weather default_parameter_locations() {
  parameter_location_weather parameter_locations;
  parameter_locations.ocean_heat_transfer = Eigen::VectorXi(3);
  parameter_locations.ocean_heat_transfer << 0, 1, 2;
  parameter_locations.ocean_heat_capacity = Eigen::VectorXi(3);
  parameter_locations.ocean_heat_capacity << 3, 4, 5;
  parameter_locations.deep_ocean_efficacy = Eigen::VectorXi(1);
  parameter_locations.deep_ocean_efficacy << 6;
  parameter_locations.sigma_eta = Eigen::VectorXi(1);
  parameter_locations.sigma_eta << 7;
  parameter_locations.sigma_xi = Eigen::VectorXi(1);
  parameter_locations.sigma_xi << 8;
  parameter_locations.gamma_autocorrelation = Eigen::VectorXi(1);
  parameter_locations.gamma_autocorrelation << 9;
  parameter_locations.forcing_4co2 = Eigen::VectorXi(1);
  parameter_locations.forcing_4co2 << 10;

  return parameter_locations;
}

/**
 * Prior expectations on position
 */
Eigen::VectorXd default_prior_expectations() {
  Eigen::VectorXd default_prior_expectation = Eigen::VectorXd::Zero(11);
  default_prior_expectation << 0.6 + 0.5, 1.3 - 0.5, 1 + 0.5, 5 - 0.5, 15 + 0.5,
      80 - 0.5, 1.29 + 0.5, 0.5 - 0.5, 0.5 + 0.5, 2 - 0.5, 8 + 0.5;

  return default_prior_expectation;
}

/**
 * Prior covariance matrix
 */
Eigen::MatrixXd default_prior_covariances() {
  double flat = 1.0;                    // flat priors
  // double informative = 1 / (double)16;  // informative priors
  double informative = 1 / (double)2;  // informative priors
  double precise = 1 / (double)256;     // precise priors
  double fixed = 1 / (double)2048;      // fixed priors
  Eigen::VectorXd default_prior_covariance = Eigen::VectorXd::Zero(11);
  default_prior_covariance << informative, informative, informative,
      informative, informative, informative, informative, informative,
      informative, informative, informative;
  // default_prior_covariance << flat, flat, flat, flat, flat, flat, flat, flat,
  //     flat, flat, flat;

  Eigen::MatrixXd return_default_prior_covariance =
      Eigen::MatrixXd::Zero(11, 11);
  return_default_prior_covariance.diagonal() = default_prior_covariance;
  return return_default_prior_covariance;
}

/**
 * Prior hyperparameter expectation vector
 */
Eigen::VectorXd default_hyper_expectations() {
  Eigen::VectorXd default_hyper_expectation = Eigen::VectorXd::Zero(8);
  return default_hyper_expectation;
}
/**
 * Prior hyperparameter covariance matrix
 */
Eigen::MatrixXd default_hyper_covariances() {
  double flat = 1.0;                    // flat priors
  double informative = 1 / (double)16;  // informative priors
  double precise = 1 / (double)256;     // precise priors
  double fixed = 1 / (double)2048;      // fixed priors
  Eigen::MatrixXd default_hyper_covariance = Eigen::MatrixXd::Zero(8, 8);
  default_hyper_covariance.diagonal() << precise, precise, precise, precise,
      precise, precise, precise, precise;
  return default_hyper_covariance;
}
