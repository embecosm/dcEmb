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

/**
 * Run the weather example
 */
int run_weather_test() {
  dynamic_weather_model model;
  // Add Sulphur?
  std::vector<std::string> species_names({"CO2 FFI", "CO2 AFOLU", "Sulfur",
                                         "CO2",     "CH4",       "N2O"});
  model.species_list = simple_species_struct(species_names);
  // model.species_list.input_mode[2] = "concentrations";
  model.species_list.input_mode[3] = "emissions";
  model.airborne_emissions = 0;
  model.ecf = simple_ecf(model.species_list, "ssp119");
  // model.eval_generative(true_prior_expectations(),
  // default_parameter_locations);

  return 0;
}

// Fill emissions, concentrations and/or forcings
Eigen::MatrixXd simple_ecf(species_struct species, std::string scenario) {
  std::cout << "names";k
  for(int i = 0; i < species.name.size(); i++)
  {
    std::cout << " " << species.name[i];
  }
  std::cout << '\n';
  int start_date = 1701;
  int end_date = 1710;
  Eigen::MatrixXd out_matrix =
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
        out_matrix(name_idx, i) =
            std::stod(emissions_split.at(start_date - 1700 + 7 + i));
      }
      std::cout << "" << emissions_split.at(3) << '\n';
      std::cout << name_idx << '\n';
      std::cout << out_matrix(name_idx, Eigen::all) << '\n';
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
        out_matrix(name_idx, i) =
            std::stod(concentrations_split.at(start_date - 1700 + 7 + i));
      }
      std::cout << "" << concentrations_split.at(3) << '\n';
      std::cout << name_idx << '\n';
      std::cout << out_matrix(name_idx, Eigen::all) << '\n';
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
        out_matrix(name_idx, i) =
            std::stod(forcings_split.at(start_date - 1700 + 7 + i));
      }
      std::cout << "" << forcings_split.at(3) << '\n';
      std::cout << name_idx << '\n';
      std::cout << out_matrix(name_idx, Eigen::all) << '\n';
    }
  }
  std::cout << '\n' << "out_matrix" << '\n' << out_matrix << '\n';
  return out_matrix;
}

species_struct simple_species_struct(std::vector<std::string> species_names) {
  std::string filename = "../src/weather/data/species_configs_properties.csv";
  species_struct species = utility::species_from_file(filename, species_names);
  species_struct erfari(1);
  erfari.name.at(0) = "ERFari";
  erfari.type.at(0) = "ari";
  erfari.input_mode.at(0) = "calculated";
  erfari.greenhouse_gas << 0;
  erfari.aerosol_chemistry_from_emissions << 0;
  erfari.aerosol_chemistry_from_concentration << 0;

  species_struct erfaci(1);
  erfaci.name.at(0) = "ERFaci";
  erfaci.type.at(0) = "aci";
  erfaci.input_mode.at(0) = "calculated";
  erfaci.greenhouse_gas << 0;
  erfaci.aerosol_chemistry_from_emissions << 0;
  erfaci.aerosol_chemistry_from_concentration << 0;

  return append_species(species, append_species(erfari, erfaci));
}

/**
 * "True" values that generate a stable system
 */
Eigen::VectorXd true_prior_expectations() {
  Eigen::MatrixXd default_prior_expectation = Eigen::MatrixXd::Zero(14, 3);
  default_prior_expectation.row(0) << 0.6, 1.3, 1;
  default_prior_expectation.row(1) << 1.1, 1.6, 0.9;
  default_prior_expectation.row(2) << 1.7, 2.0, 1.1;
  default_prior_expectation.row(3) << 5, 15, 80;
  default_prior_expectation.row(4) << 8, 14, 100;
  default_prior_expectation.row(5) << 6, 11, 75;
  default_prior_expectation.row(6) << 1.29, 1.1, 0.8;
  default_prior_expectation.row(7) << 0.5, 0.5, 0.5;
  default_prior_expectation.row(8) << 0.5, 0.5, 0.5;
  default_prior_expectation.row(9) << 2, 2, 2;
  default_prior_expectation.row(10) << 8, 8, 8;

  Eigen::Map<Eigen::VectorXd> return_default_prior_expectation(
      default_prior_expectation.data(),
      default_prior_expectation.rows() * default_prior_expectation.cols());
  return return_default_prior_expectation;
}

parameter_location_weather default_parameter_locations() {
  parameter_location_weather parameter_locations;
  parameter_locations.ocean_heat_transfer << 0, 1, 2, 3, 4, 5, 6, 7, 8;
  parameter_locations.ocean_heat_capacity << 9, 10, 11, 12, 13, 14, 15, 16, 17;
  parameter_locations.deep_ocean_efficacy << 18, 19, 20;
  parameter_locations.sigma_eta << 21, 22, 23;
  parameter_locations.sigma_xi << 24, 25, 26;
  parameter_locations.gamma_autocorrelation << 27, 28, 29;
  parameter_locations.forcing_4co2 << 30, 31, 32;

  return parameter_locations;
}

/**
 * Prior expectations on position
 */
Eigen::VectorXd default_prior_expectations() {
  Eigen::MatrixXd default_prior_expectation = Eigen::MatrixXd::Zero(7, 3);
  double x = 0.04;
  default_prior_expectation.row(0) << 1 - x, 1 + x, 1 + x;
  default_prior_expectation.row(1) << 0.97000436 + x, -0.97000436 - x, 0 + x;
  default_prior_expectation.row(2) << -0.24308753 + x, 0.24308753 + x, 0 - x;
  default_prior_expectation.row(3) << 0 + x, 0 + x, 0 - x;
  default_prior_expectation.row(4) << 0.93240737 / 2 + x, 0.93240737 / 2 - x,
      -0.93240737 + x;
  default_prior_expectation.row(5) << 0.86473146 / 2 + x, 0.86473146 / 2 - x,
      -0.86473146 - x;
  default_prior_expectation.row(6) << 0 + x, 0 - x, 0 + x;
  Eigen::Map<Eigen::VectorXd> return_default_prior_expectation(
      default_prior_expectation.data(),
      default_prior_expectation.rows() * default_prior_expectation.cols());
  return return_default_prior_expectation;
}

/**
 * Prior covariance matrix
 */
Eigen::MatrixXd default_prior_covariances() {
  double flat = 1.0;                    // flat priors
  double informative = 1 / (double)16;  // informative priors
  double precise = 1 / (double)256;     // precise priors
  double fixed = 1 / (double)2048;      // fixed priors
  Eigen::MatrixXd default_prior_covariance = Eigen::MatrixXd::Zero(7, 3);
  default_prior_covariance.row(0) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(1) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(2) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(3) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(4) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(5) = Eigen::VectorXd::Constant(3, informative);
  default_prior_covariance.row(6) = Eigen::VectorXd::Constant(3, informative);
  Eigen::Map<Eigen::VectorXd> default_prior_covariance_diag(
      default_prior_covariance.data(),
      default_prior_covariance.rows() * default_prior_covariance.cols());
  Eigen::MatrixXd return_default_prior_covariance =
      Eigen::MatrixXd::Zero(21, 21);
  return_default_prior_covariance.diagonal() = default_prior_covariance_diag;
  return return_default_prior_covariance;
}

/**
 * Prior hyperparameter expectation vector
 */
Eigen::VectorXd default_hyper_expectations() {
  Eigen::VectorXd default_hyper_expectation = Eigen::VectorXd::Zero(3);
  return default_hyper_expectation;
}
/**
 * Prior hyperparameter covariance matrix
 */
Eigen::MatrixXd default_hyper_covariances() {
  Eigen::MatrixXd default_hyper_covariance = Eigen::MatrixXd::Zero(3, 3);
  default_hyper_covariance.diagonal() << 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0;
  return default_hyper_covariance;
}
