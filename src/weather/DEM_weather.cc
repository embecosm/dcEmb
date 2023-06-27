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
#include <random>
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

  int start_date = 1750;
  int end_date = 2100;
  int sz = end_date - start_date + 1;
  model.num_samples = sz;

  model.max_invert_it = 512;

  model.num_response_vars = 5;
  model.select_response_vars = (Eigen::VectorXi(5) << 5, 6, 7, 8, 9).finished();
  std::vector<std::string> species_names(
      {"CO2 FFI", "CO2 AFOLU", "CO2", "CH4", "N2O"});

  model.species_list = simple_species_struct(species_names);

  std::vector<Eigen::MatrixXd> ecf =
      simple_ecf(model.species_list, "ssp585", start_date, end_date);

  std::vector<Eigen::MatrixXd> ecf_prior =
      simple_ecf(model.species_list, "ssp126", start_date, end_date);

  model.emissions = ecf.at(0);

  model.emissions.row(0) = model.emissions.row(0) / 1000;
  model.emissions.row(1) = model.emissions.row(1) / 1000;
  model.emissions.row(4) = model.emissions.row(4) / 1000;
  model.emissions.row(2) = model.emissions.row(0) + model.emissions.row(1);

  model.prior_parameter_expectations = default_prior_expectations(
      ecf_prior.at(0)(0, Eigen::seq(250, Eigen::last)));
  model.prior_parameter_covariances = default_prior_covariances(sz - 250);
  model.prior_hyper_expectations = default_hyper_expectations();
  model.prior_hyper_covariances = default_hyper_covariances();
  model.parameter_locations = default_parameter_locations();

  // Scale units of emissions

  model.concentrations = ecf.at(1);
  // model.forcings = ecf.at(2);
  // model.emissions(0, Eigen::all) = Eigen::VectorXd::Ones(sz) * 38;
  // model.emissions(1, Eigen::all) = Eigen::VectorXd::Ones(sz) * 3;
  // model.emissions(2, Eigen::all) = Eigen::VectorXd::Ones(sz) * 100;
  // model.concentrations(3, Eigen::all) = Eigen::VectorXd::Ones(sz) * 1800;
  // model.concentrations(4, Eigen::all) = Eigen::VectorXd::Ones(sz) * 325;
  model.concentrations(Eigen::all, 0) =
      model.species_list.baseline_concentration;
  model.concentrations(0, 0) = 0;
  model.concentrations(1, 0) = 0;
  // for(int i = 0; i < sz; i++)
  // {
  //   model.emissions(0, i) = 38.0/(sz-1) * i;
  //   model.emissions(1, i) = 3.0/(sz-1) * i;
  //   model.emissions(2, i) = 2.2 + (100.0-2.2)/(sz-1) * i;
  //   model.concentrations(3, i) = 729 + (1800.0-729.0)/(sz-1) * i;
  //   model.concentrations(4, i) = 270 + (325.0-270.0)/(sz-1) * i;
  // }
  model.forcings = Eigen::MatrixXd::Zero(model.species_list.name.size(), sz);
  model.temperature = Eigen::MatrixXd::Zero(3, sz);
  model.airborne_emissions =
      Eigen::MatrixXd::Zero(model.species_list.name.size(), sz);
  model.cumulative_emissions =
      Eigen::MatrixXd::Zero(model.species_list.name.size(), sz);

  Eigen::MatrixXd true_out = model.eval_generative(
      true_prior_expectations(model.emissions(0, Eigen::seq(250, Eigen::last))),
      default_parameter_locations(), sz);

  model.response_vars = true_out(Eigen::all, model.select_response_vars);

  model.invert_model();

  int n = 1000;

  Eigen::MatrixXd prior_e_out = model.eval_generative(
      model.prior_parameter_expectations, default_parameter_locations(), sz);

  Eigen::MatrixXd posterior_final_out =
      model.eval_generative(model.conditional_parameter_expectations,
                            default_parameter_locations(), sz);

  Eigen::MatrixXd prior_c_out_m = model.eval_generative(
      model.prior_parameter_expectations, default_parameter_locations(), sz);

  Eigen::MatrixXd prior_rand_out =
      Eigen::MatrixXd(prior_e_out.rows() * n, prior_e_out.cols());

  std::default_random_engine rd;
  std::mt19937 gen(rd());

  for (int i = 0; i < n; i++) {
    Eigen::MatrixXd prior_tmp =
        random_generative(model, model.prior_parameter_expectations,
                          model.prior_parameter_covariances, sz, gen);
    prior_rand_out(Eigen::seqN(i * sz, sz), Eigen::all) = prior_tmp;
  }

  std::ifstream param_expectations_file;
  param_expectations_file.open("param_expecations.csv");
  std::string param_expectations_line;
  std::ifstream param_covariances_file;
  param_covariances_file.open("param_covariances.csv");
  std::string param_covariances_line;

  Eigen::MatrixXd posterior_e_out =
      Eigen::MatrixXd(sz * model.performed_it, posterior_final_out.cols());

  Eigen::MatrixXd posterior_rand_out =
      Eigen::MatrixXd(sz * n * model.performed_it, posterior_final_out.cols());

  int i = 0;
  while (std::getline(param_expectations_file, param_expectations_line)) {
    std::vector<double> values_e;
    std::stringstream lineStream_e(param_expectations_line);
    std::string cell_e;
    while (std::getline(lineStream_e, cell_e, ',')) {
      values_e.push_back(std::stod(cell_e));
    }

    std::vector<double> values_c;
    for (int k = 0; k < values_e.size(); k++) {
      std::getline(param_covariances_file, param_covariances_line);

      std::stringstream lineStream_c(param_covariances_line);
      std::string cell_c;
      while (std::getline(lineStream_c, cell_c, ',')) {
        values_c.push_back(std::stod(cell_c));
      }
    }

    Eigen::VectorXd param_expectations =
        Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(values_e.data(),
                                                      values_e.size());

    Eigen::MatrixXd param_covariances =
        Eigen::Map<Eigen::MatrixXd, Eigen::Unaligned>(
            values_c.data(), values_e.size(), values_e.size());

    // Eigen::MatrixXd test1 = posterior_e_out(Eigen::seqN(i * sz, sz),
    // Eigen::all);
    // Eigen::MatrixXd test2 = model.eval_generative(param_expectations,
    //                           default_parameter_locations(), sz);
    // DEBUG(test1.rows());
    // DEBUG(test1.cols());
    // DEBUG(test2.rows());
    // DEBUG(test2.cols());
    posterior_e_out(Eigen::seqN(i * sz, sz), Eigen::all) =
        model.eval_generative(param_expectations, default_parameter_locations(),
                              sz);
    for (int j = 0; j < n; j++) {
      posterior_rand_out(Eigen::seqN(i * sz * n + (j * sz), sz), Eigen::all) =
          random_generative(model, param_expectations, param_covariances, sz,
                            gen);
    }
    DEBUG(i);
    i++;
  }

  param_expectations_file.close();
  param_covariances_file.close();

  utility::print_matrix("../visualisation/weather/true_generative.csv",
                        true_out);
  utility::print_matrix("../visualisation/weather/prior_generative.csv",
                        prior_e_out);
  utility::print_matrix("../visualisation/weather/prior_generative_rand.csv",
                        prior_rand_out);
  utility::print_matrix("../visualisation/weather/pos_generative.csv",
                        posterior_e_out);
  utility::print_matrix("../visualisation/weather/pos_generative_rand.csv",
                        posterior_rand_out);
  // std::cout << "temperature" << true_out(25, Eigen::all) << '\n';
  return 0;
}

Eigen::MatrixXd random_generative(dynamic_weather_model& model,
                                  Eigen::VectorXd& mean, Eigen::MatrixXd& var,
                                  int& sz, std::mt19937& gen) {
  std::normal_distribution<double> dis(0, 1);

  Eigen::VectorXd rand_param_prior =
      Eigen::VectorXd::Zero(mean.size()).unaryExpr([&](double dummy) {
        return dis(gen);
      });
  Eigen::LLT<Eigen::MatrixXd> lltOfA(var);
  Eigen::MatrixXd L = lltOfA.matrixL();
  rand_param_prior = ((L * rand_param_prior).array() + mean.array()).eval();

  return model.eval_generative(rand_param_prior, default_parameter_locations(),
                               sz);
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
    // There's probably a way to do this elegantly with Eigen::Map to reuse
    // the memory

    for (int i = 0; i < emissions_split.size(); i++) {
      if (emissions_split.at(i).empty()) {
        for (int j = i; j < emissions_split.size(); j++) {
          if (!emissions_split.at(j).empty()) {
            Eigen::VectorXd vec = Eigen::VectorXd::LinSpaced(
                j - i + 2, std::stod(emissions_split.at(i - 1)),
                std::stod(emissions_split.at(j)));
            for (int k = 0; k < vec.size() - 2; k++) {
              emissions_split.at(i + k) = std::to_string(vec(k + 1));
            }
            break;
          }
        }
      }
    }

    if (species.input_mode.at(name_idx) == "emissions") {
      for (int i = 0; i < (end_date - start_date + 1); i++) {
        int pos = start_date - 1750 + 7 + i;
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
    // There's probably a way to do this elegantly with Eigen::Map in a way
    // that reuses the memory
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
    // There's probably a way to do this elegantly with Eigen::Map in a way
    // that reuses the memory
    if (species.input_mode.at(name_idx) == "forcings") {
      for (int i = 0; i < (end_date - start_date + 1); i++) {
        int pos = start_date - 1750 + 7 + i;
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

  // species.input_mode.at(0) = "emissions";
  // species.input_mode.at(1) = "emissions";
  // species.input_mode.at(2) = "emissions";
  // species.input_mode.at(3) = "concentrations";
  // species.input_mode.at(4) = "concentrations";
  // species.input_mode.at(5) = "calculated";

  species.input_mode.at(2) = "calculated";
  // species.greenhouse_gas(3) = 1;
  // species.greenhouse_gas(4) = 1;

  species.unperturbed_lifetime(Eigen::all, 3) =
      Eigen::VectorXd::Ones(species.unperturbed_lifetime.rows()) * 10.8537568;
  species.baseline_emissions(3) = 19.01978312;
  species.baseline_emissions(4) = 0.08602230754;

  // species.baseline_emissions(2) = 0;
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
  species_struct species_out = species;
  // append_species(species, append_species(erfari, erfaci));

  // species_out.forcing_scale = Eigen::VectorXd::Ones(8);
  // species_out.tropospheric_adjustment = Eigen::VectorXd::Zero(8);
  // species_out.aci_scale = Eigen::VectorXd::Ones(8) * -2.09841432;
  utility::update_species_list_indicies(species_out);
  return species_out;
}

/**
 * "True" values that generate a stable system
 */
Eigen::VectorXd true_prior_expectations(Eigen::VectorXd em) {
  Eigen::VectorXd default_prior_expectation = Eigen::VectorXd::Zero(11);
  // default_prior_expectation << 0.6, 1.3, 1, 5, 15, 80, 1.29, 0.5, 0.5, 2,
  // 8;
  default_prior_expectation << 1.876, 5.154, 0.6435, 2.632, 9.262, 52.93, 1.285,
      2.691, 0.4395, 28.24, 8;

  Eigen::VectorXd out_vec = Eigen::VectorXd(em.size() + 11);
  out_vec << default_prior_expectation, em;
  return out_vec;
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
Eigen::VectorXd default_prior_expectations(Eigen::VectorXd em) {
  Eigen::VectorXd default_prior_expectation = Eigen::VectorXd::Zero(11);
  double x = 0;
  default_prior_expectation << 1.876, 5.154, 0.6435, 2.632, 9.262, 52.93, 1.285,
      2.691, 0.4395, 28.24, 8;

  Eigen::VectorXd out_vec = Eigen::VectorXd(em.size() + 11);
  out_vec << default_prior_expectation, em * x;
  return out_vec;
}

/**
 * Prior covariance matrix
 */
Eigen::MatrixXd default_prior_covariances(int sz) {
  double flat = 1.0;                    // flat priors
  double informative = 1 / (double)16;  // informative priors
  double precise = 1 / (double)256;     // precise priors
  double fixed = 1 / (double)2048;      // fixed priors
  Eigen::VectorXd default_prior_covariance = Eigen::VectorXd::Ones(sz + 11);
  default_prior_covariance = default_prior_covariance * informative;
  // default_prior_covariance << flat, flat, flat, flat, flat, flat, flat,
  // flat,
  //     flat, flat, flat;

  Eigen::MatrixXd return_default_prior_covariance =
      Eigen::MatrixXd::Zero(sz + 11, sz + 11);
  return_default_prior_covariance.diagonal() = default_prior_covariance;
  return return_default_prior_covariance;
}

/**
 * Prior hyperparameter expectation vector
 */
Eigen::VectorXd default_hyper_expectations() {
  Eigen::VectorXd default_hyper_expectation = Eigen::VectorXd::Zero(5);
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
  Eigen::MatrixXd default_hyper_covariance = Eigen::MatrixXd::Zero(5, 5);
  default_hyper_covariance.diagonal() << precise, precise, precise, precise,
      precise;
  return default_hyper_covariance;
}
