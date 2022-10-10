/**
 * A set of functions for running the COVID-19 dynamic causal model within the
 * dcEmb package
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

#include "DEM_COVID.hh"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>
#include "Eigen/Dense"
#include "bma_model.hh"
#include "bmr_model.hh"
#include "country_data.hh"
#include "dynamic_COVID_model.hh"
#include "import_COVID.hh"
#include "peb_model.hh"
#include "serialization.hh"
#include "utility.hh"

/**
 * Run the COVID example
 */
int run_COVID_test() {
  bool OCTAVE_INVERSIONS = 0;
  int num_countries = 1;
  std::vector<dynamic_COVID_model> GCM;
  std::vector<country_data> countries = read_country_data(num_countries);

  if (!OCTAVE_INVERSIONS) {
    bool STORE = 0;
    bool LOAD = 0;

    if (!LOAD) {
      for (int i = 0; i < num_countries; i++) {
        country_data country = countries.at(i);
        Eigen::MatrixXd response_vars =
            Eigen::MatrixXd::Zero(country.cases.size(), 2);
        Eigen::VectorXi select_response_vars = Eigen::VectorXi::Zero(2);
        select_response_vars << 0, 1;
        response_vars << country.deaths, country.cases;
        dynamic_COVID_model COVID_model;
        COVID_model.prior_parameter_expectations = default_prior_expectations();
        COVID_model.prior_parameter_covariances = default_prior_covariances();
        COVID_model.prior_hyper_expectations = default_hyper_expectations();
        COVID_model.prior_hyper_covariances = default_hyper_covariances();
        COVID_model.parameter_locations = default_parameter_locations();
        COVID_model.num_samples = countries.at(i).days;
        COVID_model.select_response_vars = select_response_vars;
        COVID_model.num_response_vars = 2;
        COVID_model.response_vars = response_vars;
        COVID_model.max_invert_it = 128;
        COVID_model.invert_model();

        if (STORE) {
          {
            std::ofstream outfile("DCM_COVID_" + country.name);
            cereal::BinaryOutputArchive archive_out(outfile);
            archive_out << COVID_model;
          }
        }
        GCM.push_back(COVID_model);
      }
    } else {
      for (int i = 0; i < num_countries; i++) {
        country_data country = countries.at(i);
        dynamic_COVID_model COVID_model;
        {
          std::ifstream infile("DCM_COVID_" + country.name);
          cereal::BinaryInputArchive archive_in(infile);
          archive_in >> COVID_model;
          GCM.push_back(COVID_model);
        }
      }
    }
  } else {
    GCM = generate_comparison_data(countries);
  }

  return 0;

  // Set up 3rd level design matricies

  Eigen::VectorXd lat = Eigen::VectorXd(num_countries);
  Eigen::VectorXd lon = Eigen::VectorXd(num_countries);
  Eigen::VectorXd pop = Eigen::VectorXd(num_countries);
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(num_countries);
  for (int i = 0; i < num_countries; i++) {
    lat(i) = (countries.at(i).latitude * 2 * M_PI / 360);
    lon(i) = (countries.at(i).longitude * 2 * M_PI / 360);
    pop(i) = log(countries.at(i).pop);
  }
  Eigen::MatrixXd design_matrix_tmp = Eigen::MatrixXd(5, 10);
  design_matrix_tmp.col(0) = ones;
  design_matrix_tmp.col(1) = pop;
  for (int i = 1; i < 5; i++) {
    design_matrix_tmp.col(2 * i) =
        lon.unaryExpr([i](double x) { return sin(i * x); });
    design_matrix_tmp.col(2 * i + 1) =
        lat.unaryExpr([i](double x) { return sin(i * x); });
  }
  design_matrix_tmp = utility::orth(design_matrix_tmp);
  Eigen::MatrixXd design_matrix = Eigen::MatrixXd(5, 5);
  for (int i = 0; i < 5; i++) {
    design_matrix.col(i) =
        design_matrix_tmp.col(i) /
        sqrt(design_matrix_tmp.col(i).array().square().sum());
  }
  design_matrix.col(0) = Eigen::VectorXd::Ones(5);
  peb_model<dynamic_COVID_model> PEB;
  // Current assumption: All models are using the same parameters in the same
  // Locations
  PEB.random_effects = default_random_effects(GCM.at(0).parameter_locations);
  PEB.GCM = GCM;
  PEB.between_design_matrix = design_matrix;
  PEB.max_invert_it = 64;
  PEB.invert_model();

  generate_PEB_values(PEB);

  bmr_model<peb_model<dynamic_COVID_model>> BMR;
  BMR.DCM_in = PEB;
  BMR.reduce();
  bma_model<peb_model<dynamic_COVID_model>> BMA = BMR.BMA;
  utility::print_matrix("BMR.free_energy_vector", BMR.free_energy_vector);
  utility::print_matrix("BMR.posterior_probability_vector",
                        BMR.posterior_probability_vector);
  utility::print_matrix("BMR.model_space", BMR.model_space);

  std::vector<dynamic_COVID_model> GCM_empirical = PEB.empirical_GCM;
  for (int i = 0; i < GCM_empirical.size(); i++) {
    (GCM_empirical[i]).invert_model();
  }

  for (int i = 0; i < num_countries; i++) {
    country_data country = countries.at(i);
    {
      std::ifstream infile("DCM_COVID_" + country.name);
      cereal::BinaryInputArchive archive_in(infile);
      archive_in >> GCM_empirical[i];
    }
  }

  return 0;
}

/**
 * Set default prior expectations vector
 */
Eigen::VectorXd default_prior_expectations() {
  Eigen::VectorXd default_prior_expectation = Eigen::VectorXd::Zero(28);
  default_prior_expectation << log(4),  // number of initial cases
      log(8),                           // size of population with mixing
      -log(4),                          // initial proportion
      -log(3),                          // P(going home | work)
      -log(32),                         // social distancing threshold
      log(16) - log(100000),  // bed availability threshold (per capita)
      log(4),                 // effective number of contacts: home
      log(48),                // effective number of contacts: work
      -log(3),                // P(transmission | infectious)
      log(4),                 // infected (pre-contagious) period
      log(4),                 // contagious period
      log(16),                // time until symptoms
      -log(32),               // P(severe symptoms | symptomatic)
      log(8),                 // symptomatic period
      log(10),                // period in CCU
      -log(2),                // P(fatality | CCU)
      -log(8),                // P(fatality | home)
      -log(10000),            // test, track and trace
      log(2),                 // testing latency (months)
      log(4),                 // test delay (days)
      log(8),                 // test selectivity (for infection)
      log(4) - log(10000),    // sustained testing
      log(4) - log(10000),    // baseline testing
      log(16),                // period of immunity
      log(2),                 // period of exemption
      -log(2),                // proportion of people not susceptible
      -log(2),                // proportion with innate immunity
      log(16);                // testing buildup
  return default_prior_expectation;
}

/**
 * Set default prior covariances. Covariances for each parameter can take 1 of
 * up to 3 different values on the diagonal
 */
Eigen::MatrixXd default_prior_covariances() {
  double flat = 1;                      // flat priors
  double informative = 1 / (double)16;  // informative priors
  double precise = 1 / (double)256;     // precise priors
  Eigen::MatrixXd default_prior_covariance = Eigen::MatrixXd::Zero(28, 28);
  default_prior_covariance.diagonal() << flat,  // number of initial cases
      flat,         // size of population with mixing
      precise,      // initial proportion
      precise,      // P(going home | work)
      precise,      // social distancing threshold
      precise,      // bed availability threshold (per capita)
      informative,  // effective number of contacts: home
      informative,  // effective number of contacts: work
      informative,  // P(transmission | infectious)
      precise,      // infected (pre-contagious) period
      precise,      // contagious period
      precise,      // time until symptoms
      precise,      // P(severe symptoms | symptomatic)
      precise,      // symptomatic period
      precise,      // period in CCU
      precise,      // P(fatality | CCU)
      precise,      // P(fatality | home)
      flat,         // test, track and trace
      flat,         // testing latency (months)
      precise,      // test delay (days)
      informative,  // test selectivity (for infection)
      flat,         // sustained testing
      informative,  // baseline testing
      precise,      // period of immunity
      precise,      // period of exemption
      precise,      // proportion of people not susceptible
      precise,      // proportion with innate immunity
      flat;         // testing buildup
  return default_prior_covariance;
}

parameter_location_COVID default_parameter_locations() {
  parameter_location_COVID parameter_locations;
  parameter_locations.init_cases = 0;   // number of initial cases
  parameter_locations.pop_size = 1;     // size of population with mixing
  parameter_locations.init_prop = 2;    // initial proportion
  parameter_locations.p_home_work = 3;  // P(going home | work)
  parameter_locations.social_dist = 4;  // social distancing threshold
  parameter_locations.bed_thresh =
      5;  // bed availability threshold (per capita)
  parameter_locations.home_contacts = 6;  // effective number of contacts: home
  parameter_locations.work_contacts = 7;  // effective number of contacts: work
  parameter_locations.p_conta_contact = 8;  // P(transmission | infectious)
  parameter_locations.infed_period = 9;     // /infected (pre-contagious) period
  parameter_locations.infious_period = 10;  // contagious period
  parameter_locations.tt_symptoms = 11;     // time until symptoms
  parameter_locations.p_sev_symp = 12;      // P(severe symptoms | symptomatic)
  parameter_locations.symp_period = 13;     // symptomatic period
  parameter_locations.ccu_period = 14;      // period in CCU
  parameter_locations.p_fat_sevccu = 15;    // P(fatality | CCU)
  parameter_locations.p_surv_sevhome = 16;  // P(fatality | home)
  parameter_locations.test_track_trace = 17;  // test, track and trace
  parameter_locations.test_lat = 18;          // testing latency (months)
  parameter_locations.test_del = 19;          // test delay (days)
  parameter_locations.test_selec = 20;    // test selectivity (for infection)
  parameter_locations.subs_testing = 21;  // sustained testing
  parameter_locations.base_testing = 22;  // baseline testing
  parameter_locations.imm_period = 23;    // period of immunity
  parameter_locations.exmp_period = 24;   // period of exemption
  parameter_locations.prop_res = 25;  // proportion of people not susceptible
  parameter_locations.prop_imm = 26;  // proportion with innate immunity
  parameter_locations.test_buildup = 27;  // testing buildup
  return parameter_locations;
}

/**
 * Set initial hyperparameter expectation to zero
 */
Eigen::VectorXd default_hyper_expectations() {
  Eigen::VectorXd default_hyper_expectation = Eigen::VectorXd::Zero(2);
  return default_hyper_expectation;
}

/**
 * Set initial hypermarameter covariances to 1/64
 */
Eigen::MatrixXd default_hyper_covariances() {
  Eigen::MatrixXd default_hyper_covariance = Eigen::MatrixXd::Zero(2, 2);
  default_hyper_covariance.diagonal() << 1.0 / 64.0, 1.0 / 64.0;
  return default_hyper_covariance;
}

Eigen::VectorXi default_random_effects(parameter_location_COVID& pl) {
  Eigen::VectorXi re = Eigen::VectorXi(16);
  re << pl.pop_size, pl.init_prop, pl.p_home_work, pl.social_dist,
      pl.bed_thresh, pl.home_contacts, pl.work_contacts, pl.p_conta_contact,
      pl.infed_period, pl.infious_period, pl.tt_symptoms, pl.p_sev_symp,
      pl.symp_period, pl.ccu_period, pl.p_fat_sevccu, pl.p_surv_sevhome;
  return re;
}

/**
 * Function for testing: populate a set of DCM models with pre-calculated
 * posteriors
 */
std::vector<dynamic_COVID_model> generate_comparison_data(
    std::vector<country_data> countries) {
  std::vector<dynamic_COVID_model> DCM(5);
  for (int i = 0; i < 5; i++) {
    country_data country = countries.at(i);
    Eigen::MatrixXd response_vars =
        Eigen::MatrixXd::Zero(country.cases.size(), 2);
    Eigen::VectorXi select_response_vars = Eigen::VectorXi::Zero(2);
    select_response_vars << 0, 1;
    response_vars << country.deaths, country.cases;
    dynamic_COVID_model& COVID_model = DCM[i];
    COVID_model.prior_parameter_expectations = default_prior_expectations();
    COVID_model.prior_parameter_covariances = default_prior_covariances();
    COVID_model.prior_hyper_expectations = default_hyper_expectations();
    COVID_model.prior_hyper_covariances = default_hyper_covariances();
    COVID_model.parameter_locations = default_parameter_locations();
    COVID_model.num_samples = countries.at(i).days;
    COVID_model.select_response_vars = select_response_vars;
    COVID_model.num_response_vars = 2;
    COVID_model.response_vars = response_vars;
    COVID_model.max_invert_it = 128;
    switch (i) {
      case (0):
        generate_us_posteriors(COVID_model);
        break;
      case (1):
        generate_brazil_posteriors(COVID_model);
        break;
      case (2):
        generate_india_posteriors(COVID_model);
        break;
      case (3):
        generate_russia_posteriors(COVID_model);
        break;
      case (4):
        generate_mexico_posteriors(COVID_model);
        break;
    }
  }
  return DCM;
}

/**
 * Testing function: Generate a set of posteriors for the US
 */
void generate_us_posteriors(dynamic_COVID_model& model) {
  model.conditional_parameter_expectations =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/US_conditional_parameter_expectations.csv");
  model.conditional_parameter_covariances =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/US_conditional_parameter_covariances.csv");
  model.free_energy = -11263.38592908148;
}

/**
 * Testing function: Generate a set of posteriors for Brazil
 */
void generate_brazil_posteriors(dynamic_COVID_model& model) {
  model.conditional_parameter_expectations =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/Brazil_conditional_parameter_expectations.csv");
  model.conditional_parameter_covariances =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/Brazil_conditional_parameter_covariances.csv");
  model.free_energy = -9205.455950138792;
}

/**
 * Testing function: Generate a set of posteriors for India
 */
void generate_india_posteriors(dynamic_COVID_model& model) {
  model.conditional_parameter_expectations =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/India_conditional_parameter_expectations.csv");
  model.conditional_parameter_covariances =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/India_conditional_parameter_covariances.csv");
  model.free_energy = -10481.23075593396;
}

/**
 * Testing function: Generate a set of posteriors for Russia
 */
void generate_russia_posteriors(dynamic_COVID_model& model) {
  model.conditional_parameter_expectations =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/Russia_conditional_parameter_expectations.csv");
  model.conditional_parameter_covariances =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/Russia_conditional_parameter_covariances.csv");
  model.free_energy = -8396.914141674371;
}

/**
 * Testing function: Generate a set of posteriors for Mexico
 */
void generate_mexico_posteriors(dynamic_COVID_model& model) {
  model.conditional_parameter_expectations =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/Mexico_conditional_parameter_expectations.csv");
  model.conditional_parameter_covariances =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/Mexico_conditional_parameter_covariances.csv");
  model.free_energy = -7379.296952775237;
}

/**
 * Testing function: Populate a PEB model with pre-calculated values
 */
void generate_PEB_values(peb_model<dynamic_COVID_model>& model) {
  model.prior_parameter_expectations = utility::read_matrix<Eigen::MatrixXd>(
      "../src/data/peb_prior_parameter_expectations.csv");
  model.prior_parameter_covariances = utility::read_matrix<Eigen::MatrixXd>(
      "../src/data/peb_prior_parameter_covariances.csv");
  model.conditional_parameter_expectations =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/peb_conditional_parameter_expectations.csv");
  model.conditional_parameter_covariances =
      utility::read_matrix<Eigen::MatrixXd>(
          "../src/data/peb_conditional_parameter_covariances.csv");
}