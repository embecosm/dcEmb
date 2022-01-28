/**
 * Tests of serialization functions for the dcEmb package
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

#include "tests/serialization_test.hh"
#include <gtest/gtest.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>
#include "3body/dynamic_3body_model.hh"
#include "COVID/dynamic_COVID_model.hh"
#include "bmr_model.hh"
#include "country_data.hh"
#include "import_COVID.hh"
#include "peb_model.hh"
#include "serialization.hh"

TEST(serialization, 3body) {
  dynamic_3body_model model;
  model.prior_parameter_expectations = default_3body_prior_expectations();
  model.prior_parameter_covariances = default_3body_prior_covariances();
  model.prior_hyper_expectations = default_3body_hyper_expectations();
  model.prior_hyper_covariances = default_3body_hyper_covariances();
  model.parameter_locations = default_3body_parameter_locations();
  model.num_samples = 1000;
  model.num_response_vars = 3;
  feature_selection_3body fs;
  model.fs_function = fs;
  generative_3body gen;
  model.gen_function = gen;
  gen.eval_generative(true_3body_prior_expectations(),
                      model.parameter_locations, model.num_samples);
  Eigen::MatrixXd true_output = gen.get_output();
  Eigen::MatrixXd response_vars =
      Eigen::MatrixXd::Zero(model.num_samples, model.num_response_vars);
  Eigen::VectorXi select_response_vars =
      Eigen::VectorXi::Zero(model.num_response_vars);
  select_response_vars << 1, 2, 3;
  response_vars = gen.get_output_column(select_response_vars);
  model.select_response_vars = select_response_vars;
  model.response_vars = response_vars;
  model.max_invert_it = 2;
  model.invert_model();

  // Make sure oarchive is destroyed before iarchive is instantiated
  {
    std::ofstream outfile("3body_ser_test");
    cereal::BinaryOutputArchive archive_out(outfile);
    archive_out << model;
  }
  dynamic_3body_model model2;
  {
    std::ifstream infile("3body_ser_test");
    cereal::BinaryInputArchive archive_in(infile);
    archive_in >> model2;
  }

  EXPECT_EQ(model, model2);
}

TEST(serialization, COVID) {
  int num_countries = 5;
  std::vector<country_data> countries = read_country_data(num_countries);
  std::vector<dynamic_COVID_model> GCM;
  for (int i = 0; i < num_countries; i++) {
    country_data country = countries.at(i);
    Eigen::MatrixXd response_vars =
        Eigen::MatrixXd::Zero(country.cases.size(), 2);
    Eigen::VectorXi select_response_vars = Eigen::VectorXi::Zero(2);
    select_response_vars << 0, 1;
    response_vars << country.deaths, country.cases;
    dynamic_COVID_model COVID_model;
    COVID_model.prior_parameter_expectations =
        default_COVID_prior_expectations();
    COVID_model.prior_parameter_covariances = default_COVID_prior_covariances();
    COVID_model.prior_hyper_expectations = default_COVID_hyper_expectations();
    COVID_model.prior_hyper_covariances = default_COVID_hyper_covariances();
    COVID_model.parameter_locations = default_COVID_parameter_locations();
    COVID_model.num_samples = countries.at(i).days;
    COVID_model.select_response_vars = select_response_vars;
    COVID_model.num_response_vars = 2;
    COVID_model.response_vars = response_vars;
    feature_selection_COVID COVID_fs;
    COVID_model.fs_function = COVID_fs;
    generative_COVID COVID_gen;
    COVID_model.gen_function = COVID_gen;
    COVID_model.max_invert_it = 2;
    COVID_model.invert_model();
    GCM.push_back(COVID_model);

    // Make sure oarchive is destroyed before iarchive is instantiated
    {
      std::ofstream outfile("COVID_ser_test");
      cereal::BinaryOutputArchive archive_out(outfile);
      archive_out << COVID_model;
    }
    dynamic_COVID_model model2;
    {
      std::ifstream infile("COVID_ser_test");
      cereal::BinaryInputArchive archive_in(infile);
      archive_in >> model2;
    }
    EXPECT_EQ(COVID_model, model2);
  }
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
  PEB.random_effects =
      default_COVID_random_effects(GCM.at(0).parameter_locations);
  PEB.GCM = GCM;
  PEB.between_design_matrix = design_matrix;
  PEB.max_invert_it = 64;
  PEB.invert_model();

  // Make sure oarchive is destroyed before iarchive is instantiated
  //   {
  //     std::ofstream outfile("COVID_ser_test_peb");
  //     cereal::BinaryOutputArchive archive_out(outfile);
  //     archive_out << PEB;
  //   }
  //   peb_model<dynamic_COVID_model> model2;
  //   {
  //     std::ifstream infile("COVID_ser_test_peb");
  //     cereal::BinaryInputArchive archive_in(infile);
  //     archive_in >> PEB;
  //   }
  //   EXPECT_EQ(PEB, model2);

  bmr_model<peb_model<dynamic_COVID_model>> BMR;
  BMR.DCM_in = PEB;
  BMR.reduce();
  bma_model<peb_model<dynamic_COVID_model>> BMA = BMR.BMA;

  std::vector<dynamic_COVID_model> GCM_empirical = PEB.empirical_GCM;
  for (int i = 0; i < GCM_empirical.size(); i++) {
    (GCM_empirical[i]).invert_model();
  }
}

Eigen::VectorXi default_COVID_random_effects(parameter_location_COVID& pl) {
  Eigen::VectorXi re = Eigen::VectorXi(16);
  re << pl.pop_size, pl.init_prop, pl.p_home_work, pl.social_dist,
      pl.bed_thresh, pl.home_contacts, pl.work_contacts, pl.p_conta_contact,
      pl.infed_period, pl.infious_period, pl.tt_symptoms, pl.p_sev_symp,
      pl.symp_period, pl.ccu_period, pl.p_fat_sevccu, pl.p_surv_sevhome;
  return re;
}

Eigen::VectorXd true_3body_prior_expectations() {
  Eigen::MatrixXd default_prior_expectation = Eigen::MatrixXd::Zero(7, 3);
  default_prior_expectation.row(0) << 1, 1, 1;
  default_prior_expectation.row(1) << 0.97000436, -0.97000436, 0;
  default_prior_expectation.row(2) << -0.24308753, 0.24308753, 0;
  default_prior_expectation.row(3) << 0, 0, 0;
  default_prior_expectation.row(4) << 0.93240737 / 2, 0.93240737 / 2,
      -0.93240737;
  default_prior_expectation.row(5) << 0.86473146 / 2, 0.86473146 / 2,
      -0.86473146;
  default_prior_expectation.row(6) << 0, 0, 0;
  Eigen::Map<Eigen::VectorXd> return_default_prior_expectation(
      default_prior_expectation.data(),
      default_prior_expectation.rows() * default_prior_expectation.cols());
  return return_default_prior_expectation;
}

Eigen::VectorXd default_3body_prior_expectations() {
  Eigen::MatrixXd default_prior_expectation = Eigen::MatrixXd::Zero(7, 3);
  default_prior_expectation.row(0) << 0.95, 1.05, 1.05;
  default_prior_expectation.row(1) << 0.97000436 + 0.05, -0.97000436 - 0.05, 0;
  default_prior_expectation.row(2) << -0.24308753 + 0.05, 0.24308753 + 0.05, 0;
  default_prior_expectation.row(3) << 0, 0, 0;
  default_prior_expectation.row(4) << 0.93240737 / 2 + 0.05,
      0.93240737 / 2 - 0.05, -0.93240737 + 0.05;
  default_prior_expectation.row(5) << 0.86473146 / 2 + 0.05,
      0.86473146 / 2 - 0.05, -0.86473146 - 0.05;
  default_prior_expectation.row(6) << 0, 0, 0;
  Eigen::Map<Eigen::VectorXd> return_default_prior_expectation(
      default_prior_expectation.data(),
      default_prior_expectation.rows() * default_prior_expectation.cols());
  return return_default_prior_expectation;
}

Eigen::MatrixXd default_3body_prior_covariances() {
  double flat = 1.0;                    // flat priors
  double informative = 1 / (double)16;  // informative priors
  double precise = 1 / (double)256;     // precise priors
  double fixed = 1 / (double)2048;      // precise priors
  Eigen::MatrixXd default_prior_covariance = Eigen::MatrixXd::Zero(7, 3);
  default_prior_covariance.row(0) = Eigen::VectorXd::Constant(3, precise);
  default_prior_covariance.row(1) = Eigen::VectorXd::Constant(3, precise);
  default_prior_covariance.row(2) = Eigen::VectorXd::Constant(3, precise);
  default_prior_covariance.row(3) = Eigen::VectorXd::Constant(3, fixed);
  default_prior_covariance.row(4) = Eigen::VectorXd::Constant(3, precise);
  default_prior_covariance.row(5) = Eigen::VectorXd::Constant(3, precise);
  default_prior_covariance.row(6) = Eigen::VectorXd::Constant(3, fixed);
  Eigen::Map<Eigen::VectorXd> default_prior_covariance_diag(
      default_prior_covariance.data(),
      default_prior_covariance.rows() * default_prior_covariance.cols());
  Eigen::MatrixXd return_default_prior_covariance =
      Eigen::MatrixXd::Zero(21, 21);
  return_default_prior_covariance.diagonal() = default_prior_covariance_diag;
  return return_default_prior_covariance;
}

parameter_location_3body default_3body_parameter_locations() {
  parameter_location_3body parameter_locations;
  parameter_locations.planet_masses = 0;
  parameter_locations.planet_coordsX = 0;
  parameter_locations.planet_coordsY = 1;
  parameter_locations.planet_coordsZ = 2;
  parameter_locations.planet_velocityX = 4;
  parameter_locations.planet_velocityY = 5;
  parameter_locations.planet_velocityZ = 6;
  return parameter_locations;
}

Eigen::VectorXd default_3body_hyper_expectations() {
  Eigen::VectorXd default_hyper_expectation = Eigen::VectorXd::Zero(3);
  return default_hyper_expectation;
}
Eigen::MatrixXd default_3body_hyper_covariances() {
  Eigen::MatrixXd default_hyper_covariance = Eigen::MatrixXd::Zero(3, 3);
  default_hyper_covariance.diagonal() << 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0;
  return default_hyper_covariance;
}

Eigen::VectorXd default_COVID_prior_expectations() {
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

Eigen::MatrixXd default_COVID_prior_covariances() {
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

parameter_location_COVID default_COVID_parameter_locations() {
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

Eigen::VectorXd default_COVID_hyper_expectations() {
  Eigen::VectorXd default_hyper_expectation = Eigen::VectorXd::Zero(2);
  return default_hyper_expectation;
}
Eigen::MatrixXd default_COVID_hyper_covariances() {
  Eigen::MatrixXd default_hyper_covariance = Eigen::MatrixXd::Zero(2, 2);
  default_hyper_covariance.diagonal() << 1.0 / 64.0, 1.0 / 64.0;
  return default_hyper_covariance;
}