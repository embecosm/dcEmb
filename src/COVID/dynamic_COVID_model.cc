/**
 * The COVID-19 dynamic causal model class within the dcEmb package
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

#include "dynamic_COVID_model.hh"
#include "utility.hh"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/SVD"

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <random>
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>
#define FSGEN
#define PL parameter_locations
#define ZERO(a, b) Eigen::MatrixXd::Zero(a, b)
#define SparseMD Eigen::SparseMatrix<double>
#define SparseVD Eigen::SparseVector<double>
/**
 * Observed outcomes for the COVID problem.
 */
Eigen::VectorXd dynamic_COVID_model::get_observed_outcomes() {
  Eigen::MatrixXd fs_mat = eval_features(this->response_vars);
  Eigen::Map<Eigen::VectorXd> observed_outcomes(fs_mat.data(),
                                                fs_mat.rows() * fs_mat.cols());
  return observed_outcomes;
}

/**
 * Return the wrapped forward model for the COVID problem
 */
std::function<Eigen::VectorXd(Eigen::VectorXd)>
dynamic_COVID_model::get_forward_model_function() {
  std::function<Eigen::VectorXd(Eigen::VectorXd)> forward_model = std::bind(
      &dynamic_COVID_model::forward_model, this, std::placeholders::_1,
      this->parameter_locations, this->num_samples, this->select_response_vars);
  return forward_model;
}

/**
 * Returns the forward model for the COVID problem
 */
Eigen::VectorXd dynamic_COVID_model::forward_model(
    const Eigen::VectorXd& parameters,
    const parameter_location_COVID& parameter_locations,
    const int& timeseries_length, const Eigen::VectorXi& select_response_vars) {
  Eigen::MatrixXd gen = eval_generative(
      parameters, parameter_locations, timeseries_length, select_response_vars);
  Eigen::MatrixXd fs_output = eval_features(gen);
  Eigen::Map<Eigen::VectorXd> output(fs_output.data(),
                                     fs_output.rows() * fs_output.cols());
  return output;
}

/**
 * Feature selection for the COVID model
 */
Eigen::MatrixXd dynamic_COVID_model::eval_features(
    Eigen::MatrixXd response_vars) {
  Eigen::MatrixXd ones_mat =
      Eigen::MatrixXd::Ones(response_vars.rows(), response_vars.cols());
  return (response_vars).array().max(ones_mat.array()).sqrt();
}

/**
 * Evaluate the generative model for the COVID example
 */
Eigen::MatrixXd dynamic_COVID_model::eval_generative(
    const Eigen::VectorXd& parameters,
    const parameter_location_COVID& parameter_locations,
    const int& timeseries_length) {
  Eigen::VectorXd marginal_location = Eigen::VectorXd::Zero(5);
  Eigen::VectorXd marginal_infection = Eigen::VectorXd::Zero(5);
  Eigen::VectorXd marginal_clinical = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd marginal_testing = Eigen::VectorXd::Zero(4);
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(timeseries_length, 10);
  // Declare new objects that will change, keeping old ones intact
  Eigen::VectorXd parameters_base = parameters;
  Eigen::VectorXd parameters_exp_in = parameters.array().exp();
  Eigen::VectorXd parameters_exp = parameters_exp_in;

  // Macro shortens parameter_locations as PL for readability
  double num_initial = parameters_exp(PL.init_cases);
  double pop_size_val = 1000000 * parameters_exp(PL.pop_size);
  double num_seques_cases = pop_size_val * (1 - parameters_exp(PL.init_prop));
  double num_resis_cases = pop_size_val * parameters_exp(PL.prop_res);
  double num_sucep_cases = pop_size_val - num_initial - num_resis_cases;
  double num_home = (pop_size_val - num_seques_cases) * 0.75;
  double num_work = (pop_size_val - num_seques_cases) * 0.25;

  // Initial transition probabilities
  Eigen::VectorXd location = Eigen::VectorXd::Zero(5);
  location << num_home, num_work, 0, num_seques_cases, 0;
  location = location / location.sum();
  Eigen::VectorXd infection = Eigen::VectorXd::Zero(5);
  infection << num_sucep_cases, num_initial, 0, 0, num_resis_cases;
  infection = infection / infection.sum();
  Eigen::VectorXd clinical = Eigen::VectorXd::Zero(4);
  clinical << 1, 0, 0, 0;
  Eigen::VectorXd testing = Eigen::VectorXd::Zero(4);
  testing << 1, 0, 0, 0;
  // Chained Kronecker Products on a set of column vectors produces a column
  // vector of their products
  Eigen::VectorXi size_order = Eigen::VectorXi::Zero(4);
  size_order << 5, 5, 4, 4;
  Eigen::VectorXd ensemble_density = kroneckerProduct(
      testing,
      kroneckerProduct(clinical, kroneckerProduct(infection, location)));

  // Ensemble density is dense, because it tends to be fully populated, and
  // sparse/dense multuplication can be vectorised, but sparse/sparse cannot.
  for (int i = 0; i < timeseries_length; i++) {
    parameters_exp(PL.base_testing) =
        (parameters_exp_in(PL.base_testing) +
         (parameters_exp_in(PL.subs_testing) *
          utility::phi((i + 1 - 32 * parameters_exp_in(PL.test_lat)) /
                       parameters_exp_in(PL.test_buildup))));
    SparseMD transition_probability_matrix =
        eval_transition_probability_matrix(parameters_exp, ensemble_density);
    ensemble_density =
        (transition_probability_matrix * ensemble_density).eval();
    ensemble_density = (ensemble_density / ensemble_density.sum()).eval();
    marginal_location =
        utility::calculate_marginal_vector(ensemble_density, size_order, 0);
    marginal_infection =
        utility::calculate_marginal_vector(ensemble_density, size_order, 1);
    marginal_clinical =
        utility::calculate_marginal_vector(ensemble_density, size_order, 2);
    marginal_testing =
        utility::calculate_marginal_vector(ensemble_density, size_order, 3);
    output(i, 0) = pop_size_val * marginal_clinical(3);
    output(i, 1) = pop_size_val * marginal_testing(2);
    output(i, 2) = pop_size_val * marginal_location(2);
    output(i, 3) = marginal_infection(1) + marginal_infection(2);
    output(i, 4) = marginal_infection(3) * 100;
    output(i, 5) = pop_size_val * (marginal_testing(2) + marginal_testing(3));
    output(i, 6) = 100 * (1 - pow(1 - (parameters_exp(PL.p_conta_contact) *
                                       marginal_infection(2)),
                                  15));
    output(i, 7) = marginal_infection(2) * 100;
    output(i, 8) = pop_size_val * ensemble_density.coeffRef(5, 0);
  }
  (output).col(3) =
      (PL.infious_period *
       utility::gradient((output).col(3).array().log().matrix().eval()))
          .array()
          .exp()
          .matrix();
  return output;
}
/**
 * Evaluate the generative model for the COVID example
 */
Eigen::MatrixXd dynamic_COVID_model::eval_generative(
    const Eigen::VectorXd& parameters,
    const parameter_location_COVID& parameter_locations,
    const int& timeseries_length, const Eigen::VectorXi& select_response_vars) {
  Eigen::MatrixXd output =
      eval_generative(parameters, parameter_locations, timeseries_length);

  return output(Eigen::all, select_response_vars);
}

/*
Evaluate the transition probability matrix
 */
SparseMD dynamic_COVID_model::eval_transition_probability_matrix(
    const Eigen::VectorXd& parameters_exp_in,
    const Eigen::VectorXd& ensemble_density) {
  Eigen::VectorXd parameter_exp = parameters_exp_in;
  // Upper bound probabilities
  parameter_exp(PL.p_home_work) = std::min(parameter_exp(PL.p_home_work), 1.0);
  parameter_exp(PL.p_conta_contact) =
      std::min(parameter_exp(PL.p_conta_contact), 1.0);
  parameter_exp(PL.p_sev_symp) = std::min(parameter_exp(PL.p_sev_symp), 1.0);
  parameter_exp(PL.p_fat_sevccu) =
      std::min(parameter_exp(PL.p_fat_sevccu), 1.0);
  parameter_exp(PL.p_surv_sevhome) =
      std::min(parameter_exp(PL.p_surv_sevhome), 1.0);

  SparseMD loc_transition_matrix =
      calc_location_transition_matrix(parameter_exp, ensemble_density);
  SparseMD inf_transition_matrix =
      calc_infection_transition_matrix(parameter_exp, ensemble_density);
  SparseMD cli_transition_matrix =
      calc_clinical_transition_matrix(parameter_exp);
  SparseMD test_transition_matrix =
      calc_testing_transition_matrix(parameter_exp, ensemble_density);

  SparseMD transition_matrix = loc_transition_matrix * inf_transition_matrix *
                               cli_transition_matrix * test_transition_matrix;
  return transition_matrix;
}

/**
 * Calculate the transition matrix for the location parameters
 */
SparseMD dynamic_COVID_model::calc_location_transition_matrix(
    const Eigen::VectorXd& parameter_exp,
    const Eigen::VectorXd& ensemble_density) {
  Eigen::VectorXi size_order = Eigen::VectorXi::Zero(4);
  size_order << 5, 5, 4, 4;
  Eigen::MatrixXd loc_inf_tmp =
      utility::calculate_marginal_vector(ensemble_density, size_order, 0, 1);
  Eigen::MatrixXd loc_inf_mat = loc_inf_tmp(Eigen::seq(0, 2), Eigen::all);
  loc_inf_mat = loc_inf_mat / loc_inf_mat.sum();
  double prev_infec = loc_inf_mat(Eigen::all, 1).sum();
  double CCU_occu = loc_inf_mat(2, Eigen::all).sum();
  double social_dist_val =
      utility::sigma(prev_infec, parameter_exp(PL.social_dist));
  double home_work_val = social_dist_val * parameter_exp(PL.p_home_work);
  double crit_care = utility::sigma(CCU_occu, parameter_exp(PL.bed_thresh));
  double iso_period = exp(-0.1);
  double viral_spread =
      exp(-social_dist_val * prev_infec / parameter_exp(PL.exmp_period));

  double Kday = exp(-1);

  Eigen::MatrixXd loc_asym = Eigen::MatrixXd::Zero(5, 5);
  loc_asym << (1 - home_work_val), 1, 1, (1 - viral_spread), (1 - iso_period),
      home_work_val, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, viral_spread, 0, 0, 0,
      0, 0, iso_period;

  Eigen::MatrixXd loc_sym = Eigen::MatrixXd::Zero(5, 5);
  loc_sym << 0, 0, 0, (1 - viral_spread), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, viral_spread, 0, 1, 1, 1, 0, 1;

  Eigen::MatrixXd loc_ards = Eigen::MatrixXd::Zero(5, 5);
  loc_ards << 0, 0, 0, (1 - viral_spread), 0, 0, 0, 0, 0, 0, crit_care,
      crit_care, 1, 0, crit_care, 0, 0, 0, viral_spread, 0, (1 - crit_care),
      (1 - crit_care), 0, 0, (1 - crit_care);

  Eigen::MatrixXd loc_dec = Eigen::MatrixXd::Zero(5, 5);
  loc_dec << 0, 0, 0, (1 - viral_spread), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
      1, viral_spread, 1, 0, 0, 0, 0, 0;

  Eigen::MatrixXd loc_tmp = Eigen::MatrixXd::Zero(20, 20);

  loc_tmp << loc_asym, ZERO(5, 5), ZERO(5, 5), ZERO(5, 5), ZERO(5, 5), loc_sym,
      ZERO(5, 5), ZERO(5, 5), ZERO(5, 5), ZERO(5, 5), loc_ards, ZERO(5, 5),
      ZERO(5, 5), ZERO(5, 5), ZERO(5, 5), loc_dec;

  Eigen::VectorXi new_order = Eigen::VectorXi::Zero(4);
  new_order << 0, 2, 1, 3;
  Eigen::VectorXi sizes = Eigen::VectorXi::Zero(4);
  sizes << 5, 4, 5, 4;

  SparseMD loc_full = utility::calc_permuted_kron_identity_product(
      20, loc_tmp.sparseView(), new_order, sizes);

  // Change sizes to reflect new order
  sizes << 5, 5, 4, 4;
  Eigen::VectorXi r_pos = Eigen::VectorXi::Zero(4);
  Eigen::VectorXi c_pos = Eigen::VectorXi::Zero(4);

  // Stop isolating if asymptomatic and negative
  for (int i = 0; i < 5; i++) {
    r_pos << 0, i, 0, 3;
    c_pos << 4, i, 0, 3;
    loc_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                      utility::find_kron_position(c_pos, sizes)) = 1;
  }
  for (int i = 0; i < 5; i++) {
    r_pos << 4, i, 0, 3;
    c_pos << 4, i, 0, 3;
    loc_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                      utility::find_kron_position(c_pos, sizes)) = 0;
  }
  // Isolate if positive
  for (int i = 0; i < 5; i++) {
    r_pos << 4, i, 0, 2;
    c_pos << 0, i, 0, 2;
    loc_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                      utility::find_kron_position(c_pos, sizes)) = 1;
  }
  for (int i = 0; i < 5; i++) {
    r_pos << 0, i, 0, 2;
    c_pos << 0, i, 0, 2;
    loc_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                      utility::find_kron_position(c_pos, sizes)) = 0;
  }
  for (int i = 0; i < 5; i++) {
    r_pos << 1, i, 0, 2;
    c_pos << 0, i, 0, 2;
    loc_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                      utility::find_kron_position(c_pos, sizes)) = 0;
  }
  // isolate if infected
  r_pos << 4, 1, 0, 0;
  c_pos << 0, 1, 0, 0;
  loc_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                    utility::find_kron_position(c_pos, sizes)) =
      parameter_exp(PL.test_track_trace);
  r_pos << 0, 1, 0, 0;
  c_pos << 0, 1, 0, 0;
  loc_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                    utility::find_kron_position(c_pos, sizes)) =
      (1 - home_work_val) * (1 - parameter_exp(PL.test_track_trace));
  r_pos << 1, 1, 0, 0;
  c_pos << 0, 1, 0, 0;
  loc_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                    utility::find_kron_position(c_pos, sizes)) =
      home_work_val * (1 - parameter_exp(PL.test_track_trace));

  return loc_full;
}
/**
 * Calculate the transition matrix for the infection parameters
 */
SparseMD dynamic_COVID_model::calc_infection_transition_matrix(
    const Eigen::VectorXd& parameter_exp,
    const Eigen::VectorXd& ensemble_density) {
  Eigen::VectorXi size_order = Eigen::VectorXi::Zero(4);
  size_order << 5, 5, 4, 4;
  Eigen::MatrixXd loc_inf_mat =
      utility::calculate_marginal_vector(ensemble_density, size_order, 0, 1);

  Eigen::VectorXd p_inf_home = loc_inf_mat.row(0);
  p_inf_home = p_inf_home / p_inf_home.sum();
  Eigen::VectorXd p_inf_work = loc_inf_mat.row(1);
  p_inf_work = p_inf_work / p_inf_work.sum();
  double p_trans_home =
      pow((1 - parameter_exp(PL.p_conta_contact) * p_inf_home(2)),
          parameter_exp(PL.home_contacts));
  double p_trans_work =
      pow((1 - parameter_exp(PL.p_conta_contact) * p_inf_work(2)),
          parameter_exp(PL.work_contacts));
  double imm_loss_rate = exp(-1 / parameter_exp(PL.imm_period) / 32);
  double prop_imm_val = parameter_exp(PL.prop_imm);
  double inf_rate = exp(-1 / parameter_exp(PL.infed_period));
  double con_rate = exp(-1 / parameter_exp(PL.infious_period));

  Eigen::MatrixXd inf_home = Eigen::MatrixXd::Zero(5, 5);
  inf_home << p_trans_home, 0, 0, (1 - imm_loss_rate), 0, (1 - p_trans_home),
      inf_rate, 0, 0, 0, 0, (1 - prop_imm_val) * (1 - inf_rate), con_rate, 0, 0,
      0, 0, (1 - con_rate), imm_loss_rate, 0, 0, prop_imm_val * (1 - inf_rate),
      0, 0, 1;

  Eigen::MatrixXd inf_work = Eigen::MatrixXd::Zero(5, 5);
  inf_work << p_trans_work, 0, 0, (1 - imm_loss_rate), 0, (1 - p_trans_work),
      inf_rate, 0, 0, 0, 0, (1 - prop_imm_val) * (1 - inf_rate), con_rate, 0, 0,
      0, 0, (1 - con_rate), imm_loss_rate, 0, 0, prop_imm_val * (1 - inf_rate),
      0, 0, 1;

  Eigen::MatrixXd inf_hosp = Eigen::MatrixXd::Zero(5, 5);
  inf_hosp << 1, 0, 0, (1 - imm_loss_rate), 0, 0, inf_rate, 0, 0, 0, 0,
      (1 - prop_imm_val) * (1 - inf_rate), con_rate, 0, 0, 0, 0, (1 - con_rate),
      imm_loss_rate, 0, 0, prop_imm_val * (1 - inf_rate), 0, 0, 1;

  Eigen::MatrixXd inf_remo = inf_hosp;

  Eigen::MatrixXd inf_iso = inf_hosp;

  Eigen::MatrixXd inf_tmp = Eigen::MatrixXd::Zero(25, 25);
  inf_tmp << inf_home, ZERO(5, 5), ZERO(5, 5), ZERO(5, 5), ZERO(5, 5),
      ZERO(5, 5), inf_work, ZERO(5, 5), ZERO(5, 5), ZERO(5, 5), ZERO(5, 5),
      ZERO(5, 5), inf_hosp, ZERO(5, 5), ZERO(5, 5), ZERO(5, 5), ZERO(5, 5),
      ZERO(5, 5), inf_remo, ZERO(5, 5), ZERO(5, 5), ZERO(5, 5), ZERO(5, 5),
      ZERO(5, 5), inf_iso;

  Eigen::VectorXi new_order = Eigen::VectorXi::Zero(4);
  new_order << 1, 0, 2, 3;

  Eigen::VectorXi sizes = Eigen::VectorXi::Zero(4);
  sizes << 5, 5, 4, 4;

  SparseMD inf_full = utility::calc_permuted_kron_identity_product(
      16, inf_tmp.sparseView(), new_order, sizes);

  return inf_full;
}
/**
 * Calculate the transition matrix for the clinical parameters
 */
SparseMD dynamic_COVID_model::calc_clinical_transition_matrix(
    const Eigen::VectorXd& parameter_exp) {
  double p_sev_symp_val = parameter_exp(PL.p_sev_symp);
  double accu_symp_rate = exp(-1 / parameter_exp(PL.symp_period));
  double ards_rate = exp(-1 / parameter_exp(PL.ccu_period));
  double symp_rate = exp(-1 / parameter_exp(PL.tt_symptoms));
  double p_fatal = 1 - parameter_exp(PL.p_surv_sevhome);
  double Kday = exp(-1);

  Eigen::MatrixXd cli_sus = Eigen::MatrixXd::Zero(4, 4);
  cli_sus << 1, (1 - accu_symp_rate), (1 - ards_rate) * (1 - p_fatal),
      (1 - Kday), 0, accu_symp_rate, 0, 0, 0, 0, ards_rate, 0, 0, 0,
      (1 - ards_rate) * p_fatal, Kday;

  Eigen::MatrixXd cli_infed = Eigen::MatrixXd::Zero(4, 4);
  cli_infed << symp_rate, (1 - accu_symp_rate) * (1 - p_sev_symp_val),
      (1 - ards_rate) * (1 - p_fatal), (1 - Kday), (1 - symp_rate),
      accu_symp_rate, 0, 0, 0, (1 - accu_symp_rate) * p_sev_symp_val, ards_rate,
      0, 0, 0, (1 - ards_rate) * p_fatal, Kday;

  Eigen::MatrixXd cli_infos = cli_infed;
  Eigen::MatrixXd cli_abp = cli_sus;
  Eigen::MatrixXd cli_abn = cli_sus;

  Eigen::MatrixXd cli_tmp = Eigen::MatrixXd::Zero(20, 20);
  cli_tmp << cli_sus, ZERO(4, 4), ZERO(4, 4), ZERO(4, 4), ZERO(4, 4),
      ZERO(4, 4), cli_infed, ZERO(4, 4), ZERO(4, 4), ZERO(4, 4), ZERO(4, 4),
      ZERO(4, 4), cli_infos, ZERO(4, 4), ZERO(4, 4), ZERO(4, 4), ZERO(4, 4),
      ZERO(4, 4), cli_abp, ZERO(4, 4), ZERO(4, 4), ZERO(4, 4), ZERO(4, 4),
      ZERO(4, 4), cli_abn;

  Eigen::VectorXi new_order = Eigen::VectorXi::Zero(4);
  new_order << 2, 1, 0, 3;
  Eigen::VectorXi sizes = Eigen::VectorXi::Zero(4);
  sizes << 4, 5, 5, 4;

  SparseMD cli_full = utility::calc_permuted_kron_identity_product(
      20, cli_tmp.sparseView(), new_order, sizes);
  // Change sizes to reflect new order
  sizes << 5, 5, 4, 4;
  Eigen::VectorXi r_pos = Eigen::VectorXi::Zero(4);
  Eigen::VectorXi c_pos = Eigen::VectorXi::Zero(4);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      r_pos << 2, i, 3, j;
      c_pos << 2, i, 2, j;
      cli_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                        utility::find_kron_position(c_pos, sizes)) =
          (1 - ards_rate) * parameter_exp(PL.p_fat_sevccu);
    }
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      r_pos << 2, i, 0, j;
      c_pos << 2, i, 2, j;
      cli_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                        utility::find_kron_position(c_pos, sizes)) =
          (1 - ards_rate) * (1 - parameter_exp(PL.p_fat_sevccu));
    }
  }

  return cli_full;
}

/**
 * Calculate the transition matrix for the testing parameters
 */
SparseMD dynamic_COVID_model::calc_testing_transition_matrix(
    const Eigen::VectorXd& parameter_exp,
    const Eigen::VectorXd& ensemble_density) {
  Eigen::VectorXi size_order = Eigen::VectorXi::Zero(4);
  size_order << 5, 5, 4, 4;
  Eigen::MatrixXd loc_inf_tmp =
      utility::calculate_marginal_vector(ensemble_density, size_order, 0, 1);
  Eigen::MatrixXd loc_inf_mat = loc_inf_tmp(Eigen::seq(0, 2), Eigen::all);
  loc_inf_mat = loc_inf_mat / loc_inf_mat.sum();
  Eigen::MatrixXd loc_mat =
      utility::calculate_marginal_vector(ensemble_density, size_order, 0);

  double prev_infec = loc_inf_mat(Eigen::all, 1).sum();
  double base_testing_val = parameter_exp(PL.base_testing) * (1 - loc_mat(3));
  double sens_i = 0.9;
  double sens_c = 0.95;
  double p_test = base_testing_val /
                  (1 - prev_infec + parameter_exp(PL.test_selec) * prev_infec);
  double p_test_inf = p_test * parameter_exp(PL.test_selec);
  double Kdel = exp(-1 / parameter_exp(PL.test_del));
  double Kday = exp(-1);

  Eigen::MatrixXd test_sus = Eigen::MatrixXd::Zero(4, 4);
  test_sus << (1 - p_test), 0, (1 - Kday), (1 - Kday), p_test, Kdel, 0, 0, 0, 0,
      Kday, 0, 0, (1 - Kdel), 0, Kday;

  Eigen::MatrixXd test_infed = Eigen::MatrixXd::Zero(4, 4);
  test_infed << (1 - p_test_inf), 0, (1 - Kday), (1 - Kday), p_test_inf, Kdel,
      0, 0, 0, sens_i * (1 - Kdel), Kday, 0, 0, (1 - sens_i) * (1 - Kdel), 0,
      Kday;

  Eigen::MatrixXd test_infos = Eigen::MatrixXd::Zero(4, 4);
  test_infos << (1 - p_test_inf), 0, (1 - Kday), (1 - Kday), p_test_inf, Kdel,
      0, 0, 0, sens_c * (1 - Kdel), Kday, 0, 0, (1 - sens_c) * (1 - Kdel), 0,
      Kday;

  Eigen::MatrixXd test_abp = test_sus;
  Eigen::MatrixXd test_abn = test_sus;

  Eigen::MatrixXd test_tmp = Eigen::MatrixXd::Zero(20, 20);
  test_tmp << test_sus, ZERO(4, 4), ZERO(4, 4), ZERO(4, 4), ZERO(4, 4),
      ZERO(4, 4), test_infed, ZERO(4, 4), ZERO(4, 4), ZERO(4, 4), ZERO(4, 4),
      ZERO(4, 4), test_infos, ZERO(4, 4), ZERO(4, 4), ZERO(4, 4), ZERO(4, 4),
      ZERO(4, 4), test_abp, ZERO(4, 4), ZERO(4, 4), ZERO(4, 4), ZERO(4, 4),
      ZERO(4, 4), test_abn;

  Eigen::VectorXi new_order = Eigen::VectorXi::Zero(4);
  new_order << 2, 1, 3, 0;
  Eigen::VectorXi sizes = Eigen::VectorXi::Zero(4);
  sizes << 4, 5, 5, 4;

  SparseMD test_full = utility::calc_permuted_kron_identity_product(
      20, test_tmp.sparseView(), new_order, sizes);

  sizes << 5, 5, 4, 4;
  Eigen::VectorXi r_pos = Eigen::VectorXi::Zero(4);
  Eigen::VectorXi c_pos = Eigen::VectorXi::Zero(4);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      r_pos << 3, i, j, 0;
      c_pos << 3, i, j, 0;
      test_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                         utility::find_kron_position(c_pos, sizes)) = 1;
    }
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 4; j++) {
      r_pos << 3, i, j, 1;
      c_pos << 3, i, j, 0;
      test_full.coeffRef(utility::find_kron_position(r_pos, sizes),
                         utility::find_kron_position(c_pos, sizes)) = 0;
    }
  }
  return test_full;
}

/**
 * Dynamic Causal Model constructor for the COVID problem
 */
dynamic_COVID_model::dynamic_COVID_model() { return; }