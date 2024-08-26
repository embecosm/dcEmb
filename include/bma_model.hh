/**
 * A class for performcing Bayesian Model Averaging for the dcEmb package
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

#include <eigen3/Eigen/Dense>
#include "utility.hh"

#pragma once

/**
 * Bayesian Model Averaging Class. Sample from a group of DCM models to produce
 * An estimate of the average effect across all models.
 */
template <typename dynamic_model_template>
class bma_model {
 public:
  /**
   * @brief Input posterior posterior expectations
   */
  std::vector<Eigen::MatrixXd> posterior_p_e_all;
  /**
   * @brief Input posterior posterior covariances
   */
  std::vector<Eigen::MatrixXd> posterior_p_c_all;
  /**
   * @brief Input free energies
   */
  Eigen::VectorXd free_energy_all;
  /**
   * @brief Number of samples to take from averaged posterior
   */
  int num_samples = 1000;
  /**
   * @brief Accumulated free energy across averaged models
   */
  Eigen::VectorXd accumulated_free_energy;
  /**
   * @brief Accumulated posterior model probability across averaged models
   */
  Eigen::VectorXd posterior_model_prob;
  /**
   * @brief Output average posterior expectations
   */
  Eigen::VectorXd posterior_p_e;
  /**
   * @brief Output average posterior expectations
   */
  Eigen::VectorXd posterior_p_c;
  /**
   * @brief Output individual posterior expectations
   */
  std::vector<Eigen::MatrixXd> posterior_p_e_indiv;
  /**
   * @brief Output individual posterior covariances
   */
  std::vector<Eigen::MatrixXd> posterior_p_c_indiv;

  /**
   * Calculate a "fixed effects" average across all the
   * Currently assumes:
   *  - Only one "session"
   *  - Only one "subject"
   *  - All models are included in occams window (oddsr = 0)
   *
   */
  void average_ffx() {
    int num_models = free_energy_all.size();
    int num_parameters =
        posterior_p_e_all.at(0).rows() * posterior_p_e_all.at(0).cols();
    double max_free_energy = this->free_energy_all.maxCoeff();
    Eigen::VectorXd free_energy_all_tmp = free_energy_all.unaryExpr(
        [max_free_energy](double x) { return x - max_free_energy; });
    Eigen::VectorXd free_energy_all_exp =
        free_energy_all_tmp.unaryExpr([](double x) { return exp(x); });
    double free_energy_all_exp_sum = free_energy_all_exp.sum();
    this->posterior_model_prob =
        free_energy_all_exp.unaryExpr([free_energy_all_exp_sum](double x) {
          return x / free_energy_all_exp_sum;
        });
    this->accumulated_free_energy = free_energy_all_tmp;

    Eigen::VectorXd posterior_model_prob_tmp = this->posterior_model_prob;
    double max_post_prob = posterior_model_prob_tmp.maxCoeff();

    Eigen::MatrixXd posterior_p_e_tmp =
        Eigen::MatrixXd(num_parameters, num_samples);
    for (int i = 0; i < num_samples; i++) {
      int rand_model = utility::selrnd(this->posterior_model_prob);
      Eigen::EigenSolver<Eigen::MatrixXd> eig_solver(
          this->posterior_p_c_all.at(rand_model));
      Eigen::VectorXd model_posterior_eval_c_e =
          eig_solver.eigenvalues().real();
      Eigen::MatrixXd model_posterior_evec_c_e =
          eig_solver.eigenvectors().real();

      Eigen::VectorXd mu =
          posterior_p_e_all.at(rand_model)
              .reshaped(posterior_p_e_all.at(rand_model).rows() *
                            posterior_p_e_all.at(rand_model).cols(),
                        1);
      Eigen::VectorXd tmp = utility::normrnd(mu, model_posterior_eval_c_e,
                                             model_posterior_evec_c_e);

      posterior_p_e_tmp.col(i) = tmp;
    }

    this->posterior_p_e = posterior_p_e_tmp.rowwise().mean();
    this->posterior_p_c = Eigen::VectorXd(num_parameters);
    for (int i = 0; i < num_parameters; i++) {
      this->posterior_p_c(i) = sqrt(
          (posterior_p_e_tmp.row(i).array() - posterior_p_e_tmp.row(i).mean())
              .square()
              .sum() /
          (posterior_p_e_tmp.size() - 1));
    }
  }
};