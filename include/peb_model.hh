/**
 * A base class for parametric empirical Bayes (PEB) functions within the dcEmb
 * package
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

#include <stdio.h>
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>
#include "dynamic_model.hh"
#include "utility.hh"

#pragma once

/**
 * PEB Class. This class specifies a hierarchical DCM, with "between subject"
 * constraints on the list of DCMs (GCM) in the design matrix "random effects".
 * The purpose of this class is to provide a method to optimise the
 * empirical priors of the first level models using these second level
 * constraints.
 */
template <typename dynamic_model_template>
class peb_model : public dynamic_model {
 public:
  /**
   * @brief std::vector of input DCMs
   */
  std::vector<dynamic_model_template> GCM;
  /**
   * @brief vector indicating which parameters to treat as random effects
   */
  Eigen::VectorXi random_effects;
  /**
   * @brief Design matrix of second level (between subject) parameters
   */
  Eigen::MatrixXd between_design_matrix;
  /**
   * @brief Design matrix of second level (within subject) parameters
   */
  Eigen::MatrixXd within_design_matrix;
  /**
   * @brief Design matrix of second level (within subject) parameters
   */
  Eigen::VectorXd random_precision_comps;
  Eigen::MatrixXd prior_parameter_covariances;
  /**
   * @brief Posterior expectations of second level parameters
   */
  Eigen::MatrixXd conditional_parameter_expectations;
  /**
   * @brief Posterior expectations of second level "hyper"-parameters
   */
  Eigen::MatrixXd conditional_hyper_covariances;
  /**
   * @brief Expected covariance of second level random effects
   */
  Eigen::MatrixXd expected_covariance_random_effects;
  /**
   * @brief Matrix used for reduced calculations
   */
  Eigen::MatrixXd singular_matrix;
  /**
   * @brief Posterior precisions of second level random effects
   */
  Eigen::MatrixXd precision_random_effects;
  /**
   * @brief Output DCMs with empirical priors set
   */
  std::vector<dynamic_model_template> empirical_GCM;

  /**
   * Invert a PEB model, producing estimates of optimized empirical priors
   * on the given DCMs. Note that this method requires that the DCMs
   * passed in have already been estimated.
   * Currently assumes:
   *  - no precision components are passed in, defaulting to one precision
   *  component
   *  - no priors on second_level parameters are passed in, defaulting to
   *  mean values of first level priors
   *  - num_subjects > 1
   *  - alpha and beta parameters not set, defaulting to 1 and 16.
   */
  void invert_model() {
    int num_models = this->GCM.size();
    int num_random_effects = this->random_effects.size();
    int num_precision_comps = 1;
    int num_second_level = num_models * num_random_effects;
    int num_between_effects = between_design_matrix.cols();
    this->empirical_GCM = this->GCM;
    int alpha = 1;
    int beta = 16;

    Eigen::VectorXd mean_prior_p_e = Eigen::VectorXd::Zero(num_random_effects);
    Eigen::MatrixXd mean_prior_p_c =
        Eigen::MatrixXd::Zero(num_random_effects, num_random_effects);
    for (int i = 0; i < num_models; i++) {
      mean_prior_p_e +=
          this->GCM.at(i).prior_parameter_expectations(this->random_effects);
      mean_prior_p_c += this->GCM.at(i).prior_parameter_covariances(
          this->random_effects, this->random_effects);
    }
    mean_prior_p_e = mean_prior_p_e / num_models;
    mean_prior_p_c = mean_prior_p_c / num_models;

    Eigen::BDCSVD<Eigen::MatrixXd> svd;
    svd.setThreshold(std::numeric_limits<double>::epsilon() * 64);
    svd.compute(mean_prior_p_c, Eigen::ComputeThinV | Eigen::ComputeThinU);
    Eigen::MatrixXd singular_vec = svd.matrixU();

    std::vector<Eigen::VectorXd> perDCM_prior_p_e;
    std::vector<Eigen::MatrixXd> perDCM_prior_p_c;
    std::vector<Eigen::VectorXd> perDCM_posterior_p_e;
    std::vector<Eigen::MatrixXd> perDCM_posterior_p_c;
    for (int i = 0; i < num_models; i++) {
      Eigen::VectorXd p_p_e =
          singular_vec.transpose() *
          this->GCM.at(i).prior_parameter_expectations(this->random_effects);
      Eigen::MatrixXd p_p_c = singular_vec.transpose() *
                              this->GCM.at(i).prior_parameter_covariances(
                                  this->random_effects, this->random_effects) *
                              singular_vec;
      Eigen::VectorXd c_p_e =
          singular_vec.transpose() *
          this->GCM.at(i).conditional_parameter_expectations(
              this->random_effects);
      Eigen::MatrixXd c_p_c = singular_vec.transpose() *
                              this->GCM.at(i).conditional_parameter_covariances(
                                  this->random_effects, this->random_effects) *
                              singular_vec;

      c_p_c = utility::inverse_tol(utility::inverse_tol(c_p_c) +
                                   (utility::inverse_tol(p_p_c) / 16));
      perDCM_prior_p_e.push_back(p_p_e);
      perDCM_prior_p_c.push_back(p_p_c);
      perDCM_posterior_p_e.push_back(c_p_e);
      perDCM_posterior_p_c.push_back(c_p_c);
    }

    this->within_design_matrix =
        Eigen::MatrixXd::Identity(num_random_effects, num_random_effects);

    Eigen::VectorXd tmp_prior_b_e = mean_prior_p_e;
    Eigen::MatrixXd tmp_prior_b_c = mean_prior_p_c / alpha;
    Eigen::MatrixXd prior_p_c = mean_prior_p_c / beta;

    Eigen::MatrixXd prior_p_q =
        singular_vec.transpose() * prior_p_c * singular_vec;
    prior_p_q = utility::inverse_tol(prior_p_q);

    Eigen::VectorXd kron_b_e =
        Eigen::MatrixXd::Identity(num_between_effects, num_between_effects)
            .col(0);
    Eigen::MatrixXd kron_b_c =
        Eigen::MatrixXd::Zero(num_between_effects, num_between_effects);
    for (int i = 0; i < num_between_effects; i++) {
      kron_b_c(i, i) = between_design_matrix.rows() /
                       between_design_matrix.col(i).array().pow(2).sum();
    }

    Eigen::VectorXd prior_b_e =
        kroneckerProduct(kron_b_e, singular_vec.transpose() * tmp_prior_b_e);
    Eigen::MatrixXd prior_b_c = kroneckerProduct(
        kron_b_c, (singular_vec.transpose() * tmp_prior_b_c * singular_vec));
    Eigen::MatrixXd prior_b_q = utility::inverse_tol(prior_b_c);

    Eigen::VectorXd prior_g_e = Eigen::MatrixXd(1, 1);
    prior_g_e << 0;
    Eigen::MatrixXd prior_g_c = Eigen::MatrixXd(1, 1);
    prior_g_c << 1.0 / 16;
    Eigen::MatrixXd prior_g_q = utility::inverse_tol(prior_g_c);
    Eigen::MatrixXd prior_bg_q =
        Eigen::MatrixXd::Zero(prior_b_q.rows() + prior_g_q.rows(),
                              prior_b_q.cols() + prior_g_q.cols());
    prior_bg_q(Eigen::seq(0, prior_b_q.rows() - 1),
               Eigen::seq(0, prior_b_q.cols() - 1)) = prior_b_q;
    prior_bg_q(
        Eigen::seq(prior_b_q.rows(), prior_b_q.rows() + prior_g_q.rows() - 1),
        Eigen::seq(prior_b_q.cols(), prior_b_q.cols() + prior_g_q.cols() - 1)) =
        prior_g_q;

    Eigen::VectorXd b_estimate = prior_b_e;
    Eigen::VectorXd g_estimate = prior_g_e;
    // Finally, we are ready to start inversion
    Eigen::MatrixXd prior_r_p_q = prior_p_q;
    Eigen::MatrixXd prior_r_p_c;
    std::vector<Eigen::MatrixXd> perDCM_prior_r_p_e(num_models);
    Eigen::MatrixXd posterior_p_c;
    Eigen::MatrixXd posterior_b_c;
    Eigen::MatrixXd posterior_g_c;
    double free_energy_var;
    Eigen::VectorXd current_b_estimate;
    Eigen::VectorXd current_g_estimate;
    double current_free_energy;
    Eigen::VectorXd current_dFdb;
    Eigen::MatrixXd current_dFdbb;
    Eigen::VectorXd current_dFdg;
    Eigen::MatrixXd current_dFdgg;
    double t = -4;
    double dF = 0;
    double Fc = 0;
    for (int i = 0; i < this->max_invert_it; i++) {
      if (num_precision_comps > 0) {
        prior_r_p_q = prior_p_q * exp(-8);
        for (int j = 0; j < num_precision_comps; j++) {
          prior_r_p_q += exp(g_estimate(j)) * prior_p_q;
        }
      }
      prior_r_p_c = utility::inverse_tol(prior_r_p_q);

      double free_energy_tmp = 0;

      Eigen::VectorXd dFdb = -prior_b_q * (b_estimate - prior_b_e);
      Eigen::MatrixXd dFdbb = -prior_b_q;
      Eigen::VectorXd dFdg = -prior_g_q * (g_estimate - prior_g_e);
      Eigen::MatrixXd dFdgg = -prior_g_q;
      Eigen::MatrixXd dFdbg =
          Eigen::MatrixXd::Zero(num_second_level, num_precision_comps);

      for (int j = 0; j < num_models; j++) {
        Eigen::MatrixXd design_matrix_kron = kroneckerProduct(
            this->between_design_matrix.row(j), this->within_design_matrix);
        perDCM_prior_r_p_e[j] = design_matrix_kron * b_estimate;
        std::vector<Eigen::MatrixXd> log_evidence =
            utility::reduced_log_evidence(
                perDCM_posterior_p_e[j], perDCM_posterior_p_c[j],
                perDCM_prior_p_e[j], perDCM_prior_p_c[j], perDCM_prior_r_p_e[j],
                prior_r_p_c);
        double reduced_free_energy = log_evidence[0].value();
        Eigen::VectorXd posterior_r_p_e = log_evidence[1];
        Eigen::MatrixXd posterior_r_p_c = log_evidence[2];
        free_energy_tmp =
            free_energy_tmp + reduced_free_energy + this->GCM.at(j).free_energy;
        Eigen::VectorXd dE = posterior_r_p_e - perDCM_prior_r_p_e[j];
        dFdb = dFdb + design_matrix_kron.transpose() * prior_r_p_q * dE;
        dFdbb = dFdbb + design_matrix_kron.transpose() *
                            (prior_r_p_q * posterior_r_p_c * prior_r_p_q -
                             prior_r_p_q) *
                            design_matrix_kron;

        for (int k = 0; k < num_precision_comps; k++) {
          double g_estimate_tmp = g_estimate(k);
          double dFdgj0 = exp(g_estimate_tmp);
          double dFdgj1 = (prior_p_q * (prior_r_p_c - posterior_r_p_c)).trace();
          double dFdgj2 = (dE.transpose() * prior_p_q * dE);
          double dFdgj = dFdgj0 * (dFdgj1 - dFdgj2) / 2;
          dFdg(k) = dFdg(k) + dFdgj;
          dFdgg(k, k) = dFdgg(k, k) + dFdgj;

          Eigen::MatrixXd dFdbgj =
              dFdgj0 *
              (design_matrix_kron -
               (posterior_r_p_c * prior_r_p_q * design_matrix_kron))
                  .transpose() *
              prior_p_q * dE;
          dFdbg.col(k) = dFdbg.col(k) + dFdbgj;
          for (int m = 0; m < num_precision_comps; m++) {
            double g_estimate_tmp2 = g_estimate(m);
            double dFdggj0 = exp(g_estimate_tmp + g_estimate_tmp2);
            double dFdggj1 =
                (((prior_r_p_c * prior_p_q * prior_r_p_c) -
                  (posterior_r_p_c * prior_p_q * posterior_r_p_c)) *
                 prior_p_q)
                    .trace() /
                2;
            double dFdggj2 =
                dE.transpose() * prior_p_q * posterior_r_p_c * prior_p_q * dE;
            double dFdggj = dFdggj0 * (dFdggj1 - dFdggj2);
            dFdgg(k, m) = dFdgg(k, m) - dFdggj;
          }
        }
      }
      Eigen::VectorXd dFdp = Eigen::VectorXd(dFdb.size() + dFdg.size());
      dFdp(Eigen::seq(0, dFdb.size() - 1)) = dFdb;
      dFdp(Eigen::seq(dFdb.size(), dFdb.size() + dFdg.size() - 1)) = dFdg;
      Eigen::MatrixXd dFdpp = Eigen::MatrixXd(dFdbb.rows() + dFdgg.rows(),
                                              dFdbb.cols() + dFdgg.cols());
      dFdpp(Eigen::seq(0, dFdbb.rows() - 1), Eigen::seq(0, dFdbb.cols() - 1)) =
          dFdbb;

      // Eigen Vectors are column vectors by default
      dFdpp(Eigen::seq(dFdbb.rows(), dFdbb.rows() + dFdbg.cols() - 1),
            Eigen::seq(0, dFdbg.rows() - 1)) = dFdbg.transpose();
      dFdpp(Eigen::seq(0, dFdbb.rows() - 1),
            Eigen::seq(dFdbb.cols(), dFdbb.cols() + dFdbg.cols() - 1)) = dFdbg;
      dFdpp(Eigen::seq(dFdbb.rows(), dFdbb.rows() + dFdgg.rows() - 1),
            Eigen::seq(dFdbb.cols(), dFdbb.cols() + dFdgg.cols() - 1)) = dFdgg;
      posterior_p_c = utility::inverse_tol(-dFdpp);
      posterior_b_c = utility::inverse_tol(-dFdbb);
      posterior_g_c = utility::inverse_tol(-dFdgg);

      double Fb = b_estimate.transpose() * prior_b_q * b_estimate;
      double Fg = g_estimate.transpose() * prior_g_q * g_estimate;
      Fc = Fb / 2 + Fg / 2 - utility::logdet(prior_bg_q * posterior_p_c) / 2;

      free_energy_tmp = free_energy_tmp - Fc;

      if (i == 0) {
        free_energy_var = free_energy_tmp;
      }

      if (free_energy_tmp >= free_energy_var) {
        dF = free_energy_tmp - free_energy_var;
        free_energy_var = free_energy_tmp;
        current_b_estimate = b_estimate;
        current_g_estimate = g_estimate;
        current_free_energy = free_energy_var;
        current_dFdb = dFdb;
        current_dFdbb = dFdbb;
        current_dFdg = dFdg;
        current_dFdgg = dFdgg;
        t = std::min(t + 0.25, 2.0);
      } else {
        b_estimate = current_b_estimate;
        g_estimate = current_g_estimate;
        free_energy_var = current_free_energy;
        dFdb = current_dFdb;
        dFdbb = current_dFdbb;
        dFdg = current_dFdg;
        dFdgg = current_dFdgg;
        t = std::max(t - 1, -4.0);
      }

      Eigen::VectorXd dp = utility::dx(dFdpp, dFdp, t);

      if (dp.norm() >= 8) {
        dFdpp = Eigen::MatrixXd::Zero(dFdbb.rows() + dFdgg.rows(),
                                      dFdbb.cols() + dFdgg.cols());
        dFdpp(Eigen::seq(0, dFdbb.rows() - 1),
              Eigen::seq(0, dFdbb.cols() - 1)) = dFdbb;
        dFdpp(Eigen::seq(dFdbb.rows(), dFdbb.rows() + dFdgg.rows() - 1),
              Eigen::seq(dFdbb.cols(), dFdbb.cols() + dFdgg.cols() - 1)) =
            dFdgg;
        dp = utility::dx(dFdpp, dFdp, t);
      }

      Eigen::VectorXd db = dp(Eigen::seq(0, b_estimate.size() - 1));
      Eigen::VectorXd dg = dp(Eigen::seq(
          b_estimate.size(), b_estimate.size() + g_estimate.size() - 1));
      b_estimate = b_estimate + db;
      g_estimate = g_estimate + dg.unaryExpr([](double x) { return tanh(x); });
      std::cout << "VL Iteration " << i << ": F = " << free_energy_tmp
                << " dF: " << dF << " [" << t << "]" << '\n';
      if ((i > 4) && ((t <= -4) | dF < 1e-4)) {
        break;
      }
    }
    for (int i = 0; i < num_models; i++) {
      dynamic_model_template& DCM = empirical_GCM[i];
      Eigen::MatrixXd prior_e_p_c = GCM.at(i).prior_parameter_covariances;
      prior_e_p_c = utility::inverse_tol(prior_e_p_c);
      prior_e_p_c(this->random_effects, this->random_effects) =
          singular_vec * prior_r_p_q * singular_vec.transpose();
      prior_e_p_c = utility::inverse_tol(prior_e_p_c);

      Eigen::VectorXd prior_e_p_e = GCM.at(i).prior_parameter_expectations;
      //   std::cout << "perDCM_prior_r_p_e[i]" << perDCM_prior_r_p_e[i] <<
      //   '\n'; std::cout << "GCM.at(i).prior_parameter_expectations"
      //             << GCM.at(i).prior_parameter_expectations << '\n';
      prior_e_p_e(this->random_effects) = singular_vec * perDCM_prior_r_p_e[i];

      Eigen::MatrixXd tmp_prior_p_c =
          Eigen::MatrixXd::Zero(GCM.at(i).prior_parameter_covariances.size(),
                                GCM.at(i).prior_parameter_covariances.size());
      tmp_prior_p_c = GCM.at(i).prior_parameter_covariances;
      std::vector<Eigen::MatrixXd> log_evidence = utility::reduced_log_evidence(
          GCM.at(i).conditional_parameter_expectations,
          GCM.at(i).conditional_parameter_covariances,
          GCM.at(i).prior_parameter_expectations, tmp_prior_p_c, prior_e_p_e,
          prior_e_p_c);

      DCM.prior_parameter_expectations = prior_e_p_e;
      DCM.prior_parameter_covariances = prior_e_p_c;
      DCM.conditional_parameter_expectations = log_evidence[1];
      DCM.conditional_parameter_covariances = log_evidence[2];
      DCM.free_energy =
          log_evidence[0].value() + GCM.at(i).free_energy + Fc / num_models;
    }
    Eigen::MatrixXd kron_singular_vec = kroneckerProduct(
        Eigen::MatrixXd::Identity(num_between_effects, num_between_effects),
        singular_vec);
    this->singular_matrix = singular_vec;

    this->prior_parameter_expectations =
        kroneckerProduct(kron_b_e, tmp_prior_b_e);
    this->prior_parameter_covariances =
        kroneckerProduct(kron_b_c, tmp_prior_b_c);
    this->prior_hyper_expectations = prior_g_e;
    this->prior_hyper_covariances = prior_g_c;
    this->conditional_parameter_expectations =
        singular_vec *
        b_estimate.reshaped(num_random_effects, num_between_effects);
    this->conditional_parameter_covariances =
        kron_singular_vec * posterior_b_c * kron_singular_vec.transpose();
    this->conditional_hyper_expectations = g_estimate;
    this->conditional_hyper_covariances = posterior_g_c;
    this->expected_covariance_random_effects =
        singular_vec * prior_r_p_c * singular_vec.transpose();
    this->free_energy = current_free_energy;
    this->precision_random_effects = prior_p_q;

    return;
  }

  peb_model() { return; };
};

template <typename dynamic_model_template>
inline bool operator==(const peb_model<dynamic_model_template>& lhs,
                       const peb_model<dynamic_model_template>& rhs) {
  return lhs.GCM == rhs.GCM & lhs.empirical_GCM == rhs.empirical_GCM &
         lhs.random_effects == rhs.random_effects &
         lhs.between_design_matrix == rhs.between_design_matrix &
         lhs.within_design_matrix == rhs.within_design_matrix &
         lhs.random_precision_comps == rhs.random_precision_comps &
         lhs.expected_covariance_random_effects ==
             rhs.expected_covariance_random_effects &
         lhs.singular_matrix == rhs.singular_matrix &
         lhs.precision_random_effects == rhs.precision_random_effects &
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
