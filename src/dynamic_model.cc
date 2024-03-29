/**
 * A base class for dynamic causal models within the dcEmb package
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

#include "dynamic_model.hh"
#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include "utility.hh"

using DiagM = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

#define TICK start = std::chrono::high_resolution_clock::now();
#define TOCK                                                                 \
  {                                                                          \
    auto stop = std::chrono::high_resolution_clock::now();                   \
    auto duration =                                                          \
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
    std::cout << duration.count() << std::endl;                              \
  }

void dynamic_model::invert_model() {

  std::ofstream param_e_file;
  std::ofstream param_c_file;
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
                                           Eigen::DontAlignCols, ",", "\n");
  if (this->intermediate_outputs_to_file) {

    param_e_file.open(this->intermediate_expectations_filename);
    param_c_file.open(this->intermediate_covariances_filename);
  }

  this->performed_it = 0;
  Eigen::MatrixXd response_vars_fs = this->get_observed_outcomes();
  int num_response_total = this->num_response_vars * this->num_samples;
  if (!num_response_vars) {
    num_response_vars = select_response_vars.size();
  }
  Eigen::MatrixXd initial_state = Eigen::MatrixXd::Zero(1, 1);
  int dt = 1;
  std::vector<DiagM> precision_comp(num_response_vars);
  int num_precision_comp = num_response_vars;
  for (int i = 0; i < num_response_vars; i++) {
    DiagM prec(num_response_total);
    prec.setZero();
    for (int j = i * num_samples; j < i * num_samples + num_samples; j++) {
      prec.diagonal()[j] = 1;
    }
    precision_comp[i] = prec;
  }
  Eigen::VectorXd prior_p_e = this->prior_parameter_expectations;
  Eigen::MatrixXd prior_p_c = this->prior_parameter_covariances;
  Eigen::VectorXd prior_h_e = this->prior_hyper_expectations;
  Eigen::MatrixXd inv_prior_h_c = this->prior_hyper_covariances;
  inv_prior_h_c = utility::inverse_tol(inv_prior_h_c);
  // Note: different SVD implementations may produces different but
  // equally valid decomps of U and V.
  Eigen::BDCSVD<Eigen::MatrixXd> svd;
  svd.setThreshold(std::numeric_limits<double>::epsilon() * 64);
  svd.compute(prior_p_c, Eigen::ComputeThinV | Eigen::ComputeThinU);
  Eigen::MatrixXd singular_vec = svd.matrixU();

  int num_parameters_eff = svd.nonzeroSingularValues();
  // Reduce parameter space
  prior_p_c = singular_vec.transpose() * prior_p_c * singular_vec;
  Eigen::MatrixXd inv_prior_p_c = utility::inverse_tol(prior_p_c);

  Eigen::VectorXd likel_p_e = Eigen::VectorXd::Zero(num_parameters_eff);
  Eigen::VectorXd conditional_p_e = prior_p_e;
  Eigen::MatrixXd conditional_p_c;
  double ascent_rate = -4.0;

  Eigen::VectorXd h_estimate = prior_h_e;
  Eigen::VectorXd p_estimate = Eigen::VectorXd::Zero(prior_p_e.size());

  Eigen::VectorXd current_p_estimate;
  Eigen::MatrixXd current_p_con_cov_estimate;
  Eigen::VectorXd current_h_estimate;
  double current_free_energy = -INFINITY;
  double initial_free_energy = -INFINITY;
  int criterion = 4;
  int num_success = 0;
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> forward_model_function =
      this->get_forward_model_function();

  Eigen::VectorXd dFdp;
  Eigen::MatrixXd dFdpp;
  
  for (int i = 0; i < this->max_invert_it; i++) {
    this->performed_it++;
    auto start = std::chrono::high_resolution_clock::now();

    // Runtime polymorphism - we have to use this->forward model to make
    // sure we've calling the implementation of forward model method of
    // inheriting classes
    Eigen::MatrixXd predicted_f = forward_model_function(conditional_p_e);
    Eigen::MatrixXd dfdp =
        this->diff(forward_model_function, conditional_p_e, singular_vec);
    // Check for stability. Is the Infinity Norm in resonable bounds?
    double norm = dfdp.cwiseAbs().rowwise().sum().maxCoeff();
    bool revert = (norm > 1e32);
    if (revert) {
      throw std::runtime_error("revert hit");
    }
    Eigen::VectorXd prediction_error = response_vars_fs - predicted_f;
    Eigen::VectorXd hyper_error;
    // J == -dfdp with no confounds present
    Eigen::MatrixXd J = -dfdp;
    DiagM i_cov_comp(num_response_total);
    Eigen::MatrixXd conditional_p_cov;
    Eigen::MatrixXd conditional_h_cov;
    for (int k = 0; k < 8; k++) {
      i_cov_comp.setZero();
      for (int l = 0; l < num_precision_comp; l++) {
        i_cov_comp.diagonal() +=
              (precision_comp[l].diagonal() * (exp(-32) + exp(h_estimate(l))));
      }
      Eigen::MatrixXd cov_comp = utility::inverse_tol(i_cov_comp);
      Eigen::MatrixXd i_conditional_p_cov =
          (J.transpose() * i_cov_comp * J) + inv_prior_p_c;
      conditional_p_cov = utility::inverse_tol(i_conditional_p_cov);

      std::vector<DiagM> P_mstep_op(num_response_vars);
      std::vector<DiagM> PS_mstep_op(num_response_vars);
      std::vector<Eigen::MatrixXd> JPJ_mstep_op(num_response_vars);

      for (int l = 0; l < num_response_vars; l++) {
          DiagM P(num_response_total);
          P.diagonal() = precision_comp[l].diagonal() * exp(h_estimate(l));
          P_mstep_op[l] = P;
          PS_mstep_op[l].diagonal() =
              P.diagonal().cwiseProduct(cov_comp.diagonal());
          JPJ_mstep_op[l] = J.transpose() * P * J;
      }

      Eigen::VectorXd dFdh = Eigen::VectorXd::Zero(num_response_vars);
      Eigen::MatrixXd dFdhh =
          Eigen::MatrixXd::Zero(num_response_vars, num_response_vars);
      for (int l = 0; l < num_response_vars; l++) {
        Eigen::MatrixXd ePe =
            (prediction_error.transpose() * P_mstep_op[l] * prediction_error) /
            2;
        dFdh(l) = PS_mstep_op[l].diagonal().sum() / 2 - ePe.value() -
                  (conditional_p_cov * JPJ_mstep_op[l]).trace() / 2;
        for (int m = 0; m < num_response_vars; m++) {
          dFdhh(l, m) = -(PS_mstep_op[l]
                                .diagonal()
                                .cwiseProduct(PS_mstep_op[m].diagonal())
                                .sum()) /
                          2;
          dFdhh(m, l) = dFdhh(l, m);
        }
      }
      hyper_error = (h_estimate - prior_h_e);
      dFdh = dFdh - (inv_prior_h_c * hyper_error);
      dFdhh = dFdhh - inv_prior_h_c;
      conditional_h_cov = utility::inverse_tol(-dFdhh);
      Eigen::VectorXd dh = utility::dx(dFdhh, dFdh, ascent_rate);
      dh = (dh.unaryExpr(
                [](double x) { return std::min(std::max(x, -1.0), 1.0); }))
               .eval();
      h_estimate = h_estimate + dh;
      double dFh = dFdh.transpose() * dh;

      if (dFh < 1e-2) {
        break;
      }
    }

    double L1 = (utility::logdet(i_cov_comp) -
                 prediction_error.transpose() * i_cov_comp * prediction_error -
                 num_response_total * log(8 * atan(1.0))) /
                2;
    double L2 = (utility::logdet(inv_prior_p_c * conditional_p_cov) -
                 p_estimate.transpose() * inv_prior_p_c * p_estimate) /
                2;
    double L3 = (utility::logdet(inv_prior_h_c * conditional_h_cov) -
                 hyper_error.transpose() * inv_prior_h_c * hyper_error) /
                2;
    double free_energy_var = L1 + L2 + L3;

    if (initial_free_energy == -INFINITY) {
      initial_free_energy = free_energy_var;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (i > 0) {
      std::cout << "actual: " << free_energy_var - current_free_energy
                << " microsec: " << (duration.count()) << '\n';
    }

    std::string str;
    if ((free_energy_var > current_free_energy) | (i < 2)) {
      current_p_estimate = p_estimate;
      current_p_con_cov_estimate = conditional_p_cov;
      current_h_estimate = h_estimate;
      current_free_energy = free_energy_var;

      dFdp = (-(J.transpose() * i_cov_comp * prediction_error) -
              inv_prior_p_c * p_estimate);
      dFdpp = (-(J.transpose() * i_cov_comp * J) - inv_prior_p_c);
      ascent_rate = std::min(ascent_rate + 0.5, 4.0);
      str = "EM:(+) " + std::to_string(i + 1) + " ";
    } else {
      p_estimate = current_p_estimate;
      conditional_p_cov = current_p_con_cov_estimate;
      h_estimate = current_h_estimate;
      ascent_rate = std::min(ascent_rate - 2, -4.0);
      str = "EM:(-) " + std::to_string(i + 1) + " ";
    }
    Eigen::VectorXd dp = utility::dx(dFdpp, dFdp, ascent_rate);
    p_estimate = p_estimate + dp;

    conditional_p_e =
        prior_p_e +
        singular_vec * p_estimate(Eigen::seq(0, num_parameters_eff - 1));

    double dF = dFdp.transpose() * dp;
    conditional_p_c =
        singular_vec * conditional_p_cov * singular_vec.transpose();
    if (this->intermediate_outputs_to_file) {
      param_e_file << conditional_p_e.transpose().format(CSVFormat) << '\n';
      param_c_file << conditional_p_c.format(CSVFormat) << '\n';
    }

    std::cout << str << "F: " << current_free_energy - initial_free_energy
              << ' ' << "dF predicted: " << dF << ' ';
    if (dF < this->converge_crit) {
      num_success += 1;
      if (num_success == criterion) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "actual: " << free_energy_var - current_free_energy
                  << " microsec: " << (duration.count()) << '\n';
        std::cout << "convergence" << '\n';
        break;
      }
    } else {
      num_success = 0;
    }
  }

  this->conditional_parameter_expectations =
      prior_p_e +
      singular_vec * p_estimate(Eigen::seq(0, num_parameters_eff - 1));
  this->conditional_parameter_covariances =
      singular_vec * current_p_con_cov_estimate * singular_vec.transpose();
  this->conditional_hyper_expectations = current_h_estimate;
  this->free_energy = current_free_energy;

  if (this->intermediate_outputs_to_file) {
    param_e_file.close();
    param_c_file.close();
  }

  return;
}

/*
 * get_observed_outcomes function, to be overwritten in inheriting classes. If
 * this function is reached, something is wrong and an error is thrown.
 */
Eigen::VectorXd dynamic_model::get_observed_outcomes() {
  throw std::runtime_error("error: observed outcomes function not specified");
  return Eigen::VectorXd::Zero(1);
}

/*
 * Wrap the forward model function
 */
std::function<Eigen::VectorXd(const Eigen::VectorXd&)>
dynamic_model::get_forward_model_function() {
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> forward_model =
      std::bind(&dynamic_model::forward_model, this, std::placeholders::_1);
  return forward_model;
}

/*
 * Evaluate the forward model, to be overwritten in inheriting classes. If
 * this function is reached, something is wrong and an error is thrown.
 */
Eigen::VectorXd dynamic_model::forward_model(
    const Eigen::VectorXd& parameters) {
  throw std::runtime_error("error: forward_model not specified");
  return Eigen::VectorXd::Zero(1);
}

/*
 * get_observed_outcomes function, to be overwritten in inheriting classes. If
 * this function is reached, something is wrong and an error is thrown.
 */
dynamic_model::dynamic_model() { return; }
