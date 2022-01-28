/**
 * A class for performing Bayesian Model Reduction for the dcEmb package
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

#include <Eigen/Dense>
#include "bma_model.hh"
#include "utility.hh"

#pragma once

/**
 * Bayesian Model Reduction Class. Evaluates all sub models of the specified
 * Dynamic Causal Model analytically, without running inversion on each
 * directly. Returns either the model with the highest free
 * energy (Bayesian Model Selection, WIP), or a Bayesian Model Average of all
 * different models.
 */
template <typename dynamic_model_template>
class bmr_model {
 public:
  /**
   * @brief Input DCM model
   */
  dynamic_model_template DCM_in;
  /**
   * @brief beta parameter, prior expectation of reduced parameters.
   * Currently unused.
   */
  double beta = 0;
  /**
   * @brief gamma parameter, prior variance of reduced parameters.
   * Currently unused.
   */
  double gamma = 0;
  /**
   * @brief Vector of free energies for reduced models
   */
  Eigen::VectorXd free_energy_vector;
  /**
   * @brief Vector of posterior expectations for reduced models
   */
  Eigen::VectorXd posterior_probability_vector;
  /**
   * @brief Matrix specifying parameters set or not set
   */
  Eigen::MatrixXi model_space;
  /**
   * @brief Output reduced Dynamic Causal Model.
   */
  dynamic_model_template DCM_out;
  bma_model<dynamic_model_template> BMA;
  /**
   * Calculate all reductions of the given DCM.
   *
   * The returned DCM will vary depending on the parameters of the BMR:
   * a DCm that is the Bayesian Model Average (BMA) of all candidate
   * reduced models, or a DCM that corresponds to the best reduced model
   * (by Free Energy).
   * In the case of a BMA, free energy will be set to -INFINITY
   *
   * Input Dynamic Causal Model requires at least the following to be
   * defined:
   *  - prior_parameter_expectations
   *  - prior_parameter_covariances
   *  - conditional_parameter_expectations
   *  - conditional_parameter_covariances
   *
   * Currently assumptuons and limitations:
   *  - BMA only
   *  - All parameters selected as valid choices for reduction
   *
   */
  void reduce() {
    Eigen::BDCSVD<Eigen::MatrixXd> svd;
    svd.setThreshold(std::numeric_limits<double>::epsilon() * 64);
    svd.compute(DCM_in.prior_parameter_covariances,
                Eigen::ComputeThinV | Eigen::ComputeThinU);
    Eigen::MatrixXd singular_vec = svd.matrixU();

    Eigen::VectorXd prior_p_e =
        singular_vec.transpose() * DCM_in.prior_parameter_expectations;
    Eigen::MatrixXd prior_p_c = singular_vec.transpose() *
                                DCM_in.prior_parameter_covariances *
                                singular_vec;
    Eigen::VectorXd posterior_p_e =
        singular_vec.transpose() *
        DCM_in.conditional_parameter_expectations.reshaped(
            DCM_in.conditional_parameter_expectations.rows() *
                DCM_in.conditional_parameter_expectations.cols(),
            1);
    Eigen::MatrixXd posterior_p_c = singular_vec.transpose() *
                                    DCM_in.conditional_parameter_covariances *
                                    singular_vec;

    Eigen::VectorXd diag_values =
        DCM_in.conditional_parameter_covariances.diagonal();
    Eigen::VectorXi accum_reduction_vec;

    if (diag_values.sum() < 1024) {
      double mean_sel_diag =
          diag_values.unaryExpr([](double x) { return (x < 1024) ? (x) : 0; })
              .mean() /
          1024;
      accum_reduction_vec = diag_values.unaryExpr(
          [mean_sel_diag](double x) { return (x > mean_sel_diag) ? 1 : 0; });
    } else {
      accum_reduction_vec =
          diag_values.unaryExpr([](double x) { return (x > 0) ? 1 : 0; });
    }

    int greedy = 1;
    Eigen::VectorXi params;
    Eigen::MatrixXi K;
    Eigen::VectorXd free_energy_vec2;
    Eigen::VectorXd sm_free_energy_vec2;
    while (greedy) {
      int nparams = accum_reduction_vec.sum();
      int nmax = std::max(nparams / 4, 8);
      params = Eigen::VectorXi(nparams);
      int pos = 0;
      for (int i = 0; i < accum_reduction_vec.size(); i++) {
        if (accum_reduction_vec(i) == 1) {
          params(pos) = i;
          pos++;
        }
      }
      if (nparams > nmax) {
        Eigen::VectorXd free_energy_vec1 = Eigen::VectorXd::Zero(nparams);
        Eigen::VectorXi permute_vec = Eigen::VectorXi::Zero(nparams);
        for (int i = 0; i < nparams; i++) {
          Eigen::VectorXi r = accum_reduction_vec;
          r(params(i)) = 0;
          Eigen::VectorXi s = r.unaryExpr([](int x) { return (1 - x); });
          Eigen::MatrixXd red_c_tmp1 =
              Eigen::MatrixXd::Zero(r.size(), r.size());
          // std::cout << "nparams" << nparams << '\n';

          red_c_tmp1.diagonal() = (r.cast<double>() + s.cast<double>() * gamma);
          Eigen::MatrixXd red_c_tmp2 =
              singular_vec.transpose() * red_c_tmp1 * singular_vec;
          Eigen::MatrixXd prior_r_p_c = red_c_tmp2 * prior_p_c * red_c_tmp2;
          Eigen::MatrixXd red_e_tmp1 =
              Eigen::MatrixXd::Zero(r.size(), r.size());
          red_e_tmp1.diagonal() = r.cast<double>();
          Eigen::MatrixXd red_e_tmp2 =
              singular_vec.transpose() * red_e_tmp1 * singular_vec;
          Eigen::MatrixXd prior_r_p_e =
              red_c_tmp2 * prior_p_e +
              singular_vec.transpose() * s.cast<double>() * beta;
          std::vector<Eigen::MatrixXd> log_evidence =
              utility::reduced_log_evidence(posterior_p_e, posterior_p_c,
                                            prior_p_e, prior_p_c, prior_r_p_e,
                                            prior_r_p_c);
          free_energy_vec1(i) = log_evidence[0].value();
          permute_vec[i] = i;
        }
        std::sort(permute_vec.data(), permute_vec.data() + permute_vec.size(),
                  [free_energy_vec1](int a, int b) {
                    return (-free_energy_vec1(a) < -free_energy_vec1(b));
                  });
        params = params(permute_vec(Eigen::seq(0, nmax - 1))).eval();
      } else if (nparams == 0) {
        std::cout << "There are no free parameters in this model" << '\n';
        return;
      } else {
        std::cout << "Flagging non-greedy search" << '\n';
        greedy = 0;
      }
      for (int j = 0; j < 2; j++) {
        if (j == 0) {
          K = Eigen::MatrixXi(2, params.size());
          for (int i = 0; i < params.size(); i++) {
            K(0, i) = 1;
            K(1, i) = 0;
          }
        } else {
          int size = std::min(8, (int)params.size());
          params = params(Eigen::seq(0, size - 1)).eval();
          K = Eigen::MatrixXi(2 << (size - 1), size);
          for (int i = 0; i < (2 << (size - 1)); i++) {
            for (int j = 0; j < size; j++) {
              K(i, size - 1 - j) = (i & (1 << j)) ? 0 : 1;
            }
          }
        }
        int nK = K.rows();
        free_energy_vec2 = Eigen::VectorXd::Zero(nK);
        for (int i = 0; i < K.rows(); i++) {
          Eigen::VectorXi r = accum_reduction_vec;
          for (int k = 0; k < K.cols(); k++) {
            if (K(i, k)) {
              r(params(k)) = 0;
            }
          }
          Eigen::VectorXi s = r.unaryExpr([](int x) { return (1 - x); });
          Eigen::MatrixXd red_c_tmp1 =
              Eigen::MatrixXd::Zero(r.size(), r.size());
          red_c_tmp1.diagonal() = (r.cast<double>() + s.cast<double>() * gamma);
          Eigen::MatrixXd red_c_tmp2 =
              singular_vec.transpose() * red_c_tmp1 * singular_vec;
          Eigen::MatrixXd prior_r_p_c = red_c_tmp2 * prior_p_c * red_c_tmp2;
          Eigen::MatrixXd red_e_tmp1 =
              Eigen::MatrixXd::Zero(r.size(), r.size());
          red_e_tmp1.diagonal() = r.cast<double>();
          Eigen::MatrixXd red_e_tmp2 =
              singular_vec.transpose() * red_e_tmp1 * singular_vec;
          Eigen::MatrixXd prior_r_p_e =
              red_c_tmp2 * prior_p_e +
              singular_vec.transpose() * s.cast<double>() * beta;
          std::vector<Eigen::MatrixXd> log_evidence =
              utility::reduced_log_evidence(posterior_p_e, posterior_p_c,
                                            prior_p_e, prior_p_c, prior_r_p_e,
                                            prior_r_p_c);
          free_energy_vec2(i) = log_evidence[0].value();
        }
        if (((free_energy_vec2(0) -
              free_energy_vec2(free_energy_vec2.size() - 1)) > (double)nmax) &
            (nparams > nmax)) {
          break;
        } else {
          // std::cout << "test_free_energy" << free_energy_vec2 << '\n';
          nmax = 8;
        }
      }
      sm_free_energy_vec2 = utility::softmax(free_energy_vec2);

      int best_free_energy;
      sm_free_energy_vec2.maxCoeff(&best_free_energy);
      for (int i = 0; i < K.cols(); i++) {
        if (K(best_free_energy, i) == 1) {
          accum_reduction_vec(params(i)) = 0;
        }
      }
      int nelim = K.row(best_free_energy).sum();
      greedy = greedy && nelim;
      std::cout << nelim << " out of " << nparams << " free"
                << " parameters removed" << '\n';
    }

    Eigen::MatrixXd mean_params1 = Eigen::MatrixXd(2, params.size());
    for (int i = 0; i < params.size(); i++) {
      mean_params1(0, i) =
          (double)((K.col(i).array() == 0).select(0.0, sm_free_energy_vec2))
              .mean();
      mean_params1(1, i) =
          (double)((K.col(i).array() == 0).select(sm_free_energy_vec2, 0.0))
              .mean();
    }
    Eigen::VectorXd mean_params2 = mean_params1.row(0) / mean_params1.sum();
    Eigen::VectorXd posterior_p_p_e = accum_reduction_vec.cast<double>();
    posterior_p_p_e(params) = mean_params2;

    double free_energy_max = free_energy_vec2.maxCoeff();

    (this->BMA).posterior_p_e_all = std::vector<Eigen::MatrixXd>(K.rows());
    (this->BMA).posterior_p_c_all = std::vector<Eigen::MatrixXd>(K.rows());
    (this->BMA).free_energy_all = Eigen::VectorXd(K.rows());

    Eigen::VectorXd prior_r_p_e;
    Eigen::MatrixXd prior_r_p_c;
    for (int i = 0; i < K.rows(); i++) {
      if (free_energy_vec2[i] > (free_energy_max - 8.0)) {
        Eigen::VectorXi r = accum_reduction_vec;
        for (int k = 0; k < K.cols(); k++) {
          if (K(i, k)) {
            r(params(k)) = 0;
          }
        }
        Eigen::VectorXi s = r.unaryExpr([](int x) { return (1 - x); });
        Eigen::MatrixXd red_c_tmp1 = Eigen::MatrixXd::Zero(r.size(), r.size());
        red_c_tmp1.diagonal() = (r.cast<double>() + s.cast<double>() * gamma);
        Eigen::MatrixXd red_c_tmp2 =
            singular_vec.transpose() * red_c_tmp1 * singular_vec;
        prior_r_p_c = red_c_tmp2 * prior_p_c * red_c_tmp2;
        Eigen::MatrixXd red_e_tmp1 = Eigen::MatrixXd::Zero(r.size(), r.size());
        red_e_tmp1.diagonal() = r.cast<double>();
        Eigen::MatrixXd red_e_tmp2 =
            singular_vec.transpose() * red_e_tmp1 * singular_vec;
        prior_r_p_e = red_c_tmp2 * prior_p_e +
                      singular_vec.transpose() * s.cast<double>() * beta;
        std::vector<Eigen::MatrixXd> log_evidence =
            utility::reduced_log_evidence(posterior_p_e, posterior_p_c,
                                          prior_p_e, prior_p_c, prior_r_p_e,
                                          prior_r_p_c);

        (this->BMA).free_energy_all(i) = log_evidence.at(0).value();
        (this->BMA).posterior_p_e_all.at(i) = log_evidence.at(1);
        (this->BMA).posterior_p_c_all.at(i) = log_evidence.at(2);
      }
    }
    this->BMA.average_ffx();

    this->free_energy_vector = free_energy_vec2;
    this->posterior_probability_vector = sm_free_energy_vec2;
    this->model_space = K;

    this->DCM_out = this->DCM_in;
    (this->DCM_out).prior_parameter_expectations = prior_r_p_e;
    (this->DCM_out).prior_parameter_covariances = prior_r_p_c;
    (this->DCM_out).conditional_parameter_expectations =
        (this->BMA).posterior_p_e;
    (this->DCM_out).conditional_parameter_covariances =
        (this->BMA).posterior_p_c;
    (this->DCM_out).posterior_over_parameters = posterior_p_p_e;
    (this->DCM_out).free_energy = -INFINITY;

    return;
  }
};