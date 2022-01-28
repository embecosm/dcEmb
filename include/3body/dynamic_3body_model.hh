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

#include <vector>
#include "dynamic_model.hh"
#include "feature_selection_3body.hh"
#include "generative_3body.hh"
#include "parameter_location_3body.hh"
#pragma once

/**
 * Dynamic Causal Model class for the 3body problem.
 */
class dynamic_3body_model : public dynamic_model {
 public:
  /**
   * @brief object used to evaluate the generative model
   */
  generative_3body gen_function;
  /**
   * @brief object used to evaluate feature selection
   */
  feature_selection_3body fs_function;
  /**
   * @brief parameter locations for COVID-19
   */
  parameter_location_3body parameter_locations;
  Eigen::VectorXd get_observed_outcomes();
  std::function<Eigen::VectorXd(Eigen::VectorXd)> get_forward_model_function();
  Eigen::VectorXd forward_model(
      const Eigen::VectorXd& parameters,
      const parameter_location_3body& parameter_locations,
      const int& timeseries_length,
      const Eigen::VectorXi& select_response_vars);
  dynamic_3body_model();
};

inline bool operator==(const dynamic_3body_model& lhs,
                       const dynamic_3body_model& rhs) {
  return lhs.gen_function == rhs.gen_function &
         lhs.fs_function == rhs.fs_function &
         lhs.parameter_locations == rhs.parameter_locations &
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