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

#include <vector>
#include "Eigen/Dense"
#include "country_data.hh"
#include "dynamic_COVID_model.hh"
#include "parameter_location_COVID.hh"
#include "peb_model.hh"
#pragma once

int run_COVID_test();

Eigen::VectorXd default_prior_expectations();
Eigen::MatrixXd default_prior_covariances();
parameter_location_COVID default_parameter_locations();
Eigen::VectorXd default_hyper_expectations();
Eigen::MatrixXd default_hyper_covariances();
Eigen::VectorXi default_random_effects(parameter_location_COVID& pl);

std::vector<dynamic_COVID_model> generate_comparison_data(
    std::vector<country_data> countries);
void generate_us_posteriors(dynamic_COVID_model& model);
void generate_brazil_posteriors(dynamic_COVID_model& model);
void generate_india_posteriors(dynamic_COVID_model& model);
void generate_russia_posteriors(dynamic_COVID_model& model);
void generate_mexico_posteriors(dynamic_COVID_model& model);
void generate_PEB_values(peb_model<dynamic_COVID_model>& model);
