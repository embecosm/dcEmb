/**
 * The weather dynamic causal model class within the dcEmb package
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

#include "Eigen/Dense"
#include "parameter_location_weather.hh"
#include "species_struct.hh"
#pragma once

int run_weather_test();

std::vector<Eigen::MatrixXd> simple_ecf(const species_struct& species,
                           const std::string& scenario, const int& start_date,
                           const int& end_date);
species_struct simple_species_struct(const std::vector<std::string>& species_names);

species_struct simple_species_struct();
parameter_location_weather default_parameter_locations();
Eigen::VectorXd true_prior_expectations();
Eigen::VectorXd default_prior_expectations();
Eigen::MatrixXd default_prior_covariances();
Eigen::VectorXd default_hyper_expectations();
Eigen::MatrixXd default_hyper_covariances();