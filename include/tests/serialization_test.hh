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

#include "Eigen/Dense"
#include "parameter_location_3body.hh"
#include "parameter_location_COVID.hh"

#pragma once

Eigen::VectorXd true_3body_prior_expectations();
Eigen::VectorXd default_3body_prior_expectations();
Eigen::MatrixXd default_3body_prior_covariances();
parameter_location_3body default_3body_parameter_locations();
Eigen::VectorXd default_3body_hyper_expectations();
Eigen::MatrixXd default_3body_hyper_covariances();

Eigen::VectorXi default_COVID_random_effects(parameter_location_COVID& pl);
Eigen::VectorXd true_COVID_prior_expectations();
Eigen::VectorXd default_COVID_prior_expectations();
Eigen::MatrixXd default_COVID_prior_covariances();
parameter_location_COVID default_COVID_parameter_locations();
Eigen::VectorXd default_COVID_hyper_expectations();
Eigen::MatrixXd default_COVID_hyper_covariances();
