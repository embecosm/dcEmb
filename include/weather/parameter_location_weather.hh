/**
 * Structure for locating parameters within the COVID-19 DCM for the dcEmb
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
#include <fstream>
#include <iostream>
#pragma once
struct parameter_location_weather {
  Eigen::VectorXi ocean_heat_transfer = Eigen::VectorXi::Ones(1) * -1;
  Eigen::VectorXi ocean_heat_capacity = Eigen::VectorXi::Ones(1) * -1;
  Eigen::VectorXi deep_ocean_efficacy = Eigen::VectorXi::Ones(1) * -1;
  Eigen::VectorXi sigma_eta = Eigen::VectorXi::Ones(1) * -1;
  Eigen::VectorXi sigma_xi = Eigen::VectorXi::Ones(1) * -1;
  Eigen::VectorXi gamma_autocorrelation = Eigen::VectorXi::Ones(1) * -1;
  Eigen::VectorXi forcing_4co2 = Eigen::VectorXi::Ones(1) * -1;
};

inline bool operator==(const parameter_location_weather& lhs,
                       const parameter_location_weather& rhs) {
  return 0;
}