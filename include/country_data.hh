/**
 * A struct containing COVID-19 country data for the dcEmb package
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

#pragma once
#include <Eigen/Dense>

/**
 * Struct that contains all the COVID data available in the files, for a given
 * country
 */
struct country_data {
  std::string name;
  int pop;
  double latitude;
  double longitude;
  std::string date;
  int first_case;
  int days;
  double cum_deaths;
  double cum_cases;
  Eigen::VectorXd cases;
  Eigen::VectorXd deaths;
};