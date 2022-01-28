/**
 * A collection of functions for importing and processing dynamic causal models
 * within the dcEmb package
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
#pragma once

std::vector<country_data> read_country_data(int num_countries);

int get_country_pop(const std::string& name);

Eigen::VectorXd smooth(const Eigen::VectorXi& data, int period);

Eigen::MatrixXd mat_power(const Eigen::MatrixXd& mat, int p);

Eigen::VectorXd zero_padded_add(const Eigen::VectorXd& mat1,
                                const Eigen::VectorXd& mat2);

void splitstr(std::vector<std::string>& vec, std::string& str,
              const char& delim);
