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
  std::vector<std::string> names;
  std::vector<int> locations;
};

inline bool operator==(const parameter_location_weather& lhs,
                       const parameter_location_weather& rhs) {
  return lhs.names == rhs.names & lhs.locations == rhs.locations;
}