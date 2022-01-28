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
struct parameter_location_3body {
  int planet_coordsX = -1;
  int planet_coordsY = -1;
  int planet_coordsZ = -1;
  int planet_masses = -1;
  int planet_velocityX = -1;
  int planet_velocityY = -1;
  int planet_velocityZ = -1;
  int planet_accelerationX = -1;
  int planet_accelerationY = -1;
  int planet_accelerationZ = -1;
};

inline bool operator==(const parameter_location_3body& lhs,
                       const parameter_location_3body& rhs) {
  return lhs.planet_coordsX == rhs.planet_coordsX &
         lhs.planet_coordsY == rhs.planet_coordsY &
         lhs.planet_coordsZ == rhs.planet_coordsZ &
         lhs.planet_masses == rhs.planet_masses &
         lhs.planet_velocityX == rhs.planet_velocityX &
         lhs.planet_velocityY == rhs.planet_velocityY &
         lhs.planet_velocityZ == rhs.planet_velocityZ &
         lhs.planet_accelerationX == rhs.planet_accelerationX &
         lhs.planet_accelerationY == rhs.planet_accelerationY &
         lhs.planet_accelerationZ == rhs.planet_accelerationZ;
}