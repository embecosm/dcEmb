/**
 * Main file for running the 3-body DCM within the dcEmb package
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

#include <DEM_3body.hh>
#include <Eigen/Eigen>
#include <iostream>
#include <run_3body_dcm.hh>

/**
 * Check number of threads Eigen is operating on, then run 3body test
 */
int main() {
  std::cout << "OpenMP enabled with " << Eigen::nbThreads() << " cores" << '\n';

  int test = run_3body_test();
  exit(2);
  return (0);
}
