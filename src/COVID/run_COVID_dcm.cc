/**
 * Main file for running the COVID-19 DCM within the dcEmb package
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

#include <DEM_COVID.hh>
#include <Eigen/Eigen>
#include <iostream>
#include <run_COVID_dcm.hh>

/**
 * Check number of threads Eigen is operating on, then run COVID test
 */
int main() {
  Eigen::initParallel();
#if defined(_OPENMP)
  std::cout << "OpenMP multithreading enabled with " << Eigen::nbThreads()
            << " cores" << '\n';
#else
  std::cout << "OpenMP multithreading not enabled, using " << Eigen::nbThreads()
            << " cores" << '\n';
#endif
  int COVID_test = run_COVID_test();
  return (0);
}
