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
#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#pragma once
struct species_struct {
  std::string name = "name";
  std::string type = "type";
  std::string input_mode = "input_mode";
  bool greenhouse_gas = false;
  bool aerosol_chemistry_from_emissions = false;
  bool aerosol_chemistry_from_concentration = false;
  Eigen::VectorXd partition_fraction = Eigen::VectorXd::Zero(4);
  Eigen::VectorXd unperturbed_lifetime = Eigen::VectorXd::Zero(4);
  double tropospheric_adjustment = 0;
  double forcing_efficacy = 0;
  double forcing_temperature_feedback = 0;
  double forcing_scale = 0;
  double molecular_weight = 0;
  double baseline_concentration = 0;
  double forcing_reference_concentration = 0;
  double forcing_reference_emissions = 0;
  double iirf_0 = 0;
  double iirf_airborne = 0;
  double iirf_uptake = 0;
  double iirf_temperature = 0;
  double baseline_emissions = 0;
  double g1 = 0;
  double g0 = 0;
  double greenhouse_gas_radiative_efficiency = 0;
  double contrails_radiative_efficiency = 0;
  double erfari_radiative_efficiency = 0;
  double h2o_stratospheric_factor = 0;
  double lapsi_radiative_efficiency = 0;
  double land_use_cumulative_emissions_to_forcing = 0;
  double ozone_radiative_efficiency = 0;
  double cl_atoms = 0;
  double br_atoms = 0;
  double fractional_release = 0;
  double ch4_lifetime_chemical_sensitivity = 0;
  double aci_shape = 0;
  double aci_scale = 0;
  double lifetime_temperature_sensitivity = 0;
  double concentration_per_emission = 0;
};

inline bool operator==(const species_struct& lhs, const species_struct& rhs) {
  return 0;
}