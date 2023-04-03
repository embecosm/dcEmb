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
struct species {
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

struct species_struct {
  std::vector<std::string> name;
  std::vector<std::string> type;
  std::vector<std::string> input_mode;
  Eigen::VectorXi greenhouse_gas;
  Eigen::VectorXi aerosol_chemistry_from_emissions;
  Eigen::VectorXi aerosol_chemistry_from_concentration;
  Eigen::MatrixXd partition_fraction;
  Eigen::MatrixXd unperturbed_lifetime;
  Eigen::VectorXd tropospheric_adjustment;
  Eigen::VectorXd forcing_efficacy;
  Eigen::VectorXd forcing_temperature_feedback;
  Eigen::VectorXd forcing_scale;
  Eigen::VectorXd molecular_weight;
  Eigen::VectorXd baseline_concentration;
  Eigen::VectorXd forcing_reference_concentration;
  Eigen::VectorXd forcing_reference_emissions;
  Eigen::VectorXd iirf_0;
  Eigen::VectorXd iirf_airborne;
  Eigen::VectorXd iirf_uptake;
  Eigen::VectorXd iirf_temperature;
  Eigen::VectorXd baseline_emissions;
  Eigen::VectorXd g1;
  Eigen::VectorXd g0;
  Eigen::VectorXd greenhouse_gas_radiative_efficiency;
  Eigen::VectorXd contrails_radiative_efficiency;
  Eigen::VectorXd erfari_radiative_efficiency;
  Eigen::VectorXd h2o_stratospheric_factor;
  Eigen::VectorXd lapsi_radiative_efficiency;
  Eigen::VectorXd land_use_cumulative_emissions_to_forcing;
  Eigen::VectorXd ozone_radiative_efficiency;
  Eigen::VectorXd cl_atoms;
  Eigen::VectorXd br_atoms;
  Eigen::VectorXd fractional_release;
  Eigen::VectorXd ch4_lifetime_chemical_sensitivity;
  Eigen::VectorXd aci_shape;
  Eigen::VectorXd aci_scale;
  Eigen::VectorXd lifetime_temperature_sensitivity;
  Eigen::VectorXd concentration_per_emission;
  species_struct(int i) {
    name = std::vector<std::string>(i);
    type = std::vector<std::string>(i);
    input_mode = std::vector<std::string>(i);
    greenhouse_gas = Eigen::VectorXi(i);
    aerosol_chemistry_from_emissions = Eigen::VectorXi(i);
    aerosol_chemistry_from_concentration = Eigen::VectorXi(i);
    partition_fraction = Eigen::MatrixXd(4, i);
    unperturbed_lifetime = Eigen::MatrixXd(4, i);
    tropospheric_adjustment = Eigen::VectorXd(i);
    forcing_efficacy = Eigen::VectorXd(i);
    forcing_temperature_feedback = Eigen::VectorXd(i);
    forcing_scale = Eigen::VectorXd(i);
    molecular_weight = Eigen::VectorXd(i);
    baseline_concentration = Eigen::VectorXd(i);
    forcing_reference_concentration = Eigen::VectorXd(i);
    forcing_reference_emissions = Eigen::VectorXd(i);
    iirf_0 = Eigen::VectorXd(i);
    iirf_airborne = Eigen::VectorXd(i);
    iirf_uptake = Eigen::VectorXd(i);
    iirf_temperature = Eigen::VectorXd(i);
    baseline_emissions = Eigen::VectorXd(i);
    g1 = Eigen::VectorXd(i);
    g0 = Eigen::VectorXd(i);
    greenhouse_gas_radiative_efficiency = Eigen::VectorXd(i);
    contrails_radiative_efficiency = Eigen::VectorXd(i);
    erfari_radiative_efficiency = Eigen::VectorXd(i);
    h2o_stratospheric_factor = Eigen::VectorXd(i);
    lapsi_radiative_efficiency = Eigen::VectorXd(i);
    land_use_cumulative_emissions_to_forcing = Eigen::VectorXd(i);
    ozone_radiative_efficiency = Eigen::VectorXd(i);
    cl_atoms = Eigen::VectorXd(i);
    br_atoms = Eigen::VectorXd(i);
    fractional_release = Eigen::VectorXd(i);
    ch4_lifetime_chemical_sensitivity = Eigen::VectorXd(i);
    aci_shape = Eigen::VectorXd(i);
    aci_scale = Eigen::VectorXd(i);
    lifetime_temperature_sensitivity = Eigen::VectorXd(i);
    concentration_per_emission = Eigen::VectorXd(i);
  }
  species_struct() {}
};

inline bool operator==(const species_struct& lhs, const species_struct& rhs) {
  return 0;
}