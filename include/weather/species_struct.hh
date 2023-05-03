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
  Eigen::VectorXi co2_indices;
  Eigen::VectorXi ch4_indices;
  Eigen::VectorXi n2o_indices;
  Eigen::VectorXi other_gh_indices;
  Eigen::VectorXi ghg_forward_indices;
  Eigen::VectorXi ghg_inverse_indices;
  species_struct(int i) {
    name = std::vector<std::string>(i);
    type = std::vector<std::string>(i);
    input_mode = std::vector<std::string>(i);
    greenhouse_gas = Eigen::VectorXi::Zero(i);
    aerosol_chemistry_from_emissions = Eigen::VectorXi::Zero(i);
    aerosol_chemistry_from_concentration = Eigen::VectorXi::Zero(i);
    partition_fraction = Eigen::MatrixXd::Zero(4, i);
    unperturbed_lifetime = Eigen::MatrixXd::Zero(4, i);
    tropospheric_adjustment = Eigen::VectorXd::Zero(i);
    forcing_efficacy = Eigen::VectorXd::Zero(i);
    forcing_temperature_feedback = Eigen::VectorXd::Zero(i);
    forcing_scale = Eigen::VectorXd::Zero(i);
    molecular_weight = Eigen::VectorXd::Zero(i);
    baseline_concentration = Eigen::VectorXd::Zero(i);
    forcing_reference_concentration = Eigen::VectorXd::Zero(i);
    forcing_reference_emissions = Eigen::VectorXd::Zero(i);
    iirf_0 = Eigen::VectorXd::Zero(i);
    iirf_airborne = Eigen::VectorXd::Zero(i);
    iirf_uptake = Eigen::VectorXd::Zero(i);
    iirf_temperature = Eigen::VectorXd::Zero(i);
    baseline_emissions = Eigen::VectorXd::Zero(i);
    g1 = Eigen::VectorXd::Zero(i);
    g0 = Eigen::VectorXd::Zero(i);
    greenhouse_gas_radiative_efficiency = Eigen::VectorXd::Zero(i);
    contrails_radiative_efficiency = Eigen::VectorXd::Zero(i);
    erfari_radiative_efficiency = Eigen::VectorXd::Zero(i);
    h2o_stratospheric_factor = Eigen::VectorXd::Zero(i);
    lapsi_radiative_efficiency = Eigen::VectorXd::Zero(i);
    land_use_cumulative_emissions_to_forcing = Eigen::VectorXd::Zero(i);
    ozone_radiative_efficiency = Eigen::VectorXd::Zero(i);
    cl_atoms = Eigen::VectorXd::Zero(i);
    br_atoms = Eigen::VectorXd::Zero(i);
    fractional_release = Eigen::VectorXd::Zero(i);
    ch4_lifetime_chemical_sensitivity = Eigen::VectorXd::Zero(i);
    aci_shape = Eigen::VectorXd::Zero(i);
    aci_scale = Eigen::VectorXd::Zero(i);
    lifetime_temperature_sensitivity = Eigen::VectorXd::Zero(i);
    concentration_per_emission = Eigen::VectorXd::Zero(i);
  }
  species_struct() {}
};

inline species_struct append_species(const species_struct& lhs, const species_struct& rhs)
{
  species_struct new_s = species_struct(lhs.name.size()+rhs.name.size());
  new_s.name = std::vector<std::string>();
  new_s.name.insert(new_s.name.end(), lhs.name.begin(), lhs.name.end());
  new_s.name.insert(new_s.name.end(), rhs.name.begin(), rhs.name.end());
  new_s.type = std::vector<std::string>();
  new_s.type.insert(new_s.type.end(), lhs.type.begin(), lhs.type.end());
  new_s.type.insert(new_s.type.end(), rhs.type.begin(), rhs.type.end());
  new_s.input_mode = std::vector<std::string>();
  new_s.input_mode.insert(new_s.input_mode.end(), lhs.input_mode.begin(), lhs.input_mode.end());
  new_s.input_mode.insert(new_s.input_mode.end(), rhs.input_mode.begin(), rhs.input_mode.end());
  new_s.greenhouse_gas << lhs.greenhouse_gas, rhs.greenhouse_gas;
  new_s.aerosol_chemistry_from_emissions << lhs.aerosol_chemistry_from_emissions, rhs.aerosol_chemistry_from_emissions;
  new_s.aerosol_chemistry_from_concentration << lhs.aerosol_chemistry_from_concentration, rhs.aerosol_chemistry_from_concentration;
  new_s.partition_fraction << lhs.partition_fraction, rhs.partition_fraction;
  new_s.unperturbed_lifetime << lhs.unperturbed_lifetime, rhs.unperturbed_lifetime;
  new_s.tropospheric_adjustment << lhs.tropospheric_adjustment, rhs.tropospheric_adjustment;
  new_s.forcing_efficacy << lhs.forcing_efficacy, rhs.forcing_efficacy;
  new_s.forcing_temperature_feedback << lhs.forcing_temperature_feedback, rhs.forcing_temperature_feedback;
  new_s.forcing_scale << lhs.forcing_scale, rhs.forcing_scale;
  new_s.molecular_weight << lhs.molecular_weight, rhs.molecular_weight;
  new_s.baseline_concentration << lhs.baseline_concentration, rhs.baseline_concentration;
  new_s.forcing_reference_concentration << lhs.forcing_reference_concentration, rhs.forcing_reference_concentration;
  new_s.forcing_reference_emissions << lhs.forcing_reference_emissions, rhs.forcing_reference_emissions;
  new_s.iirf_0 << lhs.iirf_0, rhs.iirf_0;
  new_s.iirf_airborne << lhs.iirf_airborne, rhs.iirf_airborne;
  new_s.iirf_uptake << lhs.iirf_uptake, rhs.iirf_uptake;
  new_s.iirf_temperature << lhs.iirf_temperature, rhs.iirf_temperature;
  new_s.baseline_emissions << lhs.baseline_emissions, rhs.baseline_emissions;
  new_s.g1 << lhs.g1, rhs.g1;
  new_s.g0 << lhs.g0, rhs.g0;
  new_s.greenhouse_gas_radiative_efficiency << lhs.greenhouse_gas_radiative_efficiency, rhs.greenhouse_gas_radiative_efficiency;
  new_s.contrails_radiative_efficiency << lhs.contrails_radiative_efficiency, rhs.contrails_radiative_efficiency;
  new_s.erfari_radiative_efficiency << lhs.erfari_radiative_efficiency, rhs.erfari_radiative_efficiency;
  new_s.h2o_stratospheric_factor << lhs.h2o_stratospheric_factor, rhs.h2o_stratospheric_factor;
  new_s.lapsi_radiative_efficiency << lhs.lapsi_radiative_efficiency, rhs.lapsi_radiative_efficiency;
  new_s.land_use_cumulative_emissions_to_forcing << lhs.land_use_cumulative_emissions_to_forcing, rhs.land_use_cumulative_emissions_to_forcing;
  new_s.ozone_radiative_efficiency << lhs.ozone_radiative_efficiency, rhs.ozone_radiative_efficiency;
  new_s.cl_atoms << lhs.cl_atoms, rhs.cl_atoms;
  new_s.br_atoms << lhs.br_atoms, rhs.br_atoms;
  new_s.fractional_release << lhs.fractional_release, rhs.fractional_release;
  new_s.ch4_lifetime_chemical_sensitivity << lhs.ch4_lifetime_chemical_sensitivity, rhs.ch4_lifetime_chemical_sensitivity;
  new_s.aci_shape << lhs.aci_shape, rhs.aci_shape;
  new_s.aci_scale << lhs.aci_scale, rhs.aci_scale;
  new_s.lifetime_temperature_sensitivity << lhs.lifetime_temperature_sensitivity, rhs.lifetime_temperature_sensitivity;
  new_s.concentration_per_emission << lhs.concentration_per_emission, rhs.concentration_per_emission;
  return new_s;
}

inline bool operator==(const species_struct& lhs, const species_struct& rhs) {
  return 0;
}