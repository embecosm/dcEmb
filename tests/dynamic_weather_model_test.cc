/**
 * Tests of 3-body functions for the dcEmb package
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

#include "dynamic_weather_model.hh"
#include <gtest/gtest.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include "bmr_model.hh"
#include "peb_model.hh"
#include "species_struct.hh"
#include "tests/dynamic_weather_model_test.hh"
#include "utility.hh"
#define MSL model.species_list

TEST(species_from_string_test, unit) {
  std::string s =
      "CO2,co2,calculated,1,0,0,0.2173,0.224,0.2824,0.2763,1000000000,394.4,"
      "36.54,4.304,0.05,1,0,1,44.009,278.3,277.15,0,29,0.000819,0.00846,4,0,"
      "11.41262243,0.010178288,0.0000133,1,2,3,4,nan,5,6,7,8,nan,9,10,nan";
  species species = utility::species_from_string(s);

  Eigen::VectorXd part(4);
  part << 0.2173, 0.224, 0.2824, 0.2763;
  Eigen::VectorXd unper(4);
  unper << 1000000000, 394.4, 36.54, 4.304;
  EXPECT_EQ(species.name, "CO2");
  EXPECT_EQ(species.type, "co2");
  EXPECT_EQ(species.input_mode, "calculated");
  EXPECT_EQ(species.greenhouse_gas, true);
  EXPECT_EQ(species.aerosol_chemistry_from_emissions, false);
  EXPECT_EQ(species.aerosol_chemistry_from_concentration, false);
  EXPECT_EQ(species.partition_fraction, part);
  EXPECT_EQ(species.unperturbed_lifetime, unper);
  EXPECT_EQ(species.tropospheric_adjustment, 0.05);
  EXPECT_EQ(species.forcing_efficacy, 1);
  EXPECT_EQ(species.forcing_temperature_feedback, 0);
  EXPECT_EQ(species.forcing_scale, 1);
  EXPECT_EQ(species.molecular_weight, 44.009);
  EXPECT_EQ(species.baseline_concentration, 278.3);
  EXPECT_EQ(species.forcing_reference_concentration, 277.15);
  EXPECT_EQ(species.forcing_reference_emissions, 0);
  EXPECT_EQ(species.iirf_0, 29);
  EXPECT_EQ(species.iirf_airborne, 0.000819);
  EXPECT_EQ(species.iirf_uptake, 0.00846);
  EXPECT_EQ(species.iirf_temperature, 4);
  EXPECT_EQ(species.baseline_emissions, 0);
  EXPECT_EQ(species.g1, 11.41262243);
  EXPECT_EQ(species.g0, 0.010178288);
  EXPECT_EQ(species.greenhouse_gas_radiative_efficiency, 0.0000133);
  EXPECT_EQ(species.contrails_radiative_efficiency, 1);
  EXPECT_EQ(species.erfari_radiative_efficiency, 2);
  EXPECT_EQ(species.h2o_stratospheric_factor, 3);
  EXPECT_EQ(species.lapsi_radiative_efficiency, 4);
  EXPECT_EQ(species.land_use_cumulative_emissions_to_forcing, 0);
  EXPECT_EQ(species.ozone_radiative_efficiency, 5);
  EXPECT_EQ(species.cl_atoms, 6);
  EXPECT_EQ(species.br_atoms, 7);
  EXPECT_EQ(species.fractional_release, 8);
  EXPECT_EQ(species.aci_shape, 0);
  EXPECT_EQ(species.aci_scale, 9);
  EXPECT_EQ(species.ch4_lifetime_chemical_sensitivity, 10);
  EXPECT_EQ(species.lifetime_temperature_sensitivity, 0);
  EXPECT_EQ(species.concentration_per_emission, 0);
}

TEST(species_from_file_test, unit) {
  std::string filename = "../src/weather/data/species_configs_properties.csv";
  std::vector<std::string> names{"CH4", "N2O"};
  species_struct species = utility::species_from_file(filename, names);
  Eigen::VectorXd part0(4);
  part0 << 1, 0, 0, 0;
  Eigen::VectorXd unper0(4);
  unper0 << 8.25, 8.25, 8.25, 8.25;
  EXPECT_EQ(species.name[0], "CH4");
  EXPECT_EQ(species.type[0], "ch4");
  EXPECT_EQ(species.input_mode[0], "emissions");
  EXPECT_EQ(species.greenhouse_gas(0), true);
  EXPECT_EQ(species.aerosol_chemistry_from_emissions(0), false);
  EXPECT_EQ(species.aerosol_chemistry_from_concentration(0), true);
  EXPECT_EQ(species.partition_fraction.col(0), part0);
  EXPECT_EQ(species.unperturbed_lifetime.col(0), unper0);
  EXPECT_EQ(species.tropospheric_adjustment(0), -0.14);
  EXPECT_EQ(species.forcing_efficacy(0), 1);
  EXPECT_EQ(species.forcing_temperature_feedback(0), 0);
  EXPECT_EQ(species.forcing_scale(0), 1);
  EXPECT_EQ(species.molecular_weight(0), 16.043);
  EXPECT_EQ(species.baseline_concentration(0), 729.2);
  EXPECT_EQ(species.forcing_reference_concentration(0), 731.41);
  EXPECT_EQ(species.forcing_reference_emissions(0), 0);
  EXPECT_EQ(species.iirf_0(0), 8.249955097);
  EXPECT_EQ(species.iirf_airborne(0), 0.00032);
  EXPECT_EQ(species.iirf_uptake(0), 0);
  EXPECT_EQ(species.iirf_temperature(0), -0.3);
  EXPECT_EQ(species.baseline_emissions(0), 0);
  EXPECT_EQ(species.g1(0), 8.249410814);
  EXPECT_EQ(species.g0(0), 0.36785517);
  EXPECT_EQ(species.greenhouse_gas_radiative_efficiency(0), 0.000388644);
  EXPECT_EQ(species.contrails_radiative_efficiency(0), 0);
  EXPECT_EQ(species.erfari_radiative_efficiency(0), -0.00000227);
  EXPECT_EQ(species.h2o_stratospheric_factor(0), 0.091914639);
  EXPECT_EQ(species.lapsi_radiative_efficiency(0), 0);
  EXPECT_EQ(species.land_use_cumulative_emissions_to_forcing(0), 0);
  EXPECT_EQ(species.ozone_radiative_efficiency(0), 0.000175);
  EXPECT_EQ(species.cl_atoms(0), 0);
  EXPECT_EQ(species.br_atoms(0), 0);
  EXPECT_EQ(species.fractional_release(0), 0);
  EXPECT_EQ(species.aci_shape(0), 0);
  EXPECT_EQ(species.aci_scale(0), 0);
  EXPECT_EQ(species.ch4_lifetime_chemical_sensitivity(0), 0.000254099);
  EXPECT_EQ(species.lifetime_temperature_sensitivity(0), -0.0408);
  EXPECT_EQ(species.concentration_per_emission(0), 0.3516458926201187);

  Eigen::VectorXd part1(4);
  part1 << 1, 0, 0, 0;
  Eigen::VectorXd unper1(4);
  unper1 << 109, 109, 109, 109;

  EXPECT_EQ(species.name[1], "N2O");
  EXPECT_EQ(species.type[1], "n2o");
  EXPECT_EQ(species.input_mode[1], "emissions");
  EXPECT_EQ(species.greenhouse_gas(1), true);
  EXPECT_EQ(species.aerosol_chemistry_from_emissions(1), false);
  EXPECT_EQ(species.aerosol_chemistry_from_concentration(1), true);
  EXPECT_EQ(species.partition_fraction.col(1), part1);
  EXPECT_EQ(species.unperturbed_lifetime.col(1), unper1);
  EXPECT_EQ(species.tropospheric_adjustment(1), 0.07);
  EXPECT_EQ(species.forcing_efficacy(1), 1);
  EXPECT_EQ(species.forcing_temperature_feedback(1), 0);
  EXPECT_EQ(species.forcing_scale(1), 1);
  EXPECT_EQ(species.molecular_weight(1), 44.013);
  EXPECT_EQ(species.baseline_concentration(1), 270.1);
  EXPECT_EQ(species.forcing_reference_concentration(1), 273.87);
  EXPECT_EQ(species.forcing_reference_emissions(1), 0);
  EXPECT_EQ(species.iirf_0(1), 65.44969575);
  EXPECT_EQ(species.iirf_airborne(1), -0.0065);
  EXPECT_EQ(species.iirf_uptake(1), 0);
  EXPECT_EQ(species.iirf_temperature(1), 0);
  EXPECT_EQ(species.baseline_emissions(1), 0);
  EXPECT_EQ(species.g1(1), 25.49528818);
  EXPECT_EQ(species.g0(1), 0.076755588);
  EXPECT_EQ(species.greenhouse_gas_radiative_efficiency(1), 0.003195507);
  EXPECT_EQ(species.contrails_radiative_efficiency(1), 0);
  EXPECT_EQ(species.erfari_radiative_efficiency(1), -0.0000364);
  EXPECT_EQ(species.h2o_stratospheric_factor(1), 0);
  EXPECT_EQ(species.lapsi_radiative_efficiency(1), 0);
  EXPECT_EQ(species.land_use_cumulative_emissions_to_forcing(1), 0);
  EXPECT_EQ(species.ozone_radiative_efficiency(1), 0.00071);
  EXPECT_EQ(species.cl_atoms(1), 0);
  EXPECT_EQ(species.br_atoms(1), 0);
  EXPECT_EQ(species.fractional_release(1), 0);
  EXPECT_EQ(species.aci_shape(1), 0);
  EXPECT_EQ(species.aci_scale(1), 0);
  EXPECT_EQ(species.ch4_lifetime_chemical_sensitivity(1), -0.000722665);
  EXPECT_EQ(species.lifetime_temperature_sensitivity(1), 0);
  EXPECT_EQ(species.concentration_per_emission(1), 0.128177017138222);
}

TEST(meinshausen, unit) {
  dynamic_weather_model model = define_minimal_weather_model();

  Eigen::VectorXd forcing_scaling =
      MSL.forcing_scale.array() *
      (1 + MSL.tropospheric_adjustment.array());
  Eigen::VectorXd reference_concentration(3);
  reference_concentration << 277, 731, 270;
  Eigen::VectorXd concentration_t0(3);
  concentration_t0 << 277, 731, 270;
  Eigen::VectorXd concentration_t1(3);
  concentration_t1 << 410, 1900, 325;
  Eigen::VectorXd ghg_forcing_t0 = model.meinshausen(
      concentration_t0, reference_concentration, forcing_scaling,
      MSL.greenhouse_gas_radiative_efficiency, MSL.co2_indices,
      MSL.ch4_indices, MSL.n2o_indices, MSL.other_gh_indices);
  Eigen::VectorXd ghg_forcing_t1 = model.meinshausen(
      concentration_t1, reference_concentration, forcing_scaling,
      MSL.greenhouse_gas_radiative_efficiency,
      MSL.co2_indices, MSL.ch4_indices,
      MSL.n2o_indices, MSL.other_gh_indices);
  Eigen::VectorXd expected_t0(3);
  expected_t0 << 0, 0, 0;
  Eigen::VectorXd expected_t1(3);
  expected_t1 << 2.1849852021506635, 0.5557465907740132, 0.18577100600435772;
  for (int i = 0; i < 3; i++) {
    EXPECT_DOUBLE_EQ(ghg_forcing_t0(i), expected_t0(i));
    EXPECT_DOUBLE_EQ(ghg_forcing_t1(i), expected_t1(i));
  }
}

TEST(calculate_alpha, unit) {
  dynamic_weather_model model = define_minimal_weather_model();

  Eigen::VectorXd forcing_array(3);
  forcing_array << 0, 0, 0;
  Eigen::VectorXd temperature(3);
  temperature << 0, 0, 0;
  Eigen::VectorXd cummins_state_array(3);
  cummins_state_array << 0, 0, 0;

  Eigen::VectorXd airborne_emissions(3);
  airborne_emissions << 0, 0, 0;
  Eigen::VectorXd cumulative_emissions(3);
  cumulative_emissions << 0, 0, 0;
  Eigen::VectorXd ghg_forcing_t0 = model.calculate_alpha(
      airborne_emissions, cumulative_emissions, MSL.g0,
      MSL.g1, MSL.iirf_0,
      MSL.iirf_airborne, MSL.iirf_temperature,
      MSL.iirf_uptake, cummins_state_array, 100);

  Eigen::VectorXd expected_t0(3);
  expected_t0 << 0.1291924236217484, 1.0000000003481242, 0.9999999950113339;
  for (int i = 0; i < 3; i++) {
    EXPECT_DOUBLE_EQ(ghg_forcing_t0(i), expected_t0(i));
  }
}

TEST(step_concentration, unit) {
  dynamic_weather_model model = define_minimal_weather_model();
  utility::update_species_list_indicies(model.species_list);
  Eigen::VectorXd emissions_array(3);
  emissions_array << 0, 0, 0;
  Eigen::MatrixXd gas_partitions_array(4, 3);
  gas_partitions_array << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  Eigen::VectorXd airborne_emissions_array(3);
  airborne_emissions_array << 0, 0, 0;
  Eigen::VectorXd alpha_lifetime_array(3);
  alpha_lifetime_array << 0, 0, 0;
  std::vector<Eigen::MatrixXd> out = model.step_concentration(
      emissions_array(MSL.ghg_forward_indices),
      gas_partitions_array(Eigen::all, MSL.ghg_forward_indices),
      airborne_emissions_array(MSL.ghg_forward_indices),
      alpha_lifetime_array(MSL.ghg_forward_indices),
      MSL.baseline_concentration(MSL.ghg_forward_indices),
      MSL.baseline_emissions(MSL.ghg_forward_indices),
      MSL.concentration_per_emission(MSL.ghg_forward_indices),
      MSL.unperturbed_lifetime(Eigen::all, MSL.ghg_forward_indices),
      MSL.partition_fraction(Eigen::all, MSL.ghg_forward_indices), 270);

  Eigen::VectorXd concentration_out = out[0];
  Eigen::MatrixXd gasboxes_new = out[1];
  Eigen::VectorXd airborne_emissions_new = out[2];

  Eigen::VectorXd concentration_test(3);
  concentration_test << 278.3, 729.2, 270.1;
  Eigen::MatrixXd gasboxes_test(4, 3);
  gasboxes_test << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  Eigen::VectorXd airborne_emissions_test(3);
  airborne_emissions_test << 0, 0, 0;

  EXPECT_EQ(concentration_out, concentration_test);
  EXPECT_EQ(gasboxes_new, gasboxes_test);
  EXPECT_EQ(airborne_emissions_new, airborne_emissions_test);
}

TEST(unstep_concentration, unit) {
  dynamic_weather_model model = define_minimal_weather_model();
  MSL.baseline_concentration << 277, 731, 270;
  Eigen::VectorXd concentration_array(3);
  concentration_array << 410, 1900, 325;
  Eigen::MatrixXd gas_partitions_array(4, 3);
  gas_partitions_array << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  Eigen::VectorXd airborne_emissions_array(3);
  airborne_emissions_array << 0, 0, 0;

  Eigen::VectorXd forcing_array(3);
  forcing_array << 0, 0, 0;
  Eigen::VectorXd temperature(3);
  temperature << 0, 0, 0;
  Eigen::VectorXd cummins_state_array(3);
  cummins_state_array << 0, 0, 0;

  Eigen::VectorXd airborne_emissions(3);
  airborne_emissions << 0, 0, 0;
  Eigen::VectorXd cumulative_emissions(3);
  cumulative_emissions << 0, 0, 0;
  Eigen::VectorXd alpha_lifetime_array = model.calculate_alpha(
      airborne_emissions, cumulative_emissions, MSL.g0,
      MSL.g1, MSL.iirf_0,
      MSL.iirf_airborne, MSL.iirf_temperature,
      MSL.iirf_uptake, cummins_state_array, 100);

  std::vector<Eigen::MatrixXd> out = model.unstep_concentration(
      concentration_array(MSL.ghg_forward_indices),
      gas_partitions_array(Eigen::all, MSL.ghg_forward_indices),
      airborne_emissions_array(MSL.ghg_forward_indices),
      alpha_lifetime_array(MSL.ghg_forward_indices),
      MSL.baseline_concentration(MSL.ghg_forward_indices),
      MSL.baseline_emissions(MSL.ghg_forward_indices),
      MSL.concentration_per_emission(MSL.ghg_forward_indices),
      MSL.unperturbed_lifetime(Eigen::all, MSL.ghg_forward_indices),
      MSL.partition_fraction(Eigen::all, MSL.ghg_forward_indices), 270);

  Eigen::VectorXd emissions_out = out[0];
  Eigen::MatrixXd gasboxes_new = out[1];
  Eigen::VectorXd airborne_emissions_new = out[2];

  Eigen::VectorXd emissions_test(3);
  emissions_test << 14.5080631466396, 402.9535752340407, 4.297595650508209;
  Eigen::MatrixXd gasboxes_test(4, 3);
  gasboxes_test << 851.2016834307615, 3324.366996838108, 429.09408588194685,
      164.7615496725284, 0.0, 0.0, 19.34103532615702, 0.0, 0.0,
      2.228946427301561, 0.0, 0.0;

  Eigen::VectorXd airborne_emissions_test(3);
  airborne_emissions_test << 1037.5332148567484, 3324.3669968381087,
      429.09408588194685;

  EXPECT_EQ(emissions_out, emissions_test);
  EXPECT_EQ(gasboxes_new, gasboxes_test);
  EXPECT_EQ(airborne_emissions_new, airborne_emissions_test);
}


dynamic_weather_model define_minimal_weather_model() {
  dynamic_weather_model model;
  std::vector<std::string> species_names{"CO2", "CH4", "N2O"};
  std::string filename = "../src/weather/data/species_configs_properties.csv";
  model.species_list = utility::species_from_file(filename, species_names);

  int num_species = model.species_list.name.size();
  model.num_samples = 2;

  return model;
}