/**
 * A set of serialization functions for the dcEmb package
 *
 * Copyright (C) 2022 Embecosm Limited
 *
 * Contributor William Jones <william.jones@embecosm.com>
 * Contributor Elliot Stein <E.Stein@soton.ac.uk>
 *
 * This file is part of the Embecosm Dynamic Causal Modeling package
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <Eigen/Core>
#include <cereal/archives/binary.hpp>
#include "bmr_model.hh"
#include "dynamic_3body_model.hh"
#include "dynamic_COVID_model.hh"
#include "dynamic_model.hh"
#include "parameter_location_3body.hh"
#include "parameter_location_COVID.hh"
#include "peb_model.hh"

#pragma once

namespace cereal {

template <class Archive>
void serialize(Archive& ar, dynamic_COVID_model& dCm,
               const unsigned int version) {
  ar& dCm.parameter_locations;
  ar& dCm.max_invert_it;
  ar& dCm.conditional_parameter_expectations;
  ar& dCm.conditional_parameter_covariances;
  ar& dCm.conditional_hyper_expectations;
  ar& dCm.free_energy;
  ar& dCm.prior_parameter_expectations;
  ar& dCm.prior_parameter_covariances;
  ar& dCm.prior_hyper_expectations;
  ar& dCm.prior_hyper_covariances;
  ar& dCm.num_samples;
  ar& dCm.num_response_vars;
  ar& dCm.select_response_vars;
  ar& dCm.response_vars;
}

template <class Archive>
void serialize(Archive& ar, peb_model<dynamic_COVID_model>& peb,
               const unsigned int version) {
  ar& peb.GCM;
  ar& peb.empirical_GCM;
  ar& peb.random_effects;
  ar& peb.between_design_matrix;
  ar& peb.within_design_matrix;
  ar& peb.random_precision_comps;
  ar& peb.expected_covariance_random_effects;
  ar& peb.singular_matrix;
  ar& peb.precision_random_effects;
  ar& peb.max_invert_it;
  ar& peb.conditional_parameter_expectations;
  ar& peb.conditional_parameter_covariances;
  ar& peb.conditional_hyper_expectations;
  ar& peb.free_energy;
  ar& peb.prior_parameter_expectations;
  ar& peb.prior_parameter_covariances;
  ar& peb.prior_hyper_expectations;
  ar& peb.prior_hyper_covariances;
  ar& peb.num_samples;
  ar& peb.num_response_vars;
  ar& peb.select_response_vars;
  ar& peb.response_vars;
}

template <class Archive>
void serialize(Archive& ar, parameter_location_COVID& pCm,
               const unsigned int version) {
  ar& pCm.init_cases;
  ar& pCm.pop_size;
  ar& pCm.init_prop;
  ar& pCm.p_home_work;
  ar& pCm.social_dist;
  ar& pCm.bed_thresh;
  ar& pCm.home_contacts;
  ar& pCm.work_contacts;
  ar& pCm.p_conta_contact;
  ar& pCm.infed_period;
  ar& pCm.infious_period;
  ar& pCm.tt_symptoms;
  ar& pCm.p_sev_symp;
  ar& pCm.symp_period;
  ar& pCm.ccu_period;
  ar& pCm.p_fat_sevccu;
  ar& pCm.p_surv_sevhome;
  ar& pCm.test_track_trace;
  ar& pCm.test_lat;
  ar& pCm.test_del;
  ar& pCm.test_selec;
  ar& pCm.subs_testing;
  ar& pCm.base_testing;
  ar& pCm.imm_period;
  ar& pCm.exmp_period;
  ar& pCm.prop_res;
  ar& pCm.prop_imm;
  ar& pCm.test_buildup;
}

// 3body templates
template <class Archive>
void serialize(Archive& ar, dynamic_3body_model& d3m,
               const unsigned int version) {
  ar& d3m.parameter_locations;
  ar& d3m.max_invert_it;
  ar& d3m.conditional_parameter_expectations;
  ar& d3m.conditional_parameter_covariances;
  ar& d3m.conditional_hyper_expectations;
  ar& d3m.free_energy;
  ar& d3m.prior_parameter_expectations;
  ar& d3m.prior_parameter_covariances;
  ar& d3m.prior_hyper_expectations;
  ar& d3m.prior_hyper_covariances;
  ar& d3m.num_samples;
  ar& d3m.num_response_vars;
  ar& d3m.select_response_vars;
  ar& d3m.response_vars;
}

template <class Archive>
void serialize(Archive& ar, parameter_location_3body& p3m,
               const unsigned int version) {
  ar& p3m.planet_coordsX;
  ar& p3m.planet_coordsY;
  ar& p3m.planet_coordsZ;
  ar& p3m.planet_masses;
  ar& p3m.planet_velocityX;
  ar& p3m.planet_velocityY;
  ar& p3m.planet_velocityZ;
  ar& p3m.planet_accelerationX;
  ar& p3m.planet_accelerationY;
  ar& p3m.planet_accelerationZ;
}

// Eigen Templates
template <class Archive, class Derived>
inline typename std::enable_if<
    traits::is_output_serializable<BinaryData<typename Derived::Scalar>,
                                   Archive>::value,
    void>::type
save(Archive& ar, Eigen::PlainObjectBase<Derived> const& m) {
  typedef Eigen::PlainObjectBase<Derived> ArrT;
  if (ArrT::RowsAtCompileTime == Eigen::Dynamic) ar(m.rows());
  if (ArrT::ColsAtCompileTime == Eigen::Dynamic) ar(m.cols());
  ar(binary_data(m.data(), m.size() * sizeof(typename Derived::Scalar)));
}

template <class Archive, class Derived>
inline typename std::enable_if<
    traits::is_input_serializable<BinaryData<typename Derived::Scalar>,
                                  Archive>::value,
    void>::type
load(Archive& ar, Eigen::PlainObjectBase<Derived>& m) {
  typedef Eigen::PlainObjectBase<Derived> ArrT;
  Eigen::Index rows = ArrT::RowsAtCompileTime, cols = ArrT::ColsAtCompileTime;
  if (rows == Eigen::Dynamic) ar(rows);
  if (cols == Eigen::Dynamic) ar(cols);
  m.resize(rows, cols);
  ar(binary_data(m.data(),
                 static_cast<std::size_t>(rows * cols *
                                          sizeof(typename Derived::Scalar))));
}
}  // namespace cereal