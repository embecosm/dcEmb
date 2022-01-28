/**
 * Structure for locating parameters within the  COVID-19 DCM for the dcEmb
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

#pragma once

struct parameter_location_COVID {
  /**
   * @brief number of initial cases
   */
  int init_cases = -1;
  /**
   * @brief size of population with mixing
   */
  int pop_size = -1;
  /**
   * @brief initial proportion
   */
  int init_prop = -1;
  /**
   * @brief P(going home | work)
   */
  int p_home_work = -1;
  /**
   * @brief social distancing threshold
   */
  int social_dist = -1;
  /**
   * @brief bed availability threshold (per capita)
   */
  int bed_thresh = -1;
  /**
   * @brief effective number of contacts: home
   */
  int home_contacts = -1;
  /**
   * @brief effective number of contacts: work
   */
  int work_contacts = -1;
  /**
   * @brief P(transmission | infectious)
   */
  int p_conta_contact = -1;
  /**
   * @brief infected (pre-contagious) period
   */
  int infed_period = -1;
  /**
   * @brief contagious period
   */
  int infious_period = -1;
  /**
   * @brief time until symptoms
   */
  int tt_symptoms = -1;
  /**
   * @brief P(severe symptoms | symptomatic)
   */
  int p_sev_symp = -1;
  /**
   * @brief symptomatic period
   */
  int symp_period = -1;
  /**
   * @brief period in CCU
   */
  int ccu_period = -1;
  /**
   * @brief P(fatality | CCU)
   */
  int p_fat_sevccu = -1;
  /**
   * @brief P(fatality | home)
   */
  int p_surv_sevhome = -1;
  /**
   * @brief test, track and trace
   */
  int test_track_trace = -1;
  /**
   * @brief testing latency (months)
   */
  int test_lat = -1;
  /**
   * @brief test delay (days)
   */
  int test_del = -1;
  /**
   * @brief test selectivity (for infection)
   */
  int test_selec = -1;
  /**
   * @brief sustained testing
   */
  int subs_testing = -1;
  /**
   * @brief baseline testing
   */
  int base_testing = -1;
  /**
   * @brief period of immunity
   */
  int imm_period = -1;
  /**
   * @brief period of exemption
   */
  int exmp_period = -1;
  /**
   * @brief proportion of people not susceptible
   */
  int prop_res = -1;
  /**
   * @brief proportion with innate immunity
   */
  int prop_imm = -1;
  /**
   * @brief testing buildup
   */
  int test_buildup = -1;
};

inline bool operator==(const parameter_location_COVID& lhs,
                       const parameter_location_COVID& rhs) {
  return lhs.init_cases == rhs.init_cases & lhs.pop_size == rhs.pop_size &
         lhs.init_prop == rhs.init_prop & lhs.p_home_work == rhs.p_home_work &
         lhs.social_dist == rhs.social_dist & lhs.bed_thresh == rhs.bed_thresh &
         lhs.home_contacts == rhs.home_contacts &
         lhs.work_contacts == rhs.work_contacts &
         lhs.p_conta_contact == rhs.p_conta_contact &
         lhs.infed_period == rhs.infed_period &
         lhs.infious_period == rhs.infious_period &
         lhs.tt_symptoms == rhs.tt_symptoms & lhs.p_sev_symp == rhs.p_sev_symp &
         lhs.symp_period == rhs.symp_period & lhs.ccu_period == rhs.ccu_period &
         lhs.p_fat_sevccu == rhs.p_fat_sevccu &
         lhs.p_surv_sevhome == rhs.p_surv_sevhome &
         lhs.test_track_trace == rhs.test_track_trace &
         lhs.test_lat == rhs.test_lat & lhs.test_del == rhs.test_del &
         lhs.test_selec == rhs.test_selec &
         lhs.subs_testing == rhs.subs_testing &
         lhs.base_testing == rhs.base_testing &
         lhs.imm_period == rhs.imm_period & lhs.exmp_period == rhs.exmp_period &
         lhs.prop_res == rhs.prop_res & lhs.prop_imm == rhs.prop_imm &
         lhs.test_buildup == rhs.test_buildup;
}
