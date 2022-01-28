/**
 * Tests of COVID feature selection functions for the dcEmb package
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

#include "feature_selection_COVID.hh"
#include <gtest/gtest.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <iostream>

/**
 * Test COVID feature selection against square roots
 */
TEST(feature_select_COVID_test, short) {
  Eigen::VectorXd vec1(9);
  vec1 << 3, -1, 4, 0.5, 5, 9, -2, 0.25, 5;
  Eigen::VectorXd vec2(9);
  vec2 << sqrt(3), sqrt(1), sqrt(4), sqrt(1), sqrt(5), sqrt(9), sqrt(1),
      sqrt(1), sqrt(5);
  feature_selection_COVID COVID_fs;
  COVID_fs.eval_features(vec1);
  Eigen::VectorXd vec3 = COVID_fs.get_fs_response_vars();
  EXPECT_EQ(vec2, vec3);
}
