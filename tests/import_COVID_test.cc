/**
 * Tests of COVID functions for the dcEmb package
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

#include "import_COVID.hh"
#include <gtest/gtest.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <iostream>

TEST(smooth_test, short) {
  Eigen::VectorXi vec1(15);
  vec1 << 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9;
  Eigen::VectorXd vec2(21);
  vec2 << 0.116121239359018, 0.146061753488993, 0.203268031157222,
      0.2828325970072, 0.378412544968342, 0.483038431515524, 0.589456447651173,
      0.689956263167915, 0.776452312179117, 0.841851539709751,
      0.882747899623155, 0.901838879774914, 0.907817258622345,
      0.911896812376654, 0.922684980965641, 0.942598708705646,
      0.968011651177221, 0.992629650409526, 1.01147650162011, 1.0229130752009,
      1.02793342131963;
  Eigen::VectorXd vec3 = smooth(vec1, 5);
  for (int i = 0; i < 21; i++) {
    EXPECT_FLOAT_EQ(vec2(i), vec3(i));
    EXPECT_NE(0.0, vec3(i));
  }
}

TEST(matrix_power_test, positive) {
  Eigen::MatrixXd mat1(2, 2);
  mat1 << 1, 2, 3, 4;
  Eigen::MatrixXd mat2(2, 2);
  mat2 << 37, 54, 81, 118;
  Eigen::MatrixXd mat3(2, 2);
  mat3 << 4783807, 6972050, 10458075, 15241882;
  EXPECT_EQ(mat_power(mat1, 3), mat2);
  EXPECT_EQ(mat_power(mat1, 10), mat3);
}

TEST(matrix_power_test, negative) {
  Eigen::MatrixXd mat1(2, 2);
  mat1 << 1, -2, 3, -4;
  Eigen::MatrixXd mat2(2, 2);
  mat2 << 13, -14, 21, -22;
  Eigen::MatrixXd mat3(2, 2);
  mat3 << -2045, 2046, -3069, 3070;
  EXPECT_EQ(mat_power(mat1, 3), mat2);
  EXPECT_EQ(mat_power(mat1, 10), mat3);
}

TEST(matrix_power_test, identity) {
  EXPECT_EQ(mat_power(Eigen::MatrixXd::Identity(2, 2), 20),
            Eigen::MatrixXd::Identity(2, 2));
}

TEST(zero_padded_add_tests, unpadded) {
  EXPECT_EQ(zero_padded_add(Eigen::VectorXd::Constant(5, 5.0),
                            Eigen::VectorXd::Constant(5, 5.0)),
            Eigen::VectorXd::Constant(5, 10.0));
  EXPECT_EQ(zero_padded_add(Eigen::VectorXd::Constant(5, 5.0),
                            Eigen::VectorXd::Constant(5, -5.0)),
            Eigen::VectorXd::Constant(5, 0.0));
  EXPECT_NE(zero_padded_add(Eigen::VectorXd::Constant(5, 5.0),
                            Eigen::VectorXd::Constant(5, 5.0)),
            Eigen::VectorXd::Constant(5, 0.0));
}

TEST(zero_padded_add_tests, padded) {
  Eigen::VectorXd vec1(7);
  vec1 << 0, 0, 5, 5, 5, 5, 5;
  Eigen::VectorXd vec2(5);
  vec2 << 5, 5, 5, 5, 5;
  Eigen::VectorXd vec3(7);
  vec3 << 0, 0, 10, 10, 10, 10, 10;
  EXPECT_EQ(zero_padded_add(vec1, vec2), vec3);
  EXPECT_EQ(zero_padded_add(vec2, vec1), vec3);
  EXPECT_NE(zero_padded_add(vec2, vec3), vec1);
}
