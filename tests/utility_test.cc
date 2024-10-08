/**
 * Tests of utility functions for the dcEmb package
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

#include "utility.hh"
#include <gtest/gtest.h>
#include <stdio.h>
#include <eigen3/Eigen/Dense>
#include <functional>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>
#include <dynamic_model.hh>
#include <cstdlib>
#include <ctime>

#define SparseMD Eigen::SparseMatrix<double>
#define SparseVD Eigen::SparseVector<double>

TEST(utility_test, dx) {
  Eigen::VectorXd vec0(2);
  vec0 << 1, 2;
  Eigen::MatrixXd mat0(2, 2);
  mat0 << 3, 4, 5, 6;
  Eigen::VectorXd vec1(2);
  vec1 << 1, -2;
  Eigen::MatrixXd mat1(2, 2);
  mat1 << 3, 0, -5, -6;
  Eigen::VectorXd out0 = utility::dx(mat0, vec0, 0);
  Eigen::VectorXd out1 = utility::dx(mat1, vec1, 0);
  Eigen::VectorXd out2 = utility::dx(mat0, vec0, -4);
  Eigen::VectorXd out3 = utility::dx(mat1, vec1, -4);
  Eigen::VectorXd out4 = utility::dx(mat0, vec0, 4);
  Eigen::VectorXd out5 = utility::dx(mat1, vec1, 4);
  Eigen::VectorXd res0(2);
  res0 << 87.056616402654114, 135.59974494402826;
  Eigen::VectorXd res1(2);
  res1 << 0.34270499388249082, -0.37260430127462846;
  Eigen::VectorXd res2(2);
  res2 << 0.013911319795549125, 0.027386520673549249;
  Eigen::VectorXd res3(2);
  res3 << 0.0043451137811443153, -0.0085696061849260836;
  Eigen::VectorXd res4(2);
  res4 << 4.4563823049271026e+153, 6.9263216870868159e+153;
  Eigen::VectorXd res5(2);
  res5 << 19478931490598708, -10821628605888168;
  for (int i = 0; i < 2; i++) {
    EXPECT_DOUBLE_EQ(out0(i), res0(i));
    EXPECT_DOUBLE_EQ(out1(i), res1(i));
    EXPECT_DOUBLE_EQ(out2(i), res2(i));
    EXPECT_DOUBLE_EQ(out3(i), res3(i));
    EXPECT_DOUBLE_EQ(out4(i), res4(i));
    EXPECT_DOUBLE_EQ(out5(i), res5(i));
  }
}

TEST(utility_test, logdet) {
  Eigen::MatrixXd mat0(3, 3);
  mat0 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Eigen::MatrixXd mat1(3, 3);
  mat1 << 5, 0, 0, 0, 5, 0, 0, 0, 5;
  Eigen::MatrixXd mat2(3, 3);
  mat2 << -5, 0, 0, 0, -5, 0, 0, 0, -5;
  Eigen::MatrixXd mat3(3, 3);
  mat3 << -5, 0, 0, 0, 5, 0, 0, 0, -5;
  Eigen::MatrixXd mat4(3, 3);
  mat4 << -1, 2, -3, -4, -5, 6, 7, 8, -9;
  EXPECT_DOUBLE_EQ(2.8903717578961636, utility::logdet(mat0));
  EXPECT_DOUBLE_EQ(4.8283137373023006, utility::logdet(mat1));
  EXPECT_DOUBLE_EQ(0, utility::logdet(mat2));
  EXPECT_DOUBLE_EQ(1.6094379124341003, utility::logdet(mat3));
  EXPECT_DOUBLE_EQ(1.7917594692280563, utility::logdet(mat4));
}

TEST(utility_test, diff) {
  dynamic_model model;
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> test1 =
      [](const Eigen::VectorXd& a) {
        return a.unaryExpr([](double x) { return exp(x); });
      };

  Eigen::VectorXd vec1(2);
  vec1 << 1, 2;
  Eigen::MatrixXd mat1 = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd out1 = model.diff(test1, vec1, mat1);
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> test2 =
      [](const Eigen::VectorXd& a) {
        return a.unaryExpr([](double x) { return pow(x, 2); });
      };
  Eigen::VectorXd vec2(2);
  vec2 << 1, 2;
  Eigen::MatrixXd mat2 = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd out2 = model.diff(test2, vec2, mat2);
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> test3 =
      [](const Eigen::VectorXd& a) {
        return a.unaryExpr([](double x) { return log(x); });
      };
  Eigen::VectorXd vec3(2);
  vec3 << 1, 2;
  Eigen::MatrixXd mat3 = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd out3 = model.diff(test3, vec3, mat3);
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> test4 =
      [](const Eigen::VectorXd& a) {
        return a.unaryExpr([](double x) { return 1 / x; });
      };
  Eigen::VectorXd vec4(2);
  vec4 << 1, 2;
  Eigen::MatrixXd mat4 = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd out4 = model.diff(test4, vec4, mat4);
  Eigen::MatrixXd res1(2, 2);
  res1 << 2.71873782042874, 0, 0, 7.3902956136159546;
  Eigen::MatrixXd res2(2, 2);
  res2 << 2.0003354626273127, 0, 0, 4.00033546262734;
  Eigen::MatrixXd res3(2, 2);
  res3 << 0.999832306188021, 0, 0, 0.49995807185987362;
  Eigen::MatrixXd res4(2, 2);
  res4 << -0.999664649869217, 0, 0, -0.24995807420376087;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_DOUBLE_EQ(res1(i, j), out1(i, j));
      EXPECT_DOUBLE_EQ(res2(i, j), out2(i, j));
      EXPECT_DOUBLE_EQ(res3(i, j), out3(i, j));
      EXPECT_DOUBLE_EQ(res4(i, j), out4(i, j));
    }
  }
}

TEST(utility_test, softmax) {
  Eigen::VectorXd vec0(10);
  vec0 << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Eigen::VectorXd vec1(10);
  vec1 << 0, -1, 2, 3, -4, 5, 6, -7, 8, -9;
  Eigen::VectorXd vec2(10);
  vec2 << 0, 100, 2, 3, 4078, 5, 6, 7, 8, 9078;
  Eigen::VectorXd vec3(10);
  vec3 << 0, 100, 2, 3, -4078, 5, 6, 7, 8, 9078;
  Eigen::VectorXd sol0(10);
  sol0 << 7.801341612780744e-05, 0.0002120624514362328, 0.0005764455082375902,
      0.001566941350139081, 0.004259388198344144, 0.0115782175399118,
      0.03147285834468803, 0.08555209892803112, 0.2325547159025975,
      0.6321492583604866;
  Eigen::VectorXd sol1(10);
  sol1 << 0.000280767817533857, 0.0001032887078132808, 0.002074609154531994,
      0.005639372365919102, 5.142441957528133e-06, 0.04166963877453551,
      0.1132698218792723, 2.560271093172253e-07, 0.8369570681818256,
      3.464950135569788e-08;
  Eigen::VectorXd sol2(10);
  sol2 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  Eigen::VectorXd sol3(10);
  sol3 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

  for (int i = 0; i < 10; i++) {
    EXPECT_DOUBLE_EQ(utility::softmax(vec0)(i), sol0(i));
  }
  for (int i = 0; i < 10; i++) {
    EXPECT_DOUBLE_EQ(utility::softmax(vec1)(i), sol1(i));
  }
  for (int i = 0; i < 10; i++) {
    EXPECT_DOUBLE_EQ(utility::softmax(vec2)(i), sol2(i));
  }
  for (int i = 0; i < 10; i++) {
    EXPECT_DOUBLE_EQ(utility::softmax(vec3)(i), sol3(i));
  }
}

TEST(utility_test, normrnd) {
  Eigen::MatrixXd test0(3, 3);
  test0 << 3, 2, 1, 4, 5, 6, 7, 8, 9;
  Eigen::EigenSolver<Eigen::MatrixXd> eig_solver0(test0);
  Eigen::VectorXd eval0 = eig_solver0.eigenvalues().real();
  Eigen::MatrixXd evec0 = eig_solver0.eigenvectors().real();

  Eigen::MatrixXd test1(3, 3);
  test1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Eigen::EigenSolver<Eigen::MatrixXd> eig_solver1(test1);
  Eigen::VectorXd eval1 = eig_solver1.eigenvalues().real();
  Eigen::MatrixXd evec1 = eig_solver1.eigenvectors().real();

  Eigen::VectorXd mu(3);
  mu << 1, 2, 1;

  EXPECT_THROW(utility::normrnd(mu, eval1, evec1), std::runtime_error);
  EXPECT_NO_THROW(utility::normrnd(mu, eval0, evec0));
}

TEST(utility_test, selrnd) {
  Eigen::VectorXd test = Eigen::VectorXd::Ones(1000);
  for (int i = 0; i < 10000; i++) {
    utility::selrnd(test);
  }
}

TEST(utility_test, read_matrix) {
  Eigen::MatrixXd mat(13, 9);
  Eigen::MatrixXd mat2;

  mat << 1.6348095, 6.300586, 10.1087265, 12.646354, 13.033894, 14.202391,
      17.1069, 21.536491, 26.818232, 7.56754, 29.884348, 40.529175, 40.666817,
      38.786674, 35.457172, 25.866344, 13.666277, 5.80594, 0.84460443,
      4.7713275, 8.254514, 8.221595, 7.769095, 5.984839, 3.1336484, 1.2209771,
      0.36919603, 3.7227983, 8.60889, 7.13798, 4.3692884, 2.245598, 1.2048376,
      0.48439175, 0.18282649, 0.14405982, 2.5900445, 5.4463806, 4.10332,
      2.6499856, 1.4592574, 0.5490897, 0.1639817, 0.0999282, 0.15789205,
      3.3679705, 6.77626, 4.022664, 1.0069828, 0.2101618, 0.09600428,
      0.03744267, 0.032467835, 0.031171907, 0.954237, 2.556057, 1.8709981,
      0.44247383, 0.08270793, 0.043194786, 0.01601541, 0.016137041, 0.024156444,
      0.464544, 0.8164661, 0.4773866, 0.13414595, 0.036959376, 0.011390248,
      0.0069230758, 0.012365481, 0.018977778, 0.91555434, 1.3042815, 0.4444588,
      0.11638977, 0.03203234, 0.00928911, 0.0069186063, 0.007832967, 0.00999842,
      0.40533167, 0.56920767, 0.16765845, 0.036075152, 0.012611322,
      0.0040734885, 0.0021050037, 0.005289478, 0.005048708, 0.13315159,
      0.21391755, 0.078861, 0.009594708, 0.00281861, 0.0012908158, 0.0008019758,
      0.0018171906, 0.00233967, 0.04680576, 0.05701892, 0.016205171,
      0.0022423686, 0.00095875567, 0.00052623043, 0.0002733859, 0.0005455589,
      0.0006309181, 0.013931166, 0.01677046, 0.0045112027, 0.0005500894,
      0.00013497367, 0.000075709526, 0.00004587796, 0.00008546359,
      0.00007281406;

  utility::print_matrix("mat.csv", mat);
  mat2 = utility::read_matrix<Eigen::MatrixXd>("mat.csv");

  ASSERT_EQ(mat, mat2);
}

/**
 * Test SparseTrace and SparseTraceProduct function as expected with dense
 * matrix input
 *
 * Create two pairs of identical dense matrices with random elements, one pair
 * of dense type and the other sparse A: Use custom SparseProductTrace function
 * for sparse matrices B: Multiply Dense matrices together and call in built
 * trace() C: Multiply sparse matrices together, convert to dense matrix and
 * call in built trace() Assert A == B Assert A == C Assert custom SparseTrace
 * function is equivalent to (dense matrix).trace() Assert custom SparseTrace
 * function is equivalent to (sparse matrix).toDense().trace()
 */
TEST(utility_test, dense) {
  Eigen::MatrixXd mat1 = Eigen::MatrixXd::Zero(5, 5);
  Eigen::MatrixXd mat2 = Eigen::MatrixXd::Zero(5, 5);

  SparseMD sprs1(5, 5);
  SparseMD sprs2(5, 5);

  double rndm;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      rndm = std::rand() % 10;
      mat1(i, j) = rndm;
      sprs1.coeffRef(i, j) = rndm;

      rndm = std::rand() % 10;
      mat2(i, j) = rndm;
      sprs2.coeffRef(i, j) = rndm;
    }
  }

  ASSERT_EQ(utility::SparseProductTrace(sprs1, sprs2), (mat1 * mat2).trace());
  ASSERT_EQ(utility::SparseProductTrace(sprs1, sprs2),
            (sprs1 * sprs2).toDense().trace());
  ASSERT_EQ(utility::SparseTrace(sprs1), mat1.trace());
  ASSERT_EQ(utility::SparseTrace(sprs2), sprs2.toDense().trace());
}

/**
 * Test SparseTrace and SparseTraceProduct function as expected with sparse
 * matrix input
 *
 * Create two pairs of identical sparse matrices with random elements, one pair
 * of dense type and the other sparse A: Use custom SparseProductTrace function
 * for sparse matrices B: Multiply Dense matrices together and call in built
 * trace() C: Multiply sparse matrices together, convert to dense matrix and
 * call in built trace() Assert A == B Assert A == C Assert custom SparseTrace
 * function is equivalent to (dense matrix).trace() Assert custom SparseTrace
 * function is equivalent to (sparse matrix).toDense().trace()
 */
TEST(utility_test, sparse) {
  Eigen::MatrixXd mat1 = Eigen::MatrixXd::Zero(5, 5);
  Eigen::MatrixXd mat2 = Eigen::MatrixXd::Zero(5, 5);

  SparseMD sprs1(5, 5);
  SparseMD sprs2(5, 5);

  double rndm;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (std::rand() % 10 < 2) {
        rndm = std::rand() % 10;
        mat1(i, j) = rndm;
        sprs1.coeffRef(i, j) = rndm;
      }
      if (std::rand() % 10 < 2) {
        rndm = std::rand() % 10;
        mat2(i, j) = rndm;
        sprs2.coeffRef(i, j) = rndm;
      }
    }
  }

  ASSERT_EQ(utility::SparseProductTrace(sprs1, sprs2), (mat1 * mat2).trace());
  ASSERT_EQ(utility::SparseProductTrace(sprs1, sprs2),
            (sprs1 * sprs2).toDense().trace());
  ASSERT_EQ(utility::SparseTrace(sprs1), mat1.trace());
  ASSERT_EQ(utility::SparseTrace(sprs2), sprs2.toDense().trace());
}

TEST(utility_test, dimension_mismatch) {
  // test handling of incorrect input dimension

  SparseMD sprs1(4, 4);
  SparseMD sprs2(4, 5);
  SparseMD sprs3(5, 5);

  double rndm;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (std::rand() % 10 < 2) {
        rndm = std::rand() % 10;
        sprs1.coeffRef(i, j) = rndm;
        rndm = std::rand() % 10;
        sprs2.coeffRef(i, j) = rndm;
        rndm = std::rand() % 10;
        sprs3.coeffRef(i, j) = rndm;
      }
    }
  }
  ASSERT_ANY_THROW(utility::SparseTrace(sprs2));
  ASSERT_ANY_THROW(utility::SparseProductTrace(sprs1, sprs2));
  ASSERT_ANY_THROW(utility::SparseProductTrace(sprs1, sprs3));
}
