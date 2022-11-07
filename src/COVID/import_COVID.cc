/**
 * A collection of functions for importing and processing dynamic causal models
 * within the dcEmb package
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
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <vector>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "country_data.hh"

/**
 * Extract data from raw text files, and put into a vector of country_data
 * structs
 */
std::vector<country_data> read_country_data(int num_countries) {
  std::vector<country_data> countries;
  std::ifstream cases_file;
  std::ifstream deaths_file;
  cases_file.open("../src/data/time_series_covid19_confirmed_global_new.csv");
  deaths_file.open("../src/data/time_series_covid19_deaths_global_new.csv");

  std::string cases_line;
  std::string deaths_line;
  std::string last_country = "NONE";
  std::getline(cases_file, cases_line);
  std::getline(deaths_file, deaths_line);
  std::vector<std::string> date_list;
  splitstr(date_list, cases_line, ',');
  while (std::getline(cases_file, cases_line)) {
    std::getline(deaths_file, deaths_line);

    std::vector<std::string> cases_split_tmp;
    std::vector<std::string> deaths_split_tmp;
    country_data country_tmp;
    splitstr(cases_split_tmp, cases_line, ',');
    splitstr(deaths_split_tmp, deaths_line, ',');
    int first_case = 4;
    for (int i = 4; i < cases_split_tmp.size(); i++) {
      if (cases_split_tmp[i].compare("0")) {
        first_case = i;
        break;
      }
    }
    // String vector to integer vector
    std::vector<int> cases_tmp(cases_split_tmp.size() - first_case);
    std::transform(cases_split_tmp.begin() + first_case, cases_split_tmp.end(),
                   cases_tmp.begin(),
                   [](const std::string &val) { return std::stoi(val); });
    // std integer Vector to Eigen integer vector
    Eigen::VectorXi eigen_cases = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(
        cases_tmp.data(), cases_tmp.size());
    // String vector to integer vector
    std::vector<int> deaths_tmp(deaths_split_tmp.size() - first_case);
    std::transform(deaths_split_tmp.begin() + first_case,
                   deaths_split_tmp.end(), deaths_tmp.begin(),
                   [](const std::string &val) { return std::stoi(val); });
    // std integer Vector to Eigen integer vector
    Eigen::VectorXi eigen_deaths =
        Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(deaths_tmp.data(),
                                                      deaths_tmp.size());
    if (last_country.compare(cases_split_tmp[1])) {
      country_tmp.name = cases_split_tmp[1];
      country_tmp.latitude = stod(cases_split_tmp[2]);
      country_tmp.longitude = stod(cases_split_tmp[3]);

      country_tmp.pop = get_country_pop(country_tmp.name);
      country_tmp.first_case = first_case;
      country_tmp.date = date_list[first_case];

      country_tmp.cases = smooth(eigen_cases, 7);
      country_tmp.deaths = smooth(eigen_deaths, 7);
      country_tmp.days = country_tmp.cases.size();
      country_tmp.cum_cases = (country_tmp.cases).sum();
      country_tmp.cum_deaths = (country_tmp.deaths).sum();
    } else {
      country_tmp = countries.back();
      countries.pop_back();

      country_tmp.first_case = std::min(first_case, country_tmp.first_case);
      country_tmp.date = date_list[country_tmp.first_case];

      country_tmp.cases =
          zero_padded_add(country_tmp.cases, smooth(eigen_cases, 7));
      country_tmp.deaths =
          zero_padded_add(country_tmp.deaths, smooth(eigen_deaths, 7));
      country_tmp.days =
          std::max(country_tmp.days, (int)country_tmp.cases.size());
      country_tmp.cum_cases = (country_tmp.cases).sum();
      country_tmp.cum_deaths = (country_tmp.deaths).sum();
    }
    countries.push_back(country_tmp);
    last_country = cases_split_tmp[1];
  }
  sort(countries.begin(), countries.end(),
       [](const country_data &a, const country_data &b) {
         return a.cum_deaths > b.cum_deaths;
       });
  std::vector<country_data> ret_val(countries.begin(),
                                    countries.begin() + num_countries);
  return ret_val;
}

/**
 * Extract country population from csv file
 */
int get_country_pop(const std::string &name) {
  std::ifstream pop_file;
  pop_file.open("../src/data/population.csv");
  std::string pop_line;
  while (std::getline(pop_file, pop_line)) {
    std::vector<std::string> pop_split_tmp;
    splitstr(pop_split_tmp, pop_line, ',');
    if (pop_split_tmp.at(0) == name) {
      return stoi(pop_split_tmp.at(1)) * 1000;
    }
  }
  return -1;
}

/**
 * Smooth the data through applying a "Graph Laplacian" method. Essentially,
 * repeatedly adjust the value of each data point by it's immediate neighbours.
 * Implemented by creating a matrix implementing this averaging, and repeatedly
 * applying it.
 */
Eigen::VectorXd smooth(const Eigen::VectorXi &data, int period) {
  int sz = data.size() + period + 1;
  Eigen::VectorXi shifted_vector = Eigen::VectorXi::Zero(sz);
  shifted_vector << Eigen::VectorXi::Zero(period + 1), data;

  Eigen::VectorXd result_vector = Eigen::VectorXd::Zero(sz);
  for (Eigen::Index i = 1; i < sz - 1; i++) {
    result_vector(i) =
        0.5 * ((double)shifted_vector(i + 1) - (double)shifted_vector(i - 1));
  }
  result_vector(0) = shifted_vector(1) - shifted_vector(0);
  result_vector(sz - 1) = shifted_vector(sz - 1) - shifted_vector(sz - 2);

  result_vector =
      result_vector.unaryExpr([](double x) { return std::max(x, 0.0); });

  Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(sz, sz);
  laplacian.diagonal() = Eigen::VectorXd::Constant(sz, -2);
  laplacian.diagonal(+1) = Eigen::VectorXd::Constant(sz - 1, 1);
  laplacian.diagonal(-1) = Eigen::VectorXd::Constant(sz - 1, 1);
  laplacian(0, 0) = -1;
  laplacian(sz - 1, sz - 1) = -1;

  Eigen::MatrixXd tmp_diag = Eigen::MatrixXd::Zero(sz, sz);
  tmp_diag.diagonal() = Eigen::VectorXd::Constant(sz, 1);

  laplacian = tmp_diag + laplacian / 4;

  laplacian = mat_power(laplacian, 4 * period);
  Eigen::MatrixXd print_res = result_vector;
  return (laplacian * result_vector);
}

/**
 * Simple implementation of matrix exponentiation using an iterative
 * "exponentiation by squaring" method. Not suitable for negative or fractional
 * exponents.
 */
Eigen::MatrixXd mat_power(const Eigen::MatrixXd &mat, int p) {
  int iter_p = p;
  Eigen::SparseMatrix<double> iter_mat(mat.rows(), mat.cols());
  iter_mat.setZero();
  iter_mat = mat.sparseView();
  Eigen::SparseMatrix<double> iter_odd(mat.rows(), mat.cols());
  iter_odd.setIdentity();
  while (iter_p > 1) {
    if (iter_p % 2) {
      iter_odd = iter_mat * iter_odd;
      iter_mat = iter_mat * iter_mat;
      iter_p = (iter_p - 1) / 2;
    } else {
      iter_mat = iter_mat * iter_mat;
      iter_p = iter_p / 2;
    }
  }
  return Eigen::MatrixXd(iter_mat * iter_odd);
}

/**
 * Addition of two vectors, padding the shorter of the two vectors with leading
 * zeros
 */
Eigen::VectorXd zero_padded_add(const Eigen::VectorXd &mat1,
                                const Eigen::VectorXd &mat2) {
  if (mat1.size() == mat2.size()) return mat1 + mat2;
  if (mat1.size() < mat2.size()) {
    Eigen::VectorXd tmp_vector = Eigen::VectorXd::Zero(mat2.size());
    tmp_vector << Eigen::VectorXd::Zero(mat2.size() - mat1.size()), mat1;
    return tmp_vector + mat2;
  } else {
    Eigen::VectorXd tmp_vector = Eigen::VectorXd::Zero(mat1.size());
    tmp_vector << Eigen::VectorXd::Zero(mat1.size() - mat2.size()), mat2;
    return tmp_vector + mat1;
  }
}

/**
 * Split a string around a token
 */
void splitstr(std::vector<std::string> &vec, std::string &str,
              const char &delim) {
  std::stringstream ss(str);
  std::string line;
  while (std::getline(ss, line, delim)) {
    vec.push_back(line);
  }
  return;
}