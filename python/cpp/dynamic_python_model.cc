/**
 * The 3-body dynamic causal model class within the dcEmb package
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

#include "dynamic_python_model.hh"
#include "utility.hh"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#include <numpy/arrayobject.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <vector>
#define FSGEN

#define PL parameter_locations
#define ZERO(a, b) Eigen::MatrixXd::Zero(a, b)

/**
 * Observed outcomes for the python problem.
 */
Eigen::VectorXd dynamic_python_model::get_observed_outcomes() {
  Eigen::Map<Eigen::VectorXd> observed_outcomes(
      this->response_vars.data(),
      this->response_vars.rows() * this->response_vars.cols());
  return observed_outcomes;
}

/**
 * Return the wrapped forward model for the python problem
 */
std::function<Eigen::VectorXd(const Eigen::VectorXd&)>
dynamic_python_model::get_forward_model_function() {
  std::function<Eigen::VectorXd(Eigen::VectorXd)> forward_model = std::bind(
      &dynamic_python_model::forward_model,
      this, std::placeholders::_1, this->num_samples,
      this->select_response_vars, this->external_generative_model);
  return forward_model;
}

/**
 * Returns the forward model for the python problem
 */

void dynamic_python_model::forward_model_fun(
    const Eigen::VectorXd& parameters, const int& timeseries_length,
    const Eigen::VectorXi& select_response_vars,
    PyObject* external_generative_model, double* output_ptr) {
  Eigen::VectorXd out =
      forward_model(parameters, timeseries_length, select_response_vars,
                    external_generative_model);
  std::memcpy(output_ptr, out.data(), out.size() * sizeof(double));
}

Eigen::VectorXd dynamic_python_model::forward_model(
    const Eigen::VectorXd& parameters, const int& timeseries_length,
    const Eigen::VectorXi& select_response_vars,
    PyObject* external_generative_model) {
  PyObject* mv_param = PyMemoryView_FromMemory((char*)parameters.data(),
                                               parameters.size(), PyBUF_WRITE);
  Py_buffer* buf_param = PyMemoryView_GET_BUFFER(mv_param);
  buf_param->format = "d";
  buf_param->itemsize = sizeof(double);
  // std::cout << "itemsize" << &buf->itemsize << '\n';
  buf_param->strides = &buf_param->itemsize;

  PyObject* mv_sel =
      PyMemoryView_FromMemory((char*)select_response_vars.data(),
                              select_response_vars.size(), PyBUF_WRITE);
  Py_buffer* buf_sel = PyMemoryView_GET_BUFFER(mv_sel);
  buf_sel->format = "i";
  buf_sel->itemsize = sizeof(int);
  // std::cout << "itemsize" << &buf->itemsize << '\n';
  buf_sel->strides = &buf_sel->itemsize;

  PyObject* ts_len = PyLong_FromLong(timeseries_length);
  PyObject* py_args = PyTuple_Pack(3, mv_param, ts_len, mv_sel);

  // std::cout << "external" << external_generative_model << '\n';

  PyObject* out_obj = PyObject_CallObject(external_generative_model, py_args);
  PyObject* out_mv = PyMemoryView_FromObject(out_obj);
  Py_buffer* out_buf = PyMemoryView_GET_BUFFER(out_mv);

  out_buf->strides = &out_buf->itemsize;
  Eigen::Map<Eigen::VectorXd> out(
      (double*)out_buf->buf, select_response_vars.size() * timeseries_length);

  return out;
}

// void dynamic_python_model::forward_model_fun(
//     const Eigen::VectorXd& parameters, const int& timeseries_length,
//     const Eigen::VectorXi& select_response_vars, double* output_ptr,
//     PyObject* func) {
//   // npy_intp dims[1] = {parameters.size()};
//   // PyArrayObject* paramArray = (PyArrayObject*)PyArray_SimpleNewFromData(
//   //     1, dims, NPY_FLOAT64, (double*)parameters.data());

//   // PyObject *py_args = PyTuple_New(0);

//   // Eigen::VectorXd out(parameters.size());
//   // out << parameters;
//   // out(select_response_vars) =
//   //     Eigen::VectorXd::Ones(select_response_vars.size()) * 20;
//   std::cout << "forward_model2" << '\n';

//   // double* outdat = out.data();
//   // std::copy(outdat, outdat + 12, vec);
//   // vec = outdat;

//   // Py_buffer* view;
//   // view->obj = NULL;
//   // view->buf = out.data();
//   // view->len = timeseries_length * sizeof(double);
//   // view->readonly = 0;
//   // view->itemsize = sizeof(double);
//   // view->format = "d";  // integer
//   // view->ndim = 1;
//   // Py_ssize_t sha[1];
//   // sha[0] = (Py_ssize_t)out.cols();
//   // view->shape = sha;                // length-1 sequence of dimensions
//   // view->strides = &view->itemsize;  // for the simple case we can do this
//   // view->suboffsets = NULL;
//   // view->internal = NULL;

//   // pybuffer.format = (char*)"d";
//   // pybuffer.itemsize = 8;
//   // pybuffer.len = timeseries_length * sizeof(double);
//   // Py_ssize_t sha[1];
//   // sha[0] = (Py_ssize_t) PyLong_FromLong(timeseries_length);
//   // pybuffer.shape = sha;
//   // pybuffer.ndim = 1;
//   // Py_ssize_t stri[1];
//   // stri[0] = (Py_ssize_t) PyLong_FromLong(8);
//   // pybuffer.strides = stri;
//   std::cout << "here0.0" << '\n';
//   PyObject* mv_param = PyMemoryView_FromMemory((char*)parameters.data(),
//                                                parameters.size(),
//                                                PyBUF_WRITE);
//   Py_buffer* buf_param = PyMemoryView_GET_BUFFER(mv_param);
//   buf_param->format = "d";
//   buf_param->itemsize = sizeof(double);
//   // std::cout << "itemsize" << &buf->itemsize << '\n';
//   buf_param->strides = &buf_param->itemsize;

//   PyObject* mv_sel =
//       PyMemoryView_FromMemory((char*)select_response_vars.data(),
//                               select_response_vars.size(), PyBUF_WRITE);
//   Py_buffer* buf_sel = PyMemoryView_GET_BUFFER(mv_sel);
//   buf_sel->format = "i";
//   buf_sel->itemsize = sizeof(int);
//   // std::cout << "itemsize" << &buf->itemsize << '\n';
//   buf_sel->strides = &buf_sel->itemsize;

//   std::cout << "here0" << '\n';
//   PyObject* ts_len = PyLong_FromLong(timeseries_length);
//   PyObject* py_args = PyTuple_Pack(3, mv_param, ts_len, mv_sel);
//   std::cout << "here1" << '\n';

//   std::cout << "parameters.data()" << parameters.data() << '\n';
//   std::cout << "parameters" << parameters << '\n';
//   PyObject* out_obj = PyObject_CallObject(func, py_args);
//   std::cout << "here2" << '\n';
//   PyObject* out_mv = PyMemoryView_FromObject(out_obj);
//   std::cout << "Object Type " << Py_TYPE(out_obj)->tp_name << '\n';
//   Py_buffer* out_buf = PyMemoryView_GET_BUFFER(out_mv);

//   // std::cout << "Checking buffer.. " << PyObject_CheckBuffer(out_obj) <<
//   '\n';
//   // std::cout << "Getting buffer.. " << PyObject_GetBuffer(out_obj, out_mv,
//   // PyBUF_CONTIG) << '\n';
//   std::cout << "out_mv->buf" << out_buf->buf << '\n';
//   std::cout << "out_mv->len" << out_buf->len << '\n';
//   std::cout << "out_mv->format" << out_buf->format << '\n';
//   std::cout << "out_mv->itemsize" << (int*)out_buf->itemsize << '\n';
//   std::cout << "out_mv->shape" << (int*)out_buf->shape << '\n';
//   std::cout << "out_mv->strides" << (int*)out_buf->strides << '\n';
//   out_buf->strides = &out_buf->itemsize;
//   std::memcpy(output_ptr, out_buf->buf, out_buf->len);
//   Eigen::Map<Eigen::VectorXd> out(
//       (double*)output_ptr, select_response_vars.size() * timeseries_length);

//   std::cout << "out" << out << '\n';

//   return;
// }

/**
 * Dynamic Causal Model constructor for the python problem
 */
dynamic_python_model::dynamic_python_model() { return; }
