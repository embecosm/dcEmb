#include <iostream>
#include "mv_to_eigen.hh"

PyObject* vectorxd_to_mv(const Eigen::VectorXd& mat) {
  PyObject* mv =
      PyMemoryView_FromMemory((char*)mat.data(), mat.size(), PyBUF_WRITE);
  Py_buffer* buf_sel = PyMemoryView_GET_BUFFER(mv);
  buf_sel->format = "d";
  buf_sel->itemsize = sizeof(double);
  // std::cout << "itemsize" << &buf->itemsize << '\n';
  buf_sel->strides = &buf_sel->itemsize;
  return mv;
}
PyObject* matrixxd_to_mv(const Eigen::MatrixXd& mat) {
  PyObject* mv =
      PyMemoryView_FromMemory((char*)mat.data(), mat.size(), PyBUF_WRITE);
  Py_buffer* buf_sel = PyMemoryView_GET_BUFFER(mv);
  buf_sel->format = "d";
  buf_sel->itemsize = sizeof(double);
  buf_sel->ndim = 2;
  Py_ssize_t tmp_shape[2];
  tmp_shape[0] = mat.rows();
  tmp_shape[1] = mat.cols();
  buf_sel->shape = tmp_shape;
  // std::cout << "itemsize" << &buf->itemsize << '\n';
  Py_ssize_t tmp_strides[2];
  tmp_strides[0] = sizeof(double);
  tmp_strides[1] = sizeof(double) * mat.rows();
  buf_sel->strides = tmp_strides;
  return mv;
}
PyObject* vectorxi_to_mv(const Eigen::VectorXi& mat) {
  PyObject* mv =
      PyMemoryView_FromMemory((char*)mat.data(), mat.size(), PyBUF_WRITE);
  Py_buffer* buf_sel = PyMemoryView_GET_BUFFER(mv);
  buf_sel->format = "i";
  buf_sel->itemsize = sizeof(int);
  // std::cout << "itemsize" << &buf->itemsize << '\n';
  buf_sel->strides = &buf_sel->itemsize;
  return mv;
}