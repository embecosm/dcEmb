#include "mv_to_eigen.hh"

Eigen::VectorXd mv_to_vectorxd(double *mv, const int& len) {
  return Eigen::Map<Eigen::VectorXd>(mv, len);
}
Eigen::MatrixXd mv_to_matrixxd(double *mv, const int& len1, const int& len2) {
  return Eigen::Map<Eigen::MatrixXd>(mv, len1, len2);
}
Eigen::VectorXi mv_to_vectorxi(int *mv, const int& len) {
  return Eigen::Map<Eigen::VectorXi>(mv, len);
}