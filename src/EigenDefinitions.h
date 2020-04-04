#ifndef SPECTAVI_EIGENDEFINITIONS_H
#define SPECTAVI_EIGENDEFINITIONS_H

#include <Eigen/Core>
#include <Eigen/Dense>

namespace spectavi {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMatrixXd;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMatrixXi;
typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMatrixXs;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMatrixXf;
typedef Eigen::Map<RowMatrixXd> RowMatrixXdMap;
typedef Eigen::Map<RowMatrixXi> RowMatrixXiMap;
typedef Eigen::Map<RowMatrixXf> RowMatrixXfMap;
typedef Eigen::Map<RowMatrixXs> RowMatrixXsMap;

#include <NdArray.h>
template <typename MatrixType = RowMatrixXd>
void ndarray_copy_matrix(const MatrixType &mat, NdArray *arr) {
  typedef typename MatrixType::Scalar Scalar;
  ndarray_set_size(arr, mat.rows(), mat.cols());
  ndarray_alloc(arr);
  std::copy(mat.data(), mat.data() + mat.size(), (Scalar *)arr->m_data);
}

} // namespace spectavi

#endif // SPECTAVI_EIGENDEFINITIONS_H
