#ifndef SPECTAVI_EIGENDEFINITIONS_H
#define SPECTAVI_EIGENDEFINITIONS_H

#include <Eigen/Core>
#include <Eigen/Dense>

using EigenDStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType> using EigenDRef = Eigen::Ref<MatrixType, 0, EigenDStride>;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXdMap = Eigen::Map<RowMatrixXd>;
using RowMatrixXiMap = Eigen::Map<RowMatrixXi>;

#include <NdArray.h>
template<typename MatrixType = RowMatrixXd>
void ndarray_copy_matrix(const MatrixType &mat, NdArray *arr) {
	typedef typename MatrixType::Scalar Scalar;
	ndarray_set_size(arr,mat.rows(),mat.cols());
	ndarray_alloc(arr);
	std::copy(mat.data(),mat.data()+mat.size(),(Scalar*)arr->m_data);
}

#endif //SPECTAVI_EIGENDEFINITIONS_H
