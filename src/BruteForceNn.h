#ifndef BRUTEFORCENN_H
#define BRUTEFORCENN_H

#include "EigenDefinitions.h"

/* brute-force computation of nearest neighbours under a p-norm */

namespace spectavi {

template <typename MatrixType = RowMatrixXf,
          typename MatrixTypeLabel = RowMatrixXs>
class BruteForceNn {
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::Map<MatrixType> MatrixTypeMap;
  const Eigen::Ref<const MatrixType> &m_x;
  const Eigen::Ref<const MatrixType> &m_y;
  double m_p;

public:
  BruteForceNn(const Eigen::Ref<const MatrixType> &x,
               const Eigen::Ref<const MatrixType> &y, double p = .5)
      : m_x(x), m_y(y), m_p(p) {}

  void find_neighbours(Eigen::Ref<MatrixTypeLabel> out, int k = 2) const {
    MatrixType diff(1, m_x.cols());
    for (int irow = 0; irow < m_y.rows(); ++irow) {
      diff = m_x - m_y;
    }
  }
};

} // namespace spectavi

#endif // BRUTEFORCENN_H
