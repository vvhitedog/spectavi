#ifndef HNSWLIB_H
#define HNSWLIB_H

#include <hnswlib/hnswlib.h>

#include "EigenDefinitions.h"

/* wraps the implemention of the HNSWLIB Approximate Nearest Neighbour (ann) */

namespace spectavi {

template <typename MatrixType = RowMatrixXf,
          typename MatrixTypeLabel = RowMatrixXs>
class Hnswlib {
  /*mg: I think hnswlib is hard-coded to use floats */
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::Map<MatrixType> MatrixTypeMap;

 private:
  size_t m_vec_dim;
  hnswlib::L2Space m_l2space;
  hnswlib::HierarchicalNSW<Scalar> m_approx_alg;

 public:
  Hnswlib(size_t vec_dim, size_t max_elements)
      : m_vec_dim(vec_dim),
        m_l2space(m_vec_dim),
        m_approx_alg(&m_l2space, max_elements, 128) {}


  /**
   * @brief Add points to the underlying Hnswlib construct.
   *
   * @param points Matrix of points to be added, matrix to have `npt` rows and
   * `ndim` columns.
   */
  void add_points(const Eigen::Ref<const MatrixType>& points) {
    assert(points.cols() == m_vec_dim);
    for (int i = 0; i < points.rows(); ++i) {
      const Scalar* row = points.row(i).data();
      m_approx_alg.addPoint(row, i);
    }
  }

  /**
   * @brief Find the nearest neighbour to a set a points in `vec_dim`
   * dimensions.
   *
   * @param query The matrix with each row being a query point.
   * @param out A matrix (pre-allocated) with `k` entries per row, sorted
   * descending distance.
   * @param k The number of nearest neighbour points that are returned.
   */
  void find_approx_neighbours(const Eigen::Ref<const MatrixType>& query,
                              Eigen::Ref<MatrixTypeLabel> out,
                              int k = 2) const {
    assert(query.cols() == m_vec_dim);
    typedef std::priority_queue<std::pair<Scalar, hnswlib::labeltype>>
        return_type;
    for (int i = 0; i < query.rows(); ++i) {
      const Scalar* row = query.row(i).data();
      return_type result = m_approx_alg.searchKnn(row, k);
      int ik = 0;
      while (result.size()) {
        auto& pair = result.top();
        out(i, ik++) = pair.second;
        result.pop();
      }
    }
  }
};

}  // namespace spectavi

#endif  // HNSWLIB_H
