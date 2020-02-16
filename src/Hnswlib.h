#ifndef HNSWLIB_H
#define HNSWLIB_H

#include <hnswlib/hnswlib.h>

#include "EigenDefinitions.h"
#include <list>
#include <set>

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
  size_t m_batch_size;
  size_t m_nbatch;
  hnswlib::L2Space m_l2space;
  std::list<hnswlib::HierarchicalNSW<Scalar>> m_approx_algs;

public:
  Hnswlib(size_t vec_dim, size_t max_elements, size_t batch_size = 5000)
      : m_vec_dim(vec_dim), m_batch_size(batch_size), m_l2space(m_vec_dim) {
    m_nbatch = std::ceil(((double)max_elements) / batch_size);
    for (int i = 0; i < m_nbatch; ++i) {
      size_t amount = batch_size;
      if (i == m_nbatch - 1) {
        amount = max_elements % batch_size;
      }
      m_approx_algs.emplace_back(&m_l2space, amount, 128);
    }
  }

  /**
   * @brief Add points to the underlying Hnswlib construct.
   *
   * @param points Matrix of points to be added, matrix to have `npt` rows and
   * `ndim` columns.
   */
  void add_points(const Eigen::Ref<const MatrixType> &points) {
    assert(points.cols() == m_vec_dim);
    auto cur_alg = m_approx_algs.begin();
    for (int i = 0; i < points.rows(); ++i) {
      if (i % m_batch_size == 0 && i > 0) {
        cur_alg++;
      }
      const Scalar *row = points.row(i).data();
      cur_alg->addPoint(row, i);
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
  void find_approx_neighbours(const Eigen::Ref<const MatrixType> &query,
                              Eigen::Ref<MatrixTypeLabel> out,
                              int k = 2) const {
    assert(query.cols() == m_vec_dim);
    typedef std::priority_queue<std::pair<Scalar, hnswlib::labeltype>>
        return_type;
    typedef std::set<std::pair<Scalar, hnswlib::labeltype>> result_collection;
    result_collection collect;
    for (int i = 0; i < query.rows(); ++i) {
      collect.clear();
      const Scalar *row = query.row(i).data();
      for (auto &alg : m_approx_algs) {
        return_type result = alg.searchKnn(row, k);
        while (result.size()) {
          auto &pair = result.top();
          collect.insert(pair);
          result.pop();
        }
      }
      auto it = collect.begin();
      for (int ik = 0; ik < k; ++ik, it++) {
        auto &pair = *it;
        out(i, ik) = pair.second;
      }
    }
  }
};

} // namespace spectavi

#endif // HNSWLIB_H
