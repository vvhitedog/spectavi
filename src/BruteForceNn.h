#ifndef BRUTEFORCENN_H
#define BRUTEFORCENN_H

#include "EigenDefinitions.h"
#include <queue>
#include <vector>

/* brute-force computation of nearest neighbours under a p-norm */

namespace spectavi {

template <typename MatrixType = RowMatrixXf,
          typename MatrixTypeLabel = RowMatrixXs>
class BruteForceNn {
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::Map<MatrixType> MatrixTypeMap;

private:
  const Eigen::Ref<const MatrixType> &m_x;
  const Eigen::Ref<const MatrixType> &m_y;
  const double m_p;
  const double m_mu;

public:
  BruteForceNn(const Eigen::Ref<const MatrixType> &x,
               const Eigen::Ref<const MatrixType> &y, double p = .5,
               double mu = 0.)
      : m_x(x), m_y(y), m_p(p), m_mu(mu) {}

  void find_neighbours(Eigen::Ref<MatrixTypeLabel> out_idx,
                       Eigen::Ref<MatrixType> out_dist, int k = 2) const {

    int xcols = m_x.cols();
    std::vector<std::priority_queue<std::pair<double, int>>> top_ks(m_y.rows());
    for (int irow = 0; irow < m_y.rows(); ++irow) {
      // setup local access for better caching (hopefully)
      auto &yrow = m_y.row(irow);
      auto &top_k = top_ks[irow];
      // setup worst distance
      double worst_dist = -1;
      // compute the best match for every possible row in x
      for (int irowx = 0; irowx < m_x.rows(); ++irowx) {
        // compute distance under lp norm
        double distp = 0;
        for (int icol = 0; icol < xcols; ++icol) {
          double diff = m_x(irowx, icol) - yrow(icol);
          distp += std::pow(std::abs(diff), m_p);
          // prune based on worst-distance so far, and allow for a extrapolation
          // based on mu parameter
          if (worst_dist >= 0 &&
              distp + m_mu * (xcols - (icol + 1)) >= worst_dist) {
            break;
          }
        }
        // update the heap
        if (worst_dist >= 0 && distp < worst_dist) {
          // heap needs to be modified
          top_k.pop();
          top_k.push(std::make_pair(distp, irowx));
          worst_dist = top_k.top().first;
        } else if (worst_dist < 0) {
          // heap is empty
          top_k.push(std::make_pair(distp, irowx));
          worst_dist = distp;
        }
      }
      // update out
      auto &outi_row = out_idx.row(irow);
      auto &outd_row = out_dist.row(irow);
      int ik = k-1;
      while (top_k.size()) {
        auto &pair = top_k.top();
        outd_row(ik) = pair.first;
        outi_row(ik) = pair.second;
        --ik;
        top_k.pop();
      }
    }
  }
};

} // namespace spectavi

#endif // BRUTEFORCENN_H
