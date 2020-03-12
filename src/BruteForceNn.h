#ifndef BRUTEFORCENN_H
#define BRUTEFORCENN_H

#include "EigenDefinitions.h"
#include <queue>
#include <vector>
#include <unordered_set>

/* brute-force computation of nearest neighbours under a p-norm */

namespace spectavi {

template <typename MatrixType = RowMatrixXf,
          typename MatrixTypeLabel = RowMatrixXs>
class BruteForceNn {
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixTypeLabel::Scalar Label;
  typedef Eigen::Map<MatrixType> MatrixTypeMap;

private:
  MatrixTypeMap m_x;
  MatrixTypeMap m_y;
  const double m_p;
  const Scalar m_mu;

public:
  BruteForceNn(const Eigen::Ref<const MatrixType> &x,
               const Eigen::Ref<const MatrixType> &y, double p = .5,
               Scalar mu = 0.)
      : m_x(const_cast<Scalar *>(x.data()), x.rows(), x.cols()),
        m_y(const_cast<Scalar *>(y.data()), y.rows(), y.cols()), m_p(p),
        m_mu(mu) {}

  void find_neighbours(Eigen::Ref<MatrixTypeLabel> out_idx,
                       Eigen::Ref<MatrixType> out_dist, int k = 2) const {
    std::unordered_set<Label> null;
    find_neighbours(out_idx, out_dist, null, null, k);
  }

  void find_neighbours(Eigen::Ref<MatrixTypeLabel> out_idx,
                       Eigen::Ref<MatrixType> out_dist,
                       std::unordered_set<Label> &filter_x,
                       std::unordered_set<Label> &filter_y, int k = 2) const {

    int dim = m_x.cols();
    int yrows = m_y.rows();
    int xrows = m_x.rows();
    int ylimit = filter_y.empty() ? yrows : filter_y.size();
    int xlimit = filter_x.empty() ? xrows : filter_x.size();

    auto yit = filter_y.begin();

    for (int ir = 0; ir < ylimit; ++ir) {
      auto xit = filter_x.begin();
      int irow = filter_y.empty() ? ir : *(yit++);
      // use a heap to track top neighbours
      std::priority_queue<std::pair<Scalar, int>,
                          std::vector<std::pair<Scalar, int>>>
          top_k;
      // setup local access for better caching (hopefully)
      auto &yrow = m_y.row(irow);
      // setup worst distance
      Scalar worst_dist = -1;
      // compute the best match for every possible row in x
      for (int irx = 0; irx < xlimit; ++irx) {
        int irowx = filter_x.empty() ? irx : *(xit++);
        // compute distance under lp norm
        Scalar distp = 0;
        bool prune = false;
        for (int icol = 0; icol < dim; ++icol) {
          float diff = m_x(irowx, icol) - yrow(icol);
          Scalar pval;
          if (m_p == 1) {
            pval = std::abs(diff);
          } else if (m_p == 2) {
            pval = diff * diff;
          } else if (m_p == .5) {
            pval = std::sqrt(std::abs(diff));
          } else {
            pval = std::pow(std::abs(diff), m_p);
          }
          distp += pval;
          // prune based on worst-distance so far, and allow for a extrapolation
          // based on mu parameter
          if (worst_dist >= 0 &&
              (distp + m_mu * (dim - (icol + 1)) > worst_dist)) {
            prune = true;
            break;
          }
        }
        if (prune) {
          // early exit detected
          continue;
        }
        // update the heap
        if (worst_dist >= 0 && distp < worst_dist) {
          // heap needs to be modified
          if (top_k.size() >= k) {
            // if heap is of size k, remove worst element
            top_k.pop();
          }
          top_k.push(std::make_pair(distp, irowx));
          // worst_dist might change, rely on heap to find it
          worst_dist = top_k.top().first;
        } else {
          // heap is empty
          top_k.push(std::make_pair(distp, irowx));
          if (top_k.size() == k) {
            worst_dist = top_k.top().first;
          }
        }
      }
      // update out
      int ik = k - 1;
      while (top_k.size()) {
        auto &pair = top_k.top();
        out_dist(irow, ik) = pair.first;
        out_idx(irow, ik) = pair.second;
        --ik;
        top_k.pop();
      }
    }
  }
};

} // namespace spectavi

#endif // BRUTEFORCENN_H
