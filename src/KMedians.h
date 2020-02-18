#ifndef KMEDIANS_H
#define KMEDIANS_H

#include "EigenDefinitions.h"
#include <list>
#include <queue>
#include <random>
#include <set>
#include <vector>

namespace spectavi {

template <typename MatrixType = RowMatrixXf,
          typename MatrixTypeLabel = RowMatrixXs>
class KMedians {
  typedef typename MatrixTypeLabel::Scalar Label;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::Map<MatrixType> MatrixTypeMap;

private:
  MatrixTypeMap m_x;
  const int m_k;

  std::vector<std::vector<std::vector<Scalar>>> m_median_cols;
  std::vector<std::list<Label>> m_median_idx;

  MatrixType m_medians;
  std::vector<Label> m_sigma;

  std::vector<Label> m_point_to_median;

  void next_permutation() {
    // randomize m_sigma
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(m_sigma.begin(), m_sigma.end(), g);
  }

  Scalar compute_median(std::vector<Scalar> &vector) const {
    if (vector.size() % 2 == 0) {
      const auto median_it1 = vector.begin() + vector.size() / 2 - 1;
      const auto median_it2 = vector.begin() + vector.size() / 2;

      std::nth_element(vector.begin(), median_it1, vector.end());
      const auto e1 = *median_it1;

      std::nth_element(vector.begin(), median_it2, vector.end());
      const auto e2 = *median_it2;

      return (e1 + e2) / 2.;

    } else {
      const auto median_it = vector.begin() + vector.size() / 2;
      std::nth_element(vector.begin(), median_it, vector.end());
      return *median_it;
    }
  }

  void print_stats() {
    int dim = m_x.cols();
    printf(" stats of each cluster:\n");
    double mean_median_dist = 0.;
    for (int ik = 0; ik < m_k; ++ik) {
      printf(">cluster id: %d\n", ik);
      std::list<Label> &idx = m_median_idx[ik];
      printf(">>#points: %d\n", idx.size());
      Scalar mean_dist = 0;
      Scalar max_dist = -1;
      Scalar min_dist = -1;
      for (auto &ip : idx) {
        Scalar dist = 0;
        for (int id = 0; id < dim; ++id) {
          dist += std::abs(m_x(ip, id) - m_medians(ik, id));
        }
        mean_dist += dist;
        max_dist = max_dist < 0 ? dist : std::max(max_dist, dist);
        min_dist = min_dist < 0 ? dist : std::min(min_dist, dist);
      }
      mean_dist /= idx.size();
      mean_median_dist += mean_dist;
      printf(">>avg dist: %f\n", mean_dist);
      printf(">>min dist: %f\n", min_dist);
      printf(">>max dist: %f\n", max_dist);
    }
    mean_median_dist /= m_k;
    printf("mean median dist: %f\n", mean_median_dist);
  }

  bool update_medians() {
    int change = 0;
    int dim = m_x.cols();
    for (int ik = 0; ik < m_k; ++ik) {
      // recompute medians based on data
      auto &colv = m_median_cols[ik];
      for (int id = 0; id < dim; ++id) {
        m_medians(ik, id) = compute_median(colv[id]);
      }
      // assign each point a median id
      std::list<Label> &idx = m_median_idx[ik];
      for (auto &ip : idx) {
        Label old_ik = m_point_to_median[ip];
        m_point_to_median[ip] = ik;
        change += old_ik != ik ? 1 : 0;
      }
    }
    return change > 0;
  }

  void assign_medians() {
    // get dimensions
    int n = m_x.rows();
    int dim = m_x.cols();
    // clear all datastructures
    for (int ik = 0; ik < m_k; ++ik) {
      auto &colv = m_median_cols[ik];
      for (int id = 0; id < dim; ++id) {
        colv[id].clear();
      }
    }
    for (auto &list : m_median_idx) {
      list.clear();
    }
    // compute full distance matrix between medians and points
    typedef std::pair<Scalar, std::pair<int, int>> heap_entry_t;
    //    std::priority_queue<heap_entry_t, std::vector<heap_entry_t>,
    //                        std::greater<heap_entry_t>>
    std::vector<heap_entry_t> active;
    std::list<heap_entry_t> deferred;
    // full matrix is stored in active
    for (int ik = 0; ik < m_k; ++ik) {
      for (int ip = 0; ip < n; ++ip) {
        std::pair<int, int> coord(ip, ik);
        Scalar dist = 0;
        for (int id = 0; id < dim; ++id) {
          dist += std::abs(m_x(ip, id) - m_medians(ik, id));
        }
        active.push_back(std::make_pair(-dist, coord));
      }
    }
    size_t heap_size = active.size();
    std::make_heap(active.begin(), active.end());

    std::vector<bool> point_used(n, false);
    int point_used_count = 0;

    while (point_used_count < n) {

      int allowed_count = 1;
      int median_used_count = 0;
      //      std::vector<bool> median_used(m_k, false);
      std::vector<int> median_used(m_k, 0);

      //      while (active.size() && median_used_count < allowed_count * m_k) {
      while (heap_size && median_used_count < allowed_count * m_k) {
        //        auto &kv = active.top();
        auto &kv = *active.begin();
        int ip = kv.second.first;
        if (point_used[ip]) {
          // skip used points as quickly as possible
          //          active.pop();
          std::pop_heap(active.begin(), active.begin() + heap_size--);
          continue;
        }
        int ik = kv.second.second;
        if (median_used[ik] >= allowed_count) {
          // assure medians grow equally
          deferred.push_back(kv);
          //          active.pop();
          std::pop_heap(active.begin(), active.begin() + heap_size--);
          continue;
        }
        // mark median as used
        median_used[ik]++;
        median_used_count++;
        // mark point as used
        point_used[ip] = true;
        point_used_count++;
        // update data-structures keeping track of idx's and values
        Label _idx = ip;
        std::list<Label> &idx = m_median_idx[ik];
        auto &colv = m_median_cols[ik];
        for (int id = 0; id < dim; ++id) {
          auto &col = colv[id];
          col.push_back(m_x(_idx, id));
        }
        idx.push_back(_idx);
        // remove item permanently
        //        active.pop();
        std::pop_heap(active.begin(), active.begin() + heap_size--);
      }

      // add the deferred entries back to active heap
      for (auto &x : deferred) {
        //        active.push(x);
        active[heap_size] = x;
        std::push_heap(active.begin(), active.begin() + heap_size++);
      }
      deferred.clear();
    }
  }

  /**
   * @brief initialize_medians yields initial guess for medians.
   */
  void initialize_medians() {
    int n = m_x.rows();
    int dim = m_x.cols();
    for (int ik = 0; ik < m_k; ++ik) {
      // ensure each dimension has its own list
      m_median_cols[ik].resize(dim);
    }
    next_permutation();
    int c = 0;
    while (c < n) {
      for (int ik = 0; ik < m_k; ++ik) {
        std::list<Label> &idx = m_median_idx[ik];
        auto &colv = m_median_cols[ik];
        Label _idx = m_sigma[c++];
        for (int id = 0; id < dim; ++id) {
          auto &col = colv[id];
          col.push_back(m_x(_idx, id));
        }
        idx.push_back(_idx);
        if (c >= n) {
          break;
        }
      }
    }
    update_medians();
  }

public:
  KMedians(const Eigen::Ref<const MatrixType> &x, int k)
      : m_x(const_cast<Scalar *>(x.data()), x.rows(), x.cols()), m_k(k),
        m_median_cols(m_k), m_median_idx(m_k), m_medians(m_k, x.cols()),
        m_point_to_median(x.rows(), -1) {
    m_sigma.resize(m_x.rows());
    int c = 0;
    for (auto &x : m_sigma) {
      x = c++;
    }
  }

  void run(int niter = 1, bool verbose = false) {
    initialize_medians();
    int iter = 0;
    do {
      if (verbose) {
        print_stats();
      }
      assign_medians();
    } while (update_medians() && ++iter < niter);
    if (verbose) {
      print_stats();
    }
  }
};

} // namespace spectavi

#endif // KMEDIANS_H
