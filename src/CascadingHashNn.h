#ifndef CASCADINGHASHNN_H
#define CASCADINGHASHNN_H

#include "BruteForceNnL1K2.h"
#include "EigenDefinitions.h"

#include <iostream>
#include <list>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* compute nearest neighbours using a cascade of hash tables:
  http://openaccess.thecvf.com/content_cvpr_2014/papers/Cheng_Fast_and_Accurate_2014_CVPR_paper.pdf
*/

namespace spectavi {

namespace filter {
template <typename HashingNn> class SetFilter {
private:
  std::unordered_set<int> m_idx;
  std::unordered_set<int>::iterator m_pos;
  const HashingNn &m_nn;

public:
  SetFilter(const HashingNn &nn) : m_pos(m_idx.begin()), m_nn(nn) {}

  void add(int x) {
    m_idx.insert(x);
    m_pos = m_idx.begin();
  }

  void init(int iyr) {
    m_idx.clear();
    m_nn.filter_potential_neighbours(iyr, m_idx);
    m_pos = m_idx.begin();
  }

  int operator()() {
    if (m_pos == m_idx.end()) {
      return -1;
    }
    int ret = *m_pos;
    m_pos++;
    return ret;
  }
};
} // namespace filter

template <typename MatrixType = RowMatrixXf,
          typename MatrixTypeLabel = RowMatrixXs>
class CascadingHashNn {
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixTypeLabel::Scalar Label;
  typedef Eigen::Map<const MatrixType> MatrixTypeMap;
  typedef int32_t hashcode;
  typedef Eigen::Matrix<hashcode, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      MatrixTypeHash;
  typedef Eigen::Matrix<hashcode, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::ColMajor>
      MatrixTypeHashCm;

private:
  MatrixTypeMap m_x;
  MatrixTypeMap m_y;
  const int m_hash_bit_rate;
  const int m_num_hash_table;
  const int m_num_candidate_neighbours;

  std::vector<MatrixType>
      m_hash_dicts; // the hyper-planes used to generate bit-codes

  std::vector<MatrixType>
      m_intermediate_form_y; // intermediate calculation for y

  MatrixTypeHash m_hashcodes_x; // converted hashcodes for x
  MatrixTypeHash m_hashcodes_y; // converted hashcodes for y

  std::vector<std::unordered_map<hashcode, std::list<Label>>>
      m_hash_tables; // only hash tables for x are necessary

  void generate_hash_dict() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::normal_distribution<Scalar> normal(0., 1.);
    int dim = m_x.cols();
    m_hash_dicts.resize(m_num_hash_table);
    for (auto &hash_dict : m_hash_dicts) {
      hash_dict.resize(dim, m_hash_bit_rate);
      for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < m_hash_bit_rate; ++j) {
          hash_dict(i, j) = normal(g);
        }
      }
    }
  }

  template <typename Derived>
  hashcode convert_to_hashcode(const Eigen::MatrixBase<Derived> &row) const {
    hashcode ret = 0;
    for (int i = 0; i < row.size(); ++i) {
      if (row(i) >= 0) {
        ret |= (1 << i);
      }
    }
    return ret;
  }

  template <typename Derived>
  void generate_hashcodes_for_dict(const Eigen::Ref<const MatrixType> &in,
                                   const Eigen::Ref<const MatrixType> &dict,
                                   const Eigen::MatrixBase<Derived> &codes) {
    Eigen::MatrixBase<Derived> &_codes =
        const_cast<Eigen::MatrixBase<Derived> &>(codes);
    MatrixType R = in * dict;
    for (int i = 0; i < in.rows(); ++i) {
      _codes(i) = convert_to_hashcode(R.row(i));
    }
  }

  template <typename Derived>
  void
  generate_hashcodes_for_data(const Eigen::Ref<const MatrixType> &in,
                              const Eigen::MatrixBase<Derived> &fullcodes) {
    Eigen::MatrixBase<Derived> &_fullcodes =
        const_cast<Eigen::MatrixBase<Derived> &>(fullcodes);
    _fullcodes.derived().resize(in.rows(), m_num_hash_table);
    for (int i = 0; i < m_num_hash_table; ++i) {
      generate_hashcodes_for_dict(in, m_hash_dicts[i], _fullcodes.col(i));
    }
  }

  void generate_intermediate_form_for_y() {
    m_intermediate_form_y.resize(m_num_hash_table);
    for (int i = 0; i < m_num_hash_table; ++i) {
      m_intermediate_form_y[i] = m_y * m_hash_dicts[i];
    }
  }

  void generate_hashcodes() {
    generate_hashcodes_for_data(m_x, m_hashcodes_x);
    //    generate_hashcodes_for_data(m_y, m_hashcodes_y);
    generate_intermediate_form_for_y();
  }

  void generate_y_candidate_hashcodes(int i, int j, std::list<hashcode> &dst,
                                      int cutoff = 2) const {
    MatrixType row = m_intermediate_form_y[j].row(i); // need a copy
    std::priority_queue<std::pair<Scalar, int>> candidates;
    hashcode ret = 0;
    for (int i = 0; i < row.size(); ++i) {
      candidates.push(std::make_pair(std::abs(row(i)), i));
      if (candidates.size() > cutoff) {
        candidates.pop();
      }
      if (row(i) >= 0) {
        ret |= (1 << i);
      }
    }
    hashcode start_ret = ret;
    std::list<int> top_idx;
    while (candidates.size()) {
      top_idx.push_back(candidates.top().second);
      candidates.pop();
    }
    int limit = 1 << cutoff;
    for (int ii = 0; ii < limit; ++ii) {
      int i_idx = 0;
      for (auto &idx : top_idx) {
        ret ^= (ret & (1 << idx)); // zero out the bit
        int x = (ii & (1 << i_idx++)) ? 1 : 0;
        ret |= (x << idx); // set it to permutation
      }
      dst.push_back(ret);
    }

    if (std::find(dst.begin(), dst.end(), start_ret) == dst.end()) {
      throw std::runtime_error(
          "hashing calc: sanity check finds calc is incorrect.");
    }
  }

  void initialize_hash_tables() {
    generate_hash_dict();
    generate_hashcodes();
    m_hash_tables.resize(m_num_hash_table);
    for (int j = 0; j < m_num_hash_table; ++j) {
      for (int i = 0; i < m_x.rows(); ++i) {
        m_hash_tables[j][m_hashcodes_x(i, j)].push_back(i);
      }
    }
  }

public:
  CascadingHashNn(const Scalar *x, const Scalar *y, int xrows, int yrows,
                  int dim, int hash_bit_rate = 12, int num_hash_table = 4,
                  int num_candidate_neighbours = 2)
      : m_x(x, xrows, dim), m_y(y, yrows, dim), m_hash_bit_rate(hash_bit_rate),
        m_num_hash_table(num_hash_table),
        m_num_candidate_neighbours(num_candidate_neighbours) {
    initialize_hash_tables();
  }

  void filter_potential_neighbours(Label iyr,
                                   std::unordered_set<int> &neighbours) const {
    for (int iht = 0; iht < m_num_hash_table; ++iht) {
      hashcode ycode = m_hashcodes_y(iyr, iht);
      std::list<hashcode> candidate_hashcodes;
      generate_y_candidate_hashcodes(iyr, iht, candidate_hashcodes,
                                     m_num_candidate_neighbours);
      const auto &ht = m_hash_tables[iht];
      for (auto &_ycode : candidate_hashcodes) {
        auto it = ht.find(_ycode);
        if (it == ht.end()) { // didn't find any elements in this bucket
          continue;
        }
        const auto &list = it->second;
        for (const auto &idx : list) {
          neighbours.insert(idx);
        }
      }
    }
  }

  void find_neighbours(Eigen::Ref<MatrixTypeLabel> out_idx,
                       Eigen::Ref<MatrixType> out_dist, int nthread = 8) const {
    typedef filter::SetFilter<CascadingHashNn> FilterType;
    FilterType filter(*this);
    typedef BruteForceNnL1K2<MatrixTypeLabel> BruteForceNn;
    typename BruteForceNn::MatrixType bf_x, bf_y;
    typename BruteForceNn::MatrixTypeBig bf_dist(m_y.rows(), 2);
    bf_x = (m_x).template cast<typename BruteForceNn::Scalar>();
    bf_y = (m_y).template cast<typename BruteForceNn::Scalar>();
    bf_x.array() += 128;
    bf_y.array() += 128;
    BruteForceNn bfnn(bf_x.data(), bf_y.data(), bf_x.rows(), bf_y.rows(),
                      bf_x.cols());
    bfnn.template find_neighbours<FilterType>(out_idx, bf_dist, filter,
                                              nthread);
    out_dist = bf_dist.template cast<Scalar>();
  }
};

} // namespace spectavi

#endif // CASCADINGHASHNN_H
