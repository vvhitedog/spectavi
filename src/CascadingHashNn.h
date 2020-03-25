#ifndef CASCADINGHASHNN_H
#define CASCADINGHASHNN_H

#include "BruteForceNnL1K2.h"
#include "EigenDefinitions.h"

#include <iostream>
#include <list>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* compute nearest neighbours using a cascade of hash tables:
  http://openaccess.thecvf.com/content_cvpr_2014/papers/Cheng_Fast_and_Accurate_2014_CVPR_paper.pdf
*/

namespace spectavi {

namespace filter {
template<typename HashingNn>
class SetFilter {
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
    m_nn.filter_potential_neighbours(iyr,m_idx);
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
  const Scalar m_num_hash_table;

  std::vector<MatrixType>
      m_hash_dicts; // the hyper-planes used to generate bit-codes

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

  void generate_hashcodes() {
    generate_hashcodes_for_data(m_x, m_hashcodes_x);
    generate_hashcodes_for_data(m_y, m_hashcodes_y);
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

  // void print_stats() const {
  //  for (int j = 0; j < m_num_hash_table; ++j) {
  //    std::vector<int> sizedump;
  //    FILE *stream = nullptr;
  //    std::string fn = "/tmp/dump" + std::to_string(j);
  //    stream = fopen(fn.c_str(), "w");
  //    for (auto &ht : m_hash_tables[j]) {
  //      sizedump.push_back(ht.second.size());
  //    }
  //    fwrite(sizedump.data(), sizeof(int), sizedump.size(), stream);
  //    fclose(stream);
  //  }
  //}

public:
  CascadingHashNn(const Scalar *x, const Scalar *y, int xrows, int yrows,
                  int dim, int hash_bit_rate = 12, int num_hash_table = 8)
      : m_x(x, xrows, dim), m_y(y, yrows, dim), m_hash_bit_rate(hash_bit_rate),
        m_num_hash_table(num_hash_table) {
    initialize_hash_tables();
    // print_stats();
  }

  void
  filter_potential_neighbours(Label iyr,
                              std::unordered_set<int> &neighbours) const {
    for (int iht = 0; iht < m_num_hash_table; ++iht) {
      hashcode ycode = m_hashcodes_y(iyr, iht);
      const auto &ht = m_hash_tables[iht];
      auto it = ht.find(ycode);
      if (it == ht.end()) { // didn't find any elements in this bucket
        continue;
      }
      const auto &list = it->second;
      for (const auto &idx : list) {
        neighbours.insert(idx);
      }
    }
  }

  void find_neighbours(Eigen::Ref<MatrixTypeLabel> out_idx,
                       Eigen::Ref<MatrixType> out_dist, int nthread = 8) const {
    typedef filter::SetFilter<CascadingHashNn> FilterType;
    FilterType filter(*this);
//    for (int iyr = 0; iyr < m_y.rows(); ++iyr) {
//      filter::SetFilter &filt = filts[iyr];
//      for (int iht = 0; iht < m_num_hash_table; ++iht) {
//        hashcode ycode = m_hashcodes_y(iyr, iht);
//        const auto &ht = m_hash_tables[iht];
//        auto it = ht.find(ycode);
//        if (it == ht.end()) { // didn't find any elements in this bucket
//          continue;
//        }
//        const auto &list = it->second;
//        for (const auto &idx : list) {
//          filt.add(idx);
//        }
//      }
//    }
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
