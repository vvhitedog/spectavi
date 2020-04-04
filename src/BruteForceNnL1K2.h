#ifndef BRUTEFORCENNL1K2_H
#define BRUTEFORCENNL1K2_H

#include "EigenDefinitions.h"

#include <cstdint>
#include <emmintrin.h>

/*
 * specialized brute-force computation of L1 norm with K=2 neighbours, designed
 * to work with SSE instructions *
 */

namespace spectavi {

namespace implementation {
uint16_t sad_16(const uint8_t a[16], const uint8_t b[16]) {
  __m128i _a = _mm_load_si128(reinterpret_cast<const __m128i *>(a));
  __m128i _b = _mm_load_si128(reinterpret_cast<const __m128i *>(b));
  __m128i sad = _mm_sad_epu8(_a, _b);
  return sad[0] + sad[1];
}
} // namespace implementation

using RowMatrixXu8 =
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXu8Map = Eigen::Map<RowMatrixXu8>;

template <typename MatrixTypeLabel = RowMatrixXs> class BruteForceNnL1K2 {

public:
  typedef RowMatrixXu8 MatrixType;
  typedef RowMatrixXi MatrixTypeBig;
  typedef MatrixType::Scalar Scalar;
  typedef MatrixTypeBig::Scalar ScalarBig;
  typedef typename MatrixTypeLabel::Scalar Label;
  typedef Eigen::Map<const MatrixType> MatrixTypeMap;

private:
  MatrixTypeMap m_x;
  MatrixTypeMap m_y;
  const int m_K;

public:
  BruteForceNnL1K2(const Scalar *x, const Scalar *y, int xrows, int yrows,
                   int dim)
      : m_x(x, xrows, dim), m_y(y, yrows, dim), m_K(2) {
    if (m_x.cols() != m_y.cols()) {
      throw std::runtime_error("Matrix inner dimensions must match.");
    }
    if (m_x.cols() % 16 != 0) {
      // dimensions of x,y must be aligned with 16
      throw std::runtime_error(
          "Input matrix inner dimensions must be 16-byte aligned.");
    }
  }

  void find_neighbours(Eigen::Ref<MatrixTypeLabel> out_idx,
                       Eigen::Ref<MatrixTypeBig> out_dist,
                       int nthread = 8) const {
    const int dim = m_x.cols();
    const int n128i = (dim / 16); // number of 128-byte datatypes per row
    // scan through every row
#pragma omp parallel for num_threads(nthread)
    for (int irow = 0; irow < m_y.rows(); ++irow) {
      // get local references to outputs
      auto &first_i = out_idx(irow, 0);
      auto &second_i = out_idx(irow, 1);
      auto &first_d = out_dist(irow, 0);
      auto &second_d = out_dist(irow, 1);
      // set distances to infinity
      first_d = std::numeric_limits<ScalarBig>::max();
      second_d = std::numeric_limits<ScalarBig>::max();
      first_i = -1;
      second_i = -1;
      // start main computation
      ScalarBig worst_dist = -1;
      const Scalar *_y = m_y.row(irow).data();
      for (int irowx = 0; irowx < m_x.rows(); ++irowx) {
        const Scalar *_x = m_x.row(irowx).data();
        ScalarBig distp = 0;
        bool prune = false;
        // loop over 128-byte groups and use SSE instructions to calculate SAD
        for (int i128i = 0; i128i < n128i; ++i128i) {
          const Scalar *__x = _x + 16 * i128i;
          const Scalar *__y = _y + 16 * i128i;
          distp += implementation::sad_16(__x, __y);
          if (worst_dist >= 0 && distp > worst_dist) {
            prune = true;
            break;
          }
        }
        if (prune) {
          // early exit detected
          continue;
        }
        // std::cout << "no-prune, distp: " << distp
        //           << " worst_dist: " << worst_dist << std::endl;
        if (distp < first_d) {
          // move things down one
          std::swap(first_d, second_d);
          std::swap(first_i, second_i);
          first_d = distp;
          first_i = irowx;
        } else if (distp < second_d) {
          // set the second value only
          second_d = distp;
          second_i = irowx;
        }
        if (second_i != -1) {
          worst_dist = second_d;
        }
      }
    }
  }
};

} // namespace spectavi

#endif // BRUTEFORCENNL1K2
