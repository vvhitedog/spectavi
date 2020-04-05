#ifndef SPECTAVI_RANSACFITTER_H
#define SPECTAVI_RANSACFITTER_H

#include "Camera.h"
#include "DltTriangulator.h"
#include "EigenDefinitions.h"
#include "FundamentalMatrixFitter.h"
#include <random>

#include <iostream>
#include <unordered_set>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

namespace spectavi {

#define PROGRESS_BAR_LENGTH 100

template <typename MatrixType = RowMatrixXd, typename MatrixTypeI = RowMatrixXi>
class RansacFitter {

  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::Map<const MatrixType> MatrixTypeMapConst;

private:
  typedef Camera<MatrixType> Camera_t;
  MatrixTypeMapConst m_x0;
  MatrixTypeMapConst m_x1;
  double m_required_percent_inliers;
  double m_reprojection_error_allowed;
  int m_maximum_tries;

  MatrixType m_best_fit_essential_matrix;
  Camera_t m_best_fit_camera;
  double m_best_fit_inlier_percent;
  bool m_success;
  bool m_find_best_even_in_failure;
  MatrixTypeI m_inlier_idx;

  bool process_fundamental_matrix(const Eigen::Ref<const MatrixType> &F,
                                  double singular_value_ratio_allowed,
                                  int &inlier_count, Camera_t &best_P,
                                  MatrixTypeI &inlier_idx) const {
    bool success = false;
    Eigen::JacobiSVD<MatrixType> svd(F,
                                     Eigen::ComputeFullV | Eigen::ComputeFullU);
    auto sv = svd.singularValues();
    double ratio = std::abs(sv(0) - sv(1)) / (std::abs(sv(0) + sv(1)) / 2.);
    if (ratio > singular_value_ratio_allowed) {
      return success;
    }
    MatrixType ee(3, 1);
    ee << 1., 1., 0;
    MatrixType E = svd.matrixU() * ee.asDiagonal() * svd.matrixV().transpose();
    Camera_t P0;
    double best_percent_inlier = 0;
    for (Camera_t P1 :
         Essential2Cameras<std::vector<Camera_t>, MatrixType>(E)) {
      DltTriangulator<MatrixType> dlt(P0.get_matrix().data(),
                                      P1.get_matrix().data());
      int nex = m_x0.rows();
      int ninlier = 0;
      for (int ix = 0; ix < nex; ++ix) {
        dlt.solve(m_x0.row(ix).data(), m_x1.row(ix).data(), 3);
        bool is_inlier =
            dlt.reprojection_error() <= m_reprojection_error_allowed &&
            dlt.is_infront_both_cameras();
        if (is_inlier) {
          ninlier++;
        }
      }
      double percent_inlier = ninlier / (double)(nex);
      if ((percent_inlier >= m_required_percent_inliers ||
           m_find_best_even_in_failure) &&
          (percent_inlier > best_percent_inlier)) {
        best_percent_inlier = percent_inlier;
        inlier_count = ninlier;
        best_P = P1;
        success = true;
        // When successful, find the indices of inliers
        inlier_idx.resize(ninlier, 1);
        int iidx = 0;
        // XXX: Duplicated code, see above
        for (int ix = 0; ix < nex; ++ix) {
          dlt.solve(m_x0.row(ix).data(), m_x1.row(ix).data(), 3);
          bool is_inlier =
              dlt.reprojection_error() <= m_reprojection_error_allowed &&
              dlt.is_infront_both_cameras();
          if (is_inlier) {
            inlier_idx(iidx++) = ix;
          }
        }
      }
    }
    return success;
  }

  template <typename OutputContainer>
  OutputContainer draw_random_indices(int range, int k) const {
    OutputContainer ret(k);
    std::vector<int> arange(range, 0);
    for (int i = 0; i < range; ++i) {
      arange[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(arange.begin(), arange.end(), g);
    for (int i = 0; i < k; ++i) {
      ret[i] = arange[i];
    }
    return ret;
  }

  // taken from https://stackoverflow.com/a/58198037/177931
  std::unordered_set<int> floyd_sample(int N, int k) const {
    //   std::default_random_engine gen;
    std::random_device rd;
    std::mt19937 gen(rd());
    // for the benchmark I used a faster hash table
    std::unordered_set<int> elems; // preallocation is good
    for (int r = N - k; r < N; ++r) {
      int v = std::uniform_int_distribution<>(1, r)(gen);
      if (!elems.insert(v).second)
        elems.insert(r);
    }
    return elems;
  }

public:
  RansacFitter(const Scalar *x0, const Scalar *x1, int npt,
               double required_percent_inliers,
               double reprojection_error_allowed, int maximum_tries,
               bool find_best_even_in_failure)
      : m_x0(x0, npt, 3), m_x1(x1, npt, 3),
        m_required_percent_inliers(required_percent_inliers),
        m_reprojection_error_allowed(reprojection_error_allowed),
        m_maximum_tries(maximum_tries), m_success(false),
        m_best_fit_inlier_percent(0.),
        m_find_best_even_in_failure(find_best_even_in_failure) {

    if (m_x0.rows() != m_x1.rows()) {
      throw std::invalid_argument(
          "Supplied incorrect point matches, numbers do not match.");
    }

    if (m_x0.rows() < 10) {
      throw std::invalid_argument(
          "Supplied less than 10 point matches, unsupported.");
    }
  }

  void fit_essential(double singular_value_ratio_allowed = 3e-2,
                     bool progressbar = false,
                     int nthread = 8 ) {
    int nex = m_x0.rows();
    if (!m_success) {
      m_best_fit_inlier_percent = 0;
    }
#ifdef ENABLE_OPENMP
    omp_lock_t success_lock;
    omp_init_lock(&success_lock);
#endif
    int done = 0;
#pragma omp parallel for num_threads(nthread) shared(m_success)
    for (int itry = 0; itry < m_maximum_tries; ++itry) {
      if (progressbar) {
#pragma omp critical
        {
          done += 1;
          std::cout << "\r |";
          double percent_done = (double(done)) / m_maximum_tries;
          for (int i = 0; i < PROGRESS_BAR_LENGTH; ++i) {
            if (i <= percent_done * PROGRESS_BAR_LENGTH) {
              std::cout << "-";
            } else {
              std::cout << " ";
            }
          }
          std::cout << "|" << std::flush;
        }
      }
      if (m_success) {
        continue;
      }
      FundamentalMatrixFitter<MatrixType> fmat_fitter;
      for (int ix : floyd_sample(nex, 7)) {
        MatrixType row0 = m_x0.row(ix).hnormalized().eval();
        MatrixType row1 = m_x1.row(ix).hnormalized().eval();
        double x = row0(0);
        double y = row0(1);
        double xp = row1(0);
        double yp = row1(1);
        fmat_fitter.add_putative_match(x, y, xp, yp);
      }
      MatrixType F0, F1, F2;
      MatrixTypeI inlier_idx;
      Camera_t cam;
      int nroot = fmat_fitter.solve(F0, F1, F2);
      int ninlier = 0;
      if (nroot >= 1 &&
          process_fundamental_matrix(F0, singular_value_ratio_allowed, ninlier,
                                     cam, inlier_idx)) {
        double percent_inlier = (double)ninlier / (double)nex;
        if ((percent_inlier > m_required_percent_inliers ||
             m_find_best_even_in_failure) &&
            m_best_fit_inlier_percent < percent_inlier) {
#ifdef ENABLE_OPENMP
          omp_set_lock(&success_lock);
#endif
          if (!m_success) {
            m_success = percent_inlier > m_required_percent_inliers;
            m_best_fit_inlier_percent = percent_inlier;
            m_best_fit_essential_matrix = F0;
            m_best_fit_camera = cam;
            m_inlier_idx = inlier_idx;
          }
#ifdef ENABLE_OPENMP
          omp_unset_lock(&success_lock);
#endif
        }
      }
      if (nroot >= 2 &&
          process_fundamental_matrix(F1, singular_value_ratio_allowed, ninlier,
                                     cam, inlier_idx)) {
        double percent_inlier = (double)ninlier / (double)nex;
        if ((percent_inlier > m_required_percent_inliers ||
             m_find_best_even_in_failure) &&
            m_best_fit_inlier_percent < percent_inlier) {
#ifdef ENABLE_OPENMP
          omp_set_lock(&success_lock);
#endif
          if (!m_success) {
            m_success = percent_inlier > m_required_percent_inliers;
            m_best_fit_inlier_percent = percent_inlier;
            m_best_fit_essential_matrix = F1;
            m_best_fit_camera = cam;
            m_inlier_idx = inlier_idx;
          }
#ifdef ENABLE_OPENMP
          omp_unset_lock(&success_lock);
#endif
        }
      }
      if (nroot >= 3 &&
          process_fundamental_matrix(F2, singular_value_ratio_allowed, ninlier,
                                     cam, inlier_idx)) {
        double percent_inlier = (double)ninlier / (double)nex;
        if ((percent_inlier > m_required_percent_inliers ||
             m_find_best_even_in_failure) &&
            m_best_fit_inlier_percent < percent_inlier) {
#ifdef ENABLE_OPENMP
          omp_set_lock(&success_lock);
#endif
          if (!m_success) {
            m_success = percent_inlier > m_required_percent_inliers;
            m_best_fit_inlier_percent = percent_inlier;
            m_best_fit_essential_matrix = F2;
            m_best_fit_camera = cam;
            m_inlier_idx = inlier_idx;
          }
#ifdef ENABLE_OPENMP
          omp_unset_lock(&success_lock);
#endif
        }
      }
    }
    if (progressbar) {
      std::cout << std::endl;
    }
#ifdef ENABLE_OPENMP
    omp_destroy_lock(&success_lock);
#endif
  }

  bool success() const { return m_success; }

  MatrixType essential() const { return m_best_fit_essential_matrix; }

  MatrixType camera() const { return m_best_fit_camera.get_matrix(); }

  MatrixTypeI inlier_idx() const { return m_inlier_idx; }

  double inlier_percent() const { return m_best_fit_inlier_percent; }
};

#undef PROGRESS_BAR_LENGTH

} // namespace spectavi

#endif // SPECTAVI_RANSACFITTER_H
