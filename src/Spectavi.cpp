#include "Spectavi.h"

extern "C" {

//////////////////////////////////////////////////////////////////////////////////////////////
//
//            Common Ctypes Exposed Functionality
//
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

using namespace spectavi;

void seven_point_algorithm(const double *x, const double *xp, int *nroot,
                           double *dst) {

  const int stride = 2;
  const int dststride = 3 * 3;

  FundamentalMatrixFitter<RowMatrixXd> fmatfitter;
  for (int i = 0; i < 7; ++i) {
    const double *xrow = x + stride * i;
    const double *xprow = xp + stride * i;
    fmatfitter.add_putative_match(xrow[0], xrow[1], xprow[0], xprow[1]);
  }

  RowMatrixXd Fs[3];
  int nroots = fmatfitter.solve(Fs[0], Fs[1], Fs[2]);

  *nroot = nroots;

  for (int iroot = 0; iroot < nroots; ++iroot) {
    std::copy(Fs[iroot].data(), Fs[iroot].data() + dststride,
              (dst + dststride * iroot));
  }
}

void dlt_triangulate(const double *P0, const double *P1, int npt,
                     const double *x, const double *xp, double *dst) {

  RowMatrixXdMap _x(const_cast<double *>(x), npt, 3);
  RowMatrixXdMap _xp(const_cast<double *>(xp), npt, 3);

  RowMatrixXdMap _dst(dst, npt, 4);

  DltTriangulator<RowMatrixXd> dlt(P0, P1);
  for (int i = 0; i < npt; ++i) {
    _dst.row(i) =
        dlt.solve(_x.row(i).data(), _xp.row(i).data(), 3).X().transpose();
  }
}

void dlt_reprojection_error(const double *P0, const double *P1, int npt,
                            const double *x, const double *xp, double *dst) {

  RowMatrixXdMap _x(const_cast<double *>(x), npt, 3);
  RowMatrixXdMap _xp(const_cast<double *>(xp), npt, 3);

  RowMatrixXdMap _dst(dst, npt, 1);

  DltTriangulator<RowMatrixXd> dlt(P0, P1);
  for (int i = 0; i < npt; ++i) {
    _dst(i) =
        dlt.solve(_x.row(i).data(), _xp.row(i).data(), 3).reprojection_error();
  }
}

void ransac_fitter(const double *x0, const double *x1, int npt,
                   double required_percent_inliers,
                   double reprojection_error_allowed, int maximum_tries,
                   bool find_best_even_in_failure,
                   double singular_value_ratio_allowed, bool progressbar,
                   bool *success, NdArray *essential, NdArray *camera,
                   double *inlier_percent, NdArray *inlier_idx) {

  RansacFitter<> fitter(x0, x1, npt, required_percent_inliers,
                        reprojection_error_allowed, maximum_tries,
                        find_best_even_in_failure);
  fitter.fit_essential(singular_value_ratio_allowed, progressbar);
  *success = fitter.success();
  ndarray_copy_matrix(fitter.essential(), essential);
  ndarray_copy_matrix(fitter.camera(), camera);
  *inlier_percent = fitter.inlier_percent();
  ndarray_copy_matrix(fitter.inlier_idx(), inlier_idx);
}

void image_pair_rectification(const double *P0, const double *P1,
                              const double *im0, const double *im1, int wid,
                              int hgt, int nchan, double sampling_factor,
                              NdArray *rectified0, NdArray *rectified1,
                              NdArray *rectified_idx0,
                              NdArray *rectified_idx1) {
  RowMatrixXdMap _P0(const_cast<double *>(P0), 3, 4);
  RowMatrixXdMap _P1(const_cast<double *>(P1), 3, 4);
  RowMatrixXdMap _im0(const_cast<double *>(im0), hgt, wid);
  RowMatrixXdMap _im1(const_cast<double *>(im1), hgt, wid);
  Rectifier<> rectifier(_P0, _P1, _im0, _im1, nchan);
  rectifier.resample(sampling_factor);
  if (nchan == 1) {
    ndarray_copy_matrix(rectifier.rectified0(), rectified0);
    ndarray_copy_matrix(rectifier.rectified1(), rectified1);
  } else if (nchan > 1) {
    typedef typename RowMatrixXd::Scalar Scalar;
    {
      ndarray_set_size(rectified0, rectifier.rows(), rectifier.cols(), nchan);
      ndarray_alloc(rectified0);
      auto mat = rectifier.rectified0();
      std::copy(mat.data(), mat.data() + mat.size(),
                (Scalar *)rectified0->m_data);
    }
    {
      ndarray_set_size(rectified1, rectifier.rows(), rectifier.cols(), nchan);
      ndarray_alloc(rectified1);
      auto mat = rectifier.rectified1();
      std::copy(mat.data(), mat.data() + mat.size(),
                (Scalar *)rectified1->m_data);
    }
  }
  ndarray_copy_matrix(rectifier.rectified_idx0(), rectified_idx0);
  ndarray_copy_matrix(rectifier.rectified_idx1(), rectified_idx1);
}

/**
 * @brief SIFT keypoint detection and description generation
 *
 * @param im Image buffer to apply on, assumed to be grayscale (1-channel),
 * scaling (seems) uninmportant
 * @param wid Width of the image
 * @param hgt Height of the image
 * @param out Buffer that output will be written to (float NdArray)
 */
void sift_filter(const double *im, int wid, int hgt, NdArray *out) {
  RowMatrixXdMap _im(const_cast<double *>(im), hgt, wid);
  // copy image data into a buffer for sift
  RowMatrixXf fim = _im.cast<float>();
  SiftFilter filt(fim);
  filt.filter();
  size_t nkp = filt.get_nkeypoints();
  ndarray_set_size(out, nkp, SIFT_KP_SIZE);
  ndarray_alloc(out);
  filt.get_data((float *)out->m_data);
}

/**
 * @brief Approximate nearest neighbour (ANN) using HNSWlib (L2-distance
 * metric)
 *
 * @param x Matrix of `dim` dimensional points on each row to be matched
 * against
 * @param y Matrix of `dim` dimensional points on each row to be matched for
 * @param xrows  Number of rows in `x`
 * @param yrows  Number of rows in `y`
 * @param dim  Dimensionality of the points
 * @param k Number of top matches to return (with smallest L2 distance)
 * @param out Buffer that output will be written to (size_t Ndarray)
 */
void ann_hnswlib(const float *x, const float *y, int xrows, int yrows, int dim,
                 int k, NdArray *out) {
  RowMatrixXfMap _x(const_cast<float *>(x), xrows, dim);
  RowMatrixXfMap _y(const_cast<float *>(y), yrows, dim);
  ndarray_set_size(out, yrows, k);
  ndarray_alloc(out);
  RowMatrixXsMap _out(reinterpret_cast<size_t *>(out->m_data), yrows, k);
  Hnswlib<> ann(dim, xrows);
  ann.add_points(_x);
  ann.find_approx_neighbours(_y, _out, k);
}

/**
 * @brief nn_bruteforce Nearest neighbour using pseudo-bruteforce technique
 *
 * @param x Matrix of `dim` dimensional points on each row to be matched
 * against
 * @param y Matrix of `dim` dimensional points on each row to be matched for
 * @param xrows  Number of rows in `x`
 * @param yrows  Number of rows in `y`
 * @param dim  Dimensionality of the points
 * @param k Number of top matches to return (with smallest L2 distance)
 * @param p p-value of the norm to compute (`p` > 0 expected)
 * @param mu Approximation parameter, when `mu`=0 it's exact
 * @param outidx Buffer that holds idx output (size_t Ndarray)
 * @param outdist Buffer that holds distance output (float Ndarray)
 */
void nn_bruteforce(const float *x, const float *y, int xrows, int yrows,
                   int dim, int k, float p, float mu, NdArray *outidx,
                   NdArray *outdist) {
  RowMatrixXfMap _x(const_cast<float *>(x), xrows, dim);
  RowMatrixXfMap _y(const_cast<float *>(y), yrows, dim);
  ndarray_set_size(outidx, yrows, k);
  ndarray_alloc(outidx);
  ndarray_set_size(outdist, yrows, k);
  ndarray_alloc(outdist);
  RowMatrixXsMap _outidx(reinterpret_cast<size_t *>(outidx->m_data), yrows, k);
  RowMatrixXfMap _outdist(reinterpret_cast<float *>(outdist->m_data), yrows, k);
  BruteForceNn<> nn(_x, _y, p, mu);
  nn.find_neighbours(_outidx, _outdist, k);
}

void nn_bruteforcei(const int *x, const int *y, int xrows, int yrows, int dim,
                    int k, float p, float mu, NdArray *outidx,
                    NdArray *outdist) {
  RowMatrixXiMap _x(const_cast<int *>(x), xrows, dim);
  RowMatrixXiMap _y(const_cast<int *>(y), yrows, dim);
  ndarray_set_size(outidx, yrows, k);
  ndarray_alloc(outidx);
  ndarray_set_size(outdist, yrows, k);
  ndarray_alloc(outdist);
  RowMatrixXsMap _outidx(reinterpret_cast<size_t *>(outidx->m_data), yrows, k);
  RowMatrixXiMap _outdist(reinterpret_cast<int *>(outdist->m_data), yrows, k);
  BruteForceNn<RowMatrixXi> nn(_x, _y, p, mu);
  nn.find_neighbours(_outidx, _outdist, k);
}

void kmedians(const float *x, int xrows, int dim, int k) {
  RowMatrixXfMap _x(const_cast<float *>(x), xrows, dim);
  KMedians<> kmed(_x,k);
  kmed.run();
}
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
}
