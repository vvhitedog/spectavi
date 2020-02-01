#include "Spectavi.h"

extern "C" {

//////////////////////////////////////////////////////////////////////////////////////////////
//
//            Common Ctypes Exposed Functionality
//
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

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
    
    RowMatrixXdMap _x(const_cast<double*>(x), npt, 3);
    RowMatrixXdMap _xp(const_cast<double*>(xp), npt, 3);
    
    RowMatrixXdMap _dst(dst, npt, 4);
    
    DltTriangulator<RowMatrixXd> dlt(P0, P1);
    for (int i = 0; i < npt; ++i) {
        _dst.row(i) = dlt.solve(_x.row(i).data(), _xp.row(i).data(), 3).X().transpose();
    }
    
}

void dlt_reprojection_error(const double *P0, const double *P1, int npt,
                            const double *x, const double *xp, double *dst) {
    
    RowMatrixXdMap _x(const_cast<double*>(x), npt, 3);
    RowMatrixXdMap _xp(const_cast<double*>(xp), npt, 3);
    
    RowMatrixXdMap _dst(dst, npt, 1);
    
    DltTriangulator<RowMatrixXd> dlt(P0, P1);
    for (int i = 0; i < npt; ++i) {
        _dst(i) =
                dlt.solve(_x.row(i).data(), _xp.row(i).data(), 3).reprojection_error();
    }
    
}

void ransac_fitter(const double *x0, const double *x1, int npt,
                   double required_percent_inliers, double reprojection_error_allowed,
                   int maximum_tries, bool find_best_even_in_failure,
                   double singular_value_ratio_allowed, bool progressbar, bool *success,
                   NdArray *essential, NdArray* camera, double *inlier_percent,
                   NdArray *inlier_idx) {
    
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
                              const double *im0, const double *im1, int wid, int hgt, int nchan,
                              double sampling_factor, NdArray *rectified0, NdArray *rectified1,
                              NdArray *rectified_idx0, NdArray *rectified_idx1 ) {
    RowMatrixXdMap _P0(const_cast<double*>(P0), 3, 4);
    RowMatrixXdMap _P1(const_cast<double*>(P1), 3, 4);
    RowMatrixXdMap _im0(const_cast<double*>(im0), hgt, wid);
    RowMatrixXdMap _im1(const_cast<double*>(im1), hgt, wid);
    Rectifier<> rectifier(_P0,_P1,_im0,_im1,nchan);
    rectifier.resample(sampling_factor);
    if ( nchan == 1 ) {
        ndarray_copy_matrix(rectifier.rectified0(), rectified0);
        ndarray_copy_matrix(rectifier.rectified1(), rectified1);
    } else if (nchan > 1) {
        typedef typename RowMatrixXd::Scalar Scalar;
        {
            ndarray_set_size(rectified0,rectifier.rows(),rectifier.cols(),nchan);
            ndarray_alloc(rectified0);
            auto mat = rectifier.rectified0();
            std::copy(mat.data(),mat.data()+mat.size(),(Scalar*)rectified0->m_data);
        }
        {
            ndarray_set_size(rectified1,rectifier.rows(),rectifier.cols(),nchan);
            ndarray_alloc(rectified1);
            auto mat = rectifier.rectified1();
            std::copy(mat.data(),mat.data()+mat.size(),(Scalar*)rectified1->m_data);
        }
    }
    ndarray_copy_matrix(rectifier.rectified_idx0(), rectified_idx0);
    ndarray_copy_matrix(rectifier.rectified_idx1(), rectified_idx1);
}


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

}
