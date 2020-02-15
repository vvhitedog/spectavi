#ifndef SPECTAVI_SIFT_H
#define SPECTAVI_SIFT_H

#include "EigenDefinitions.h"
#include <iostream>
#include <list>

#include <vl/generic.h>
#include <vl/sift.h>

namespace spectavi {

#define SIFT_KP_SIZE 132

class SiftFilter {

  typedef RowMatrixXf MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::Map<MatrixType> MatrixTypeMap;

public:
  class sift_data {
  public:
    double *m_data;
    sift_data() { m_data = new double[SIFT_KP_SIZE]; }
    ~sift_data() { delete[] m_data; }
    double &x() { return m_data[0]; }
    double &y() { return m_data[1]; }
    double &sigma() { return m_data[2]; }
    double &angle() { return m_data[3]; }
    double *desc() { return m_data + 4; }
  };

private:
  MatrixTypeMap m_im;

  double m_edge_thresh = -1;
  double m_peak_thresh = -1;
  double m_magnif = -1;
  int m_O = -1;
  int m_S = 3;
  int m_omin = -1;

  std::list<sift_data> m_sift_kps; // keypoint-descriptors

public:
  SiftFilter(const Eigen::Ref<const MatrixType> &im)
      : m_im(const_cast<Scalar *>(im.data()), im.rows(), im.cols()) {}

  void filter() {

    // initialize variables
    char err_msg[1024];
    int err = VL_ERR_OK;
    bool first = true;
    VlSiftFilt *filt = nullptr;
    auto wid = m_im.cols();
    auto hgt = m_im.rows();
    float *fdata = m_im.data();

    // make filter
    filt = vl_sift_new(wid, hgt, m_O, m_S, m_omin);
    if (!filt) {
      snprintf(err_msg, sizeof(err_msg), "Could not create SIFT filter.");
      err = VL_ERR_ALLOC;
      goto done;
    }
    if (m_edge_thresh >= 0)
      vl_sift_set_edge_thresh(filt, m_edge_thresh);
    if (m_peak_thresh >= 0)
      vl_sift_set_peak_thresh(filt, m_peak_thresh);
    if (m_magnif >= 0)
      vl_sift_set_magnif(filt, m_magnif);

    // process each octave
    while (true) {

      VlSiftKeypoint const *keys = 0;
      int nkeys;

      if (first) {
        first = false;
        err = vl_sift_process_first_octave(filt, fdata);
      } else {
        err = vl_sift_process_next_octave(filt);
      }
      if (err) {
        err = VL_ERR_OK;
        break;
      }

      // run detector
      vl_sift_detect(filt);

      keys = vl_sift_get_keypoints(filt);
      nkeys = vl_sift_get_nkeypoints(filt);

      // for each keypoint
      for (int i = 0; i < nkeys; ++i) {
        double angles[4];
        int nangles;
        VlSiftKeypoint const *k;

        k = keys + i;
        nangles = vl_sift_calc_keypoint_orientations(filt, angles, k);

        // for each orientation
        for (unsigned q = 0; q < (unsigned)nangles; ++q) {
          vl_sift_pix descr[128];
          /* compute descriptor (if necessary) */
          vl_sift_calc_keypoint_descriptor(filt, descr, k, angles[q]);

          m_sift_kps.emplace_back();
          sift_data &skp = m_sift_kps.back();

          skp.x() = k->x;
          skp.y() = k->y;
          skp.sigma() = k->sigma;
          skp.angle() = angles[q];
          for (int l = 0; l < 128; ++l) {
            skp.desc()[l] = (vl_uint8)(512.0 * descr[l]);
          }
        }
      }
    }

  done:
    /* release filter */
    if (filt) {
      vl_sift_delete(filt);
      filt = 0;
    }
    /* if bad print error message */
    if (err) {
      fprintf(stderr, "sift: err: %s (%d)\n", err_msg, err);
      std::exit(1);
    }
  }

  size_t get_nkeypoints() const { return m_sift_kps.size(); }

  void get_data(float *out) const {
    RowMatrixXfMap _out((float *)out, get_nkeypoints(), SIFT_KP_SIZE);
    size_t irow = 0;
    for (auto &entry : m_sift_kps) {
      auto data = entry.m_data;
      std::copy(data, data + SIFT_KP_SIZE, _out.row(irow++).data());
    }
  }
};

} // namespace spectavi

#endif // SPECTAVI_SIFT_H
