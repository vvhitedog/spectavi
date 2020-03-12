#ifndef SPECTAVI_CAMERA_H
#define SPECTAVI_CAMERA_H

#include "EigenDefinitions.h"
#include <iostream>
#include <vector>
#include <random>

namespace spectavi {

template<typename MatrixType = RowMatrixXd>
class Camera {

private:

	MatrixType m_P;

public:

	Camera(const Eigen::Ref<const MatrixType> &P) :
			m_P(P) {
	}

	Camera(const Eigen::Ref<const MatrixType> &R,
			const Eigen::Ref<const MatrixType> &t) :
			m_P(3, 4) {
		m_P.block(0, 0, 3, 3) = R;
		m_P.block(0, 3, 3, 1) = t;
	}

	Camera() {
		m_P = MatrixType::Identity(3, 4);
	}

	const MatrixType &get_matrix() const {
		return m_P;
	}

};

template<typename OutputContainer, typename MatrixType>
OutputContainer Essential2Cameras(const Eigen::Ref<const MatrixType> &E) {
	OutputContainer ret;
	MatrixType D(3, 3);
	D << 0., 1., 0., -1., 0., 0., 0., 0., 1.;
	Eigen::JacobiSVD<MatrixType> svd(E,
			Eigen::ComputeFullV | Eigen::ComputeFullU);
	MatrixType t = svd.matrixU().col(2);
	MatrixType Ra = svd.matrixU() * D * svd.matrixV().transpose();
	MatrixType Rb = svd.matrixU() * D.transpose() * svd.matrixV().transpose();
	ret.push_back(Camera<>(Ra, t));
	ret.push_back(Camera<>(Ra, (-t).eval()));
	ret.push_back(Camera<>(Rb, t));
	ret.push_back(Camera<>(Rb, -t));
	return ret;
}

template<typename MatrixType = RowMatrixXd>
MatrixType skew_symmetric(const Eigen::Ref<const MatrixType> &s) {
	MatrixType mat = MatrixType::Zero(3, 3);
	mat(0, 1) = -s(2, 0);
	mat(0, 2) = s(1, 0);
	mat(1, 0) = s(2, 0);
	mat(1, 2) = -s(0, 0);
	mat(2, 0) = -s(1, 0);
	mat(2, 1) = s(0, 0);
	return mat;
}

template<typename MatrixType = RowMatrixXd, typename MatrixTypeI = RowMatrixXi>
class Rectifier {

	typedef typename MatrixType::Scalar Scalar;
    typedef Eigen::Map<const MatrixType> MatrixTypeMap;

private:

    MatrixTypeMap m_P0, m_P1;
	MatrixTypeMap m_im0;
	MatrixTypeMap m_im1;

	MatrixType m_rim0;
	MatrixType m_rim1;

	MatrixTypeI m_idx0;
	MatrixTypeI m_idx1;

	int m_nchannels;
	int m_output_rows;
	int m_output_cols;

	MatrixType fundamental() const {
		Eigen::JacobiSVD<MatrixType> svd(m_P0, Eigen::ComputeFullV);
		MatrixType C = svd.matrixV().col(3); // Center of P0
		MatrixType ep = m_P1 * C; // epipole of P1
		MatrixType invP0 = m_P0.transpose()
				* (m_P0 * m_P0.transpose()).inverse(); // P0 pseudoinverse
		return skew_symmetric<MatrixType>(ep) * m_P1 * invP0; // The fundamental matrix of P0, P1
	}

	MatrixType linspace(double start, double end, int num) {
		double delta = (end - start) / (num - 1);
		MatrixType ret(1, num);
		for (int i = 0; i < num; ++i) {
			ret(i) = start + i * delta;
		}
		return ret;
	}

    MatrixType compute_line(const Eigen::Ref<const MatrixType> &line,
			const Eigen::Ref<const MatrixType> &xx) const {
		// l = [l0,l1,l2]
		// l . [x,y,1] = 0
		// <==>
		// l0*x + l1*y + l2 = 0
		// (do not divide by l2!)
		MatrixType l = line;
		MatrixType yy = (-l(2, 0) - (l(0, 0) * xx).array()) / l(1, 0);
		return yy;
	}

	MatrixTypeI resample_line_idx(const Eigen::Ref<const MatrixType> &yy,
			const Eigen::Ref<const MatrixType> &xx,
			const Eigen::Ref<const MatrixTypeI> &im) const {
		MatrixTypeI resamp(1, xx.size());
		for (int i = 0; i < xx.size(); ++i) {
			double x = xx(0, i);
			double y = yy(0, i);
			int _x = (int) x;
			int _y = (int) y;
			if (_x >= 0 && _x < im.cols() && _y >= 0
					&& _y < im.rows()) {
				resamp(i) = im(_y, _x);
			} else {
				resamp(i) = -1;
			}
		}
		return resamp;
	}

	MatrixType resample_line(const Eigen::Ref<const MatrixType> &yy,
			const Eigen::Ref<const MatrixType> &xx,
			const Eigen::Ref<const MatrixType> &im) const {
		MatrixType resamp(1, m_nchannels * xx.size());
		for (int i = 0; i < xx.size(); ++i) {
			double x = xx(0, i);
			double y = yy(0, i);
			int _x = (int) x;
			int _y = (int) y;
			if (_x >= 0 && _x < im.cols() / m_nchannels && _y >= 0
					&& _y < im.rows()) {
				for (int ic = 0; ic < m_nchannels; ++ic) {
					resamp(m_nchannels * i + ic) = im(_y,
							m_nchannels * _x + ic);
				}
			} else {
				for (int ic = 0; ic < m_nchannels; ++ic) {
					resamp(m_nchannels * i + ic) = 0;
				}
			}
		}
		return resamp;
	}

	int globally_align_rows(const Eigen::Ref<const MatrixType> &r0,
			const Eigen::Ref<const MatrixType> &r1) const {
		int nx0 = r0.size() / m_nchannels;
		int nx1 = r1.size() / m_nchannels;
		int best_shift = 0;
		double best_score = 0;
		for (int shift = -nx1; shift < nx1; ++shift) {
			double sum2_0 = 0;
			double sum_0 = 0;
			double sum2_1 = 0;
			double sum_1 = 0;
			double xcorr = 0;
			int count = 0;
			for (int ii = 0; ii < nx1; ++ii) {
				for (int ic = 0; ic < m_nchannels; ++ic) {
					int idx0 = ii * m_nchannels + ic;
					int idx1 = (ii + shift) * m_nchannels + ic;
					if (idx0 >= 0 && idx0 < nx0 && idx1 >= 0 && idx1 < nx1) {
						double v0 = r0(0, idx0);
						double v1 = r1(0, idx1);
						xcorr += v0 * v1;
						sum_0 += v0;
						sum2_0 += v0 * v0;
						sum_1 += v1;
						sum2_1 += v1 * v1;
						count++;
					}
				}
			}
			double inv_n = (1. / (double) (count));
			double numerator = xcorr - inv_n * sum_0 * sum_1;
			double nvar0 = sum2_0 - inv_n * sum_0 * sum_0;
			double nvar1 = sum2_1 - inv_n * sum_1 * sum_1;
			double norm0 = std::sqrt(nvar0 + 1e-8);
			double norm1 = std::sqrt(nvar1 + 1e-8);
			double ncc = numerator / norm0 / norm1;
			if (ncc > best_score && count >= .6 * nx1) {
				best_score = ncc;
				best_shift = shift;
			}
		}
		return best_shift;
	}

	std::pair<double, int> compute_shift_row_ncc(
			const Eigen::Ref<const MatrixType> &r0,
			const Eigen::Ref<const MatrixType> &r1, int shift) const {
		int nx0 = r0.size() / m_nchannels;
		int nx1 = r1.size() / m_nchannels;
		double sum2_0 = 0;
		double sum_0 = 0;
		double sum2_1 = 0;
		double sum_1 = 0;
		double xcorr = 0;
		int count = 0;
		for (int ii = 0; ii < nx1; ++ii) {
			for (int ic = 0; ic < m_nchannels; ++ic) {
				int idx0 = ii * m_nchannels + ic;
				int idx1 = (ii + shift) * m_nchannels + ic;
				if (idx0 >= 0 && idx0 < nx0 && idx1 >= 0 && idx1 < nx1) {
					double v0 = r0(0, idx0);
					double v1 = r1(0, idx1);
					xcorr += v0 * v1;
					sum_0 += v0;
					sum2_0 += v0 * v0;
					sum_1 += v1;
					sum2_1 += v1 * v1;
					count++;
				}
			}
		}
		double inv_n = (1. / (double) (count));
		double numerator = xcorr - inv_n * sum_0 * sum_1;
		double nvar0 = sum2_0 - inv_n * sum_0 * sum_0;
		double nvar1 = sum2_1 - inv_n * sum_1 * sum_1;
		double norm0 = std::sqrt(nvar0 + 1e-8);
		double norm1 = std::sqrt(nvar1 + 1e-8);
		double ncc = numerator / norm0 / norm1;
		return std::make_pair(ncc, count);
	}

	template<typename OutputContainer>
	OutputContainer draw_random_indices(int start, int end, int k) const {
		OutputContainer ret(k);
		std::vector<int> arange(end - start, 0);
		for (int i = start; i < end; ++i) {
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

public:

    Rectifier(const Scalar *P0,
            const Scalar *P1,
            const Scalar *im0,
            const Scalar *im1, int rows0, int cols0,
            int rows1, int cols1, int nchannels) :
            m_P0(P0,3,4), m_P1(P1,3,4),
            m_im0(im0,rows0,cols0*nchannels),
            m_im1(im1,rows1,cols1*nchannels),
            m_nchannels(nchannels),
            m_output_rows(0), m_output_cols(0) {
	}

	Rectifier& resample(double sampling_factor) {

		MatrixType F = fundamental();
		// Resize output to appropriate size
		int largest_cols = std::max(m_im0.cols(), m_im1.cols());
		int largest_rows = std::max(m_im0.rows(), m_im1.rows());
		int output_cols = sampling_factor * largest_cols / m_nchannels;
		int extra_rows = (std::max(largest_rows, largest_cols)) / 2.;
		int output_rows = largest_rows + 2 * extra_rows;
		// Extra rows above and below
		int zero_row = extra_rows;
		m_rim0 = MatrixType::Zero(output_rows, output_cols * m_nchannels);
		m_rim1 = MatrixType::Zero(output_rows, output_cols * m_nchannels);
        m_idx0 = MatrixTypeI::Zero(output_rows, output_cols).array() - 1;
        m_idx1 = MatrixTypeI::Zero(output_rows, output_cols).array() - 1;
		m_output_rows = output_rows;
		m_output_cols = output_cols;
        int rows = m_im0.rows();
        int cols = m_im1.cols()/m_nchannels;
        MatrixTypeI idxs(rows,cols);
        for ( int i = 0; i < rows*cols; ++i ) {
            idxs.data()[i] = i; // row-major counting
        }
		for (int irow = -extra_rows; irow < largest_rows + extra_rows; ++irow) {
			MatrixType origin(3, 1);
			origin << 0., irow, 1.;
			MatrixType seed(3, 1); // A point in im0 that will define the epipolar line in im1
			// 1. using an origin point in im1 find the epipolar line in im0
			// 2. pick the first point on this line to be the seed
			// 3. resample im0
			// 4. using the seed find epipolar line in im1
			// 5. resample im1
			{
				// 1.
				MatrixType line = F.transpose() * origin;
				int nx = m_im0.cols() / m_nchannels;
				int rnx = sampling_factor * nx;
				MatrixType xx = linspace(0, nx - 1, rnx);
				MatrixType yy = compute_line(line, xx);
				// 2.
				seed << xx(0), yy(0), 1.;
				int itarget_row = irow + zero_row; // shift indices to get to zero
				if (itarget_row >= 0 && itarget_row < m_rim0.rows()) {
					// 3.
					MatrixType resamp = resample_line(yy, xx, m_im0);
                    MatrixTypeI resampI = resample_line_idx(yy, xx, idxs);
                    m_rim0.block(itarget_row, 0, 1, m_nchannels * rnx) = resamp;
                    m_idx0.block(itarget_row, 0, 1, rnx) = resampI;
                }
			}
			{
				// 4.
				MatrixType line = F * seed;
				int nx = m_im1.cols() / m_nchannels;
				int rnx = sampling_factor * nx;
				MatrixType xx = linspace(0, nx - 1, rnx);
				MatrixType yy = compute_line(line, xx);
				int itarget_row = irow + zero_row; // shift indices to get to zero
				if (itarget_row >= 0 && itarget_row < m_rim1.rows()) {
					// 5.
					MatrixType resamp = resample_line(yy, xx, m_im1);
                    MatrixTypeI resampI = resample_line_idx(yy, xx, idxs);
                    m_rim1.block(itarget_row, 0, 1, m_nchannels * rnx) = resamp;
                    m_idx1.block(itarget_row, 0, 1, rnx) = resampI;
                }
			}
		}

		// Shift the right image consistently in columns to best align the two images.
		const int k = 5;
		MatrixType sol;

		double best_score = -1;
		for (int itry = 0; itry < 0; ++itry) {
			MatrixType ys(k, 2), xs(k, 1);
			std::vector<int> indices = draw_random_indices<std::vector<int> >(0,
					largest_rows, k);
			for (int ik = 0; ik < k; ++ik) {
				int idx = indices[ik];
				int itarget_row = idx + zero_row; // shift indices to get to zero
				int shift = globally_align_rows(m_rim0.row(itarget_row),
						m_rim1.row(itarget_row));
				xs(ik) = shift;
				ys(ik, 0) = idx;
			}
			ys.col(1) = MatrixType::Ones(k, 1);
			Eigen::JacobiSVD<MatrixType> svd(ys,
					Eigen::ComputeThinV | Eigen::ComputeThinU);
			MatrixType _sol = svd.solve(xs);
			double avg_ncc = 0;
			int count = 0;
			for (int irow = 0; irow < largest_rows; ++irow) {
				int shift = _sol(0) * irow + _sol(1);
				int itarget_row = irow + zero_row; // shift indices to get to zero
				auto score = compute_shift_row_ncc(m_rim0.row(itarget_row),
						m_rim1.row(itarget_row), shift);
				avg_ncc += score.first;
				count++;
			}
			avg_ncc /= count;
			if (avg_ncc > best_score) {
				best_score = avg_ncc;
				sol = _sol;
			}
		}


        //TODO:
        // Is all the code below this point very slow??


		//best_ys.col(1) = MatrixType::Zero(k,1);
		//Eigen::JacobiSVD<MatrixType> svd(best_ys,
		//        Eigen::ComputeThinV |  Eigen::ComputeThinU );
		//MatrixType sol = svd.solve(best_xs);

		int min_left = 0, max_right = m_output_cols;
		for (int irow = -extra_rows; irow < largest_rows + extra_rows; ++irow) {
			// int shift = sol(0) * irow + sol(1);
			int shift = 0;
			min_left = std::min(min_left, -shift);
			max_right = std::max(max_right, m_output_cols - shift);
		}
        m_output_cols = max_right - min_left; // reset output cols correctly
        {
            // Shift rectified images to create best match
            MatrixType copy0(m_output_rows, (m_output_cols) * m_nchannels);
            MatrixType copy1(m_output_rows, (m_output_cols) * m_nchannels);
            for (int irow = -extra_rows; irow < largest_rows + extra_rows; ++irow) {
                int itarget_row = irow + zero_row; // shift indices to get to zero
                // int shift = sol(0) * irow + sol(1);
                int shift = 0;
                copy0.block(itarget_row, m_nchannels * (-min_left), 1,
                            m_rim0.cols()) = m_rim0.row(itarget_row);
                copy1.block(itarget_row, m_nchannels * (-shift - min_left), 1,
                            m_rim1.cols()) = m_rim1.row(itarget_row);
            }
            m_rim0 = copy0;
            m_rim1 = copy1;
        }

        {
            // Apply sample shift for idx
            MatrixTypeI copy0(m_output_rows, (m_output_cols) );
            MatrixTypeI copy1(m_output_rows, (m_output_cols) );
            copy0.array() = -1;
            copy1.array() = -1;
            for (int irow = -extra_rows; irow < largest_rows + extra_rows; ++irow) {
                int itarget_row = irow + zero_row; // shift indices to get to zero
                // int shift = sol(0) * irow + sol(1);
                int shift = 0;
                copy0.block(itarget_row,  (-min_left), 1,
                            m_idx0.cols()) = m_idx0.row(itarget_row);
                copy1.block(itarget_row,  (-shift - min_left), 1,
                            m_idx1.cols()) = m_idx1.row(itarget_row);
            }
            m_idx0 = copy0;
            m_idx1 = copy1;
        }

		//for ( int irow = 500; irow < largest_rows + extra_rows; ++irow ) {
		//    int itarget_row = irow+zero_row; // shift indices to get to zero
		//    int shift = globally_align_rows(m_rim0.row(itarget_row),m_rim1.row(itarget_row));
		//    std::cout <<  " done rows: " << irow << " -> " << largest_rows + extra_rows << " shift: " <<
		//       shift  << std::endl;
		//}
		return *this;
	}

	int rows() const {
		return m_output_rows;
	}

	int cols() const {
		return m_output_cols;
	}

	MatrixType rectified0() const {
		return m_rim0;
	}

	MatrixType rectified1() const {
		return m_rim1;
	}

    MatrixTypeI rectified_idx0() const {
        return m_idx0;
    }

    MatrixTypeI rectified_idx1() const {
        return m_idx1;
    }

};

}

#endif// SPECTAVI_CAMERA_H
