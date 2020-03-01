#ifndef DLTTRIANGULATOR_H
#define DLTTRIANGULATOR_H

#include "EigenDefinitions.h"

namespace spectavi {


template<typename MatrixType = RowMatrixXd>
class DltTriangulator {

private:

	typedef typename MatrixType::Scalar Scalar;
	typedef Eigen::Map<MatrixType> MatrixTypeMap;

	MatrixTypeMap m_P0;
	MatrixTypeMap m_P1;
	MatrixType m_X;
	MatrixType m_x0;
	MatrixType m_x1;
	MatrixType m_rp_x0;
	MatrixType m_rp_x1;

	double m_sign_det_M0;
	double m_sign_det_M1;
	double m_norm_m0;
	double m_norm_m1;

public:

	DltTriangulator(const Scalar *P0, const Scalar *P1) :
			m_P0(const_cast<Scalar*>(P0), 3, 4),
			m_P1(const_cast<Scalar*>(P1), 3, 4) {
		m_sign_det_M0 = m_P0.block(0, 0, 3, 3).determinant() < 0 ? -1. : 1.;
		m_sign_det_M1 = m_P1.block(0, 0, 3, 3).determinant() < 0 ? -1. : 1.;
        m_norm_m0 = m_P0.block(0, 2, 3, 1).array().square().sum();
        m_norm_m1 = m_P1.block(0, 2, 3, 1).array().square().sum();
	}

	DltTriangulator& solve(const Scalar *_x0, const Scalar *_x1, int len) {
		MatrixType A(4, 4);
		m_x0 = MatrixTypeMap(const_cast<Scalar*>(_x0), len, 1);
		if (len == 3) {
			m_x0 = m_x0.colwise().hnormalized().eval();
		}
		m_x1 = MatrixTypeMap(const_cast<Scalar*>(_x1), len, 1);
		if (len == 3) {
			m_x1 = m_x1.colwise().hnormalized().eval();
		}
		Scalar x, y, xp, yp;
		x = m_x0(0);
		y = m_x0(1);
		xp = m_x1(0);
		yp = m_x1(1);
		A.row(0) = x * m_P0.row(2) - m_P0.row(0);
		A.row(1) = y * m_P0.row(2) - m_P0.row(1);
		A.row(2) = xp * m_P1.row(2) - m_P1.row(0);
		A.row(3) = yp * m_P1.row(2) - m_P1.row(1);

		Eigen::JacobiSVD<MatrixType> svd(A, Eigen::ComputeFullV);
		MatrixType V = svd.matrixV();
		m_X = V.col(3);

		// Reproject
		m_rp_x0 = (m_P0 * m_X);
		m_rp_x1 = (m_P1 * m_X);

		return *this;
	}

	double reprojection_error() const {
		return std::sqrt(
				(m_rp_x0.colwise().hnormalized() - m_x0).array().square().sum())
				+ std::sqrt(
						(m_rp_x1.colwise().hnormalized() - m_x1).array().square().sum());
	}

	double distance2camera0() const {
		return m_sign_det_M0 / m_norm_m0 * m_rp_x0(2) / m_X(3);
	}

	double distance2camera1() const {
		return m_sign_det_M1 / m_norm_m1 * m_rp_x1(2) / m_X(3);
	}

	bool is_infront_both_cameras() const {
		return (distance2camera0() > 0) && (distance2camera1() > 0);
	}

	MatrixType X() {
		return m_X;
	}

};

}

#endif//DLTTRIANGULATOR_H
