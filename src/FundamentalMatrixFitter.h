//
// Created by mgara on 7/19/17.
//

#ifndef SFM_FUNDAMENTALMATRIXFITTER_H
#define SFM_FUNDAMENTALMATRIXFITTER_H

#include <Eigen/Core>
#include <Eigen/Dense>

#define    TWOPI  6.28318530717958648

template<typename MatrixType = Eigen::MatrixXd>
class FundamentalMatrixFitter {

private:

    /*
     * Implements the 7-point algorithm; meant to be used with RANSAC.
     */

    MatrixType m_A;
    int m_current_row_idx;

    const double m_eps;

    //=============================================================================
    // _root3, root3 from http://prografix.narod.ru
    //=============================================================================
    static double _root3(double x) {
        double s = 1.;
        while (x < 1.) {
            x *= 8.;
            s *= 0.5;
        }
        while (x > 8.) {
            x *= 0.125;
            s *= 2.;
        }
        double r = 1.5;
        r -= 1. / 3. * (r - x / (r * r));
        r -= 1. / 3. * (r - x / (r * r));
        r -= 1. / 3. * (r - x / (r * r));
        r -= 1. / 3. * (r - x / (r * r));
        r -= 1. / 3. * (r - x / (r * r));
        r -= 1. / 3. * (r - x / (r * r));
        return r * s;
    }

    double root3(double x) {
        if (x > 0) return _root3(x);
        else if (x < 0) return -_root3(-x);
        else
            return 0.;
    }

    // These functions are taken from:
    // http://math.ivanovo.ac.ru/dalgebra/Khashin/poly/index.html
    //---------------------------------------------------------------------------
    // x - array of size 3
    // In case 3 real roots: => x[0], x[1], x[2], return 3
    //         2 real roots: x[0], x[1],          return 2
    //         1 real root : x[0], x[1] ± i*x[2], return 1
    int solve_cubic(double *x, double a, double b, double c) {    // solve cubic equation x^3 + a*x^2 + b*x + c = 0
        double a2 = a * a;
        double q = (a2 - 3 * b) / 9;
        double r = (a * (2 * a2 - 9 * b) + 27 * c) / 54;
        // equation x^3 + q*x + r = 0
        double r2 = r * r;
        double q3 = q * q * q;
        double A, B;
        if (r2 < q3) {
            double t = r / sqrt(q3);
            if (t < -1) t = -1;
            if (t > 1) t = 1;
            t = acos(t);
            a /= 3;
            q = -2 * sqrt(q);
            x[0] = q * cos(t / 3) - a;
            x[1] = q * cos((t + TWOPI) / 3) - a;
            x[2] = q * cos((t - TWOPI) / 3) - a;
            return (3);
        } else {
            A =-pow(fabs(r)+sqrt(r2-q3),1./3);
            //A = -root3(fabs(r) + sqrt(r2 - q3));
            if (r < 0) A = -A;
            B = A == 0 ? 0 : B = q / A;

            a /= 3;
            x[0] = (A + B) - a;
            x[1] = -0.5 * (A + B) - a;
            x[2] = 0.5 * sqrt(3.) * (A - B);
            if (fabs(x[2]) < m_eps) {
                x[2] = x[1];
                return (2);
            }
            return (1);
        }
    }


public:

    FundamentalMatrixFitter() : m_A(7, 9), m_current_row_idx(0), m_eps(1e-14) {}

    template<typename T>
    void add_putative_match(const T &x, const T &y, const T &xp, const T &yp) {
        assert(m_current_row_idx < m_A.rows());
        Eigen::VectorXd row(9);
        row(0) = xp * x;
        row(1) = xp * y;
        row(2) = xp;
        row(3) = yp * x;
        row(4) = yp * y;
        row(5) = yp;
        row(6) = x;
        row(7) = y;
        row(8) = 1.;
        m_A.row(m_current_row_idx++) = row.transpose();
    }

    void reset() {
        m_current_row_idx = 0;
    }

    int solve(MatrixType &sol0, MatrixType &sol1, MatrixType &sol2) {
        assert(m_current_row_idx == 7);
        Eigen::JacobiSVD<MatrixType> svd(m_A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        MatrixType V = svd.matrixV();
        Eigen::VectorXd _F0 = V.col(7);
        Eigen::VectorXd _F1 = V.col(8);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > F0(_F0.data(), 3, 3);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > F1(_F1.data(), 3, 3);

        // a z^3 + b z^2 + c z + d

        double a = -F0(0, 2) * F0(1, 1) * F0(2, 0) + F0(0, 1) * F0(1, 2) * F0(2, 0) + F0(0, 2) * F0(1, 0) * F0(2, 1) -
                   F0(0, 0) * F0(1, 2) * F0(2, 1) - F0(1, 2) * F0(2, 0) * F1(0, 1) + F0(1, 1) * F0(2, 0) * F1(0, 2) -
                   F0(1, 0) * F0(2, 1) * F1(0, 2)
                   - F0(0, 2) * F0(2, 1) * F1(1, 0) + F0(2, 1) * F1(0, 2) * F1(1, 0) + F0(0, 2) * F0(2, 0) * F1(1, 1) -
                   F0(2, 0) * F1(0, 2) * F1(1, 1) - F0(0, 1) * F0(2, 0) * F1(1, 2) + F0(0, 0) * F0(2, 1) * F1(1, 2) +
                   F0(2, 0) *
                   F1(0, 1) * F1(1, 2) + F0(0, 2) * F0(1, 1) * F1(2, 0) - F0(0, 1) * F0(1, 2) * F1(2, 0) +
                   F0(1, 2) * F1(0, 1) * F1(2, 0) - F0(1, 1) * F1(0, 2) * F1(2, 0) - F0(0, 2) * F1(1, 1) * F1(2, 0) +
                   F1(0, 2) * F1(1, 1) * F1(2, 0)
                   + F0(0, 1) * F1(1, 2) * F1(2, 0) - F1(0, 1) * F1(1, 2) * F1(2, 0) - F0(0, 2) * F0(1, 0) * F1(2, 1) +
                   F0(0, 0) * F0(1, 2) * F1(2, 1) + F0(1, 0) * F1(0, 2) * F1(2, 1) + F0(0, 2) * F1(1, 0) * F1(2, 1) -
                   F1(0, 2) * F1(1, 0) * F1(2, 1)
                   - F0(0, 0) * F1(1, 2) * F1(2, 1) + F0(0, 1) * F0(1, 0) * F1(2, 2) - F0(0, 0) * F0(1, 1) * F1(2, 2) -
                   F0(1, 0) * F1(0, 1) * F1(2, 2) - F0(0, 1) * F1(1, 0) * F1(2, 2) + F1(0, 1) * F1(1, 0) * F1(2, 2) +
                   F0(0, 0) *
                   F1(1, 1) * F1(2, 2) + F0(1, 2) * F0(2, 1) * F1(0, 0) - F0(2, 1) * F1(1, 2) * F1(0, 0) -
                   F0(1, 2) * F1(2, 1) * F1(0, 0) + F1(1, 2) * F1(2, 1) * F1(0, 0) + F0(1, 1) * F1(2, 2) * F1(0, 0) -
                   F1(1, 1) * F1(2, 2) * F1(0, 0)
                   - F0(0, 1) * F0(1, 0) * F0(2, 2) + F0(0, 0) * F0(1, 1) * F0(2, 2) + F0(1, 0) * F1(0, 1) * F0(2, 2) +
                   F0(0, 1) * F1(1, 0) * F0(2, 2) - F1(0, 1) * F1(1, 0) * F0(2, 2) - F0(0, 0) * F1(1, 1) * F0(2, 2) -
                   F0(1, 1) * F1(0, 0) * F0(2, 2)
                   + F1(1, 1) * F1(0, 0) * F0(2, 2);

        double b = F0(1, 2) * F0(2, 0) * F1(0, 1) - F0(1, 1) * F0(2, 0) * F1(0, 2) + F0(1, 0) * F0(2, 1) * F1(0, 2) +
                   F0(0, 2) * F0(2, 1) * F1(1, 0) - 2 * F0(2, 1) * F1(0, 2) * F1(1, 0) -
                   F0(0, 2) * F0(2, 0) * F1(1, 1) + 2 * F0(2, 0) * F1(0, 2) * F1(1, 1) +
                   F0(0, 1) * F0(2, 0) * F1(1, 2) - F0(0, 0) * F0(2, 1) * F1(1, 2) -
                   2 * F0(2, 0) * F1(0, 1) * F1(1, 2) - F0(0, 2) * F0(1, 1) * F1(2, 0) + F0(0, 1) * F0(1, 2) *
                   F1(2, 0) -
                   2 * F0(1, 2) * F1(0, 1) * F1(2, 0) + 2 * F0(1, 1) * F1(0, 2) * F1(2, 0) +
                   2 * F0(0, 2) * F1(1, 1) * F1(2, 0) - 3 * F1(0, 2) * F1(1, 1) * F1(2, 0) -
                   2 * F0(0, 1) * F1(1, 2) * F1(2, 0) + 3 *
                   F1(0, 1) * F1(1, 2) * F1(2, 0) +
                   F0(0, 2) * F0(1, 0) * F1(2, 1) - F0(0, 0) * F0(1, 2) * F1(2, 1) -
                   2 * F0(1, 0) * F1(0, 2) * F1(2, 1) - 2 * F0(0, 2) * F1(1, 0) * F1(2, 1) +
                   3 * F1(0, 2) * F1(1, 0) * F1(2, 1)
                   + 2 *
                   F0(0, 0) * F1(1, 2) * F1(2, 1) - F0(0, 1) * F0(1, 0) * F1(2, 2) + F0(0, 0) * F0(1, 1) * F1(2, 2) +
                   2 * F0(1, 0) * F1(0, 1) * F1(2, 2) + 2 * F0(0, 1) * F1(1, 0) * F1(2, 2) -
                   3 * F1(0, 1) * F1(1, 0) * F1(2, 2)
                   - 2 *
                   F0(0, 0) * F1(1, 1) * F1(2, 2) - F0(1, 2) * F0(2, 1) * F1(0, 0) +
                   2 * F0(2, 1) * F1(1, 2) * F1(0, 0) +
                   2 * F0(1, 2) * F1(2, 1) * F1(0, 0) - 3 * F1(1, 2) * F1(2, 1) * F1(0, 0) -
                   2 * F0(1, 1) * F1(2, 2) * F1(0, 0) +
                   3 * F1(1, 1) * F1(2, 2) * F1(0, 0) - F0(1, 0) * F1(0, 1) * F0(2, 2) -
                   F0(0, 1) * F1(1, 0) * F0(2, 2) +
                   2 * F1(0, 1) * F1(1, 0) * F0(2, 2) + F0(0, 0) * F1(1, 1) * F0(2, 2) +
                   F0(1, 1) * F1(0, 0) * F0(2, 2) -
                   2 * F1(1, 1) *
                   F1(0, 0) * F0(2, 2);

        double c = F0(2, 1) * F1(0, 2) * F1(1, 0) - F0(2, 0) * F1(0, 2) * F1(1, 1) + F0(2, 0) * F1(0, 1) * F1(1, 2) +
                   F0(1, 2) * F1(0, 1) * F1(2, 0) - F0(1, 1) * F1(0, 2) * F1(2, 0) - F0(0, 2) * F1(1, 1) * F1(2, 0) +
                   3 * F1(0, 2) * F1(1, 1) * F1(2, 0)
                   + F0(0, 1) * F1(1, 2) * F1(2, 0) - 3 * F1(0, 1) * F1(1, 2) * F1(2, 0) +
                   F0(1, 0) * F1(0, 2) * F1(2, 1) + F0(0, 2) * F1(1, 0) * F1(2, 1) -
                   3 * F1(0, 2) * F1(1, 0) * F1(2, 1) - F0(0, 0) * F1(1, 2) * F1(2, 1) -
                   F0(1, 0) * F1(0, 1) * F1(2, 2) - F0(0, 1) * F1(1, 0) *
                   F1(2, 2) + 3 * F1(0, 1) * F1(1, 0) * F1(2, 2) +
                   F0(0, 0) * F1(1, 1) * F1(2, 2) -
                   F0(2, 1) * F1(1, 2) * F1(0, 0) - F0(1, 2) * F1(2, 1) * F1(0, 0) +
                   3 * F1(1, 2) * F1(2, 1) * F1(0, 0) +
                   F0(1, 1) * F1(2, 2) * F1(0, 0) - 3 * F1(1, 1) * F1(2, 2) * F1(0, 0)
                   - F1(0, 1) * F1(1, 0) * F0(2, 2) + F1(1, 1) * F1(0, 0) * F0(2, 2);

        double d = -F1(0, 2) * F1(1, 1) * F1(2, 0) + F1(0, 1) * F1(1, 2) * F1(2, 0) + F1(0, 2) * F1(1, 0) * F1(2, 1) -
                   F1(0,
                      1) * F1(1, 0) * F1(2, 2) - F1(1, 2) * F1(2, 1) * F1(0, 0) + F1(1, 1) * F1(2, 2) * F1(0, 0);

        if (std::abs(a) < m_eps) {
            return 0;
        }

        Eigen::VectorXd alpha(3);
        int nroots = solve_cubic(alpha.data(), b / a, c / a, d / a);

        if (nroots >= 1) {
            sol0 = alpha(0) * F0 + (1 - alpha(0)) * F1;
        }
        if (nroots >= 2) {
            sol1 = alpha(1) * F0 + (1 - alpha(1)) * F1;
        }
        if (nroots == 3) {
            sol2 = alpha(2) * F0 + (1 - alpha(2)) * F1;
        }

        return nroots;


        /*
         * From Wolfram Alpha, then manipulated by me:
         *
            original:

            -c e g z^3 + b f g z^3 + c d h z^3 - a f h z^3 - f g k z^3 + e g l z^3 - d h l
            z^3 - c h m z^3 + h l m z^3 + c g n z^3 - g l n z^3 - b g o z^3 + a h o z^3 + g
            k o z^3 + c e p z^3 - b f p z^3 + f k p z^3 - e l p z^3 - c n p z^3 + l n p z^3
            + b o p z^3 - k o p z^3 - c d q z^3 + a f q z^3 + d l q z^3 + c m q z^3 - l m q
            z^3 - a o q z^3 + b d r z^3 - a e r z^3 - d k r z^3 - b m r z^3 + k m r z^3 + a
            n r z^3 + f h v z^3 - h o v z^3 - f q v z^3 + o q v z^3 + e r v z^3 - n r v z^3
            - b d x z^3 + a e x z^3 + d k x z^3 + b m x z^3 - k m x z^3 - a n x z^3 - e v x
            z^3 + n v x z^3 + f g k z^2 - e g l z^2 + d h l z^2 + c h m z^2 - 2 h l m z^2 -
            c g n z^2 + 2 g l n z^2 + b g o z^2 - a h o z^2 - 2 g k o z^2 - c e p z^2 + b f
            p z^2 - 2 f k p z^2 + 2 e l p z^2 + 2 c n p z^2 - 3 l n p z^2 - 2 b o p z^2 + 3
            k o p z^2 + c d q z^2 - a f q z^2 - 2 d l q z^2 - 2 c m q z^2 + 3 l m q z^2 + 2
            a o q z^2 - b d r z^2 + a e r z^2 + 2 d k r z^2 + 2 b m r z^2 - 3 k m r z^2 - 2
            a n r z^2 - f h v z^2 + 2 h o v z^2 + 2 f q v z^2 - 3 o q v z^2 - 2 e r v z^2 +
            3 n r v z^2 - d k x z^2 - b m x z^2 + 2 k m x z^2 + a n x z^2 + e v x z^2 - 2 n
            v x z^2 + h l m z - g l n z + g k o z + f k p z - e l p z - c n p z + 3 l n p z
            + b o p z - 3 k o p z + d l q z + c m q z - 3 l m q z - a o q z - d k r z - b m
            r z + 3 k m r z + a n r z - h o v z - f q v z + 3 o q v z + e r v z - 3 n r v z
            - k m x z + n v x z - l n p + k o p + l m q - k m r - o q v + n r v

            rewritten:

            z^3 *
            (
            -c e g  + b f g  + c d h  - a f h  - f g k  + e g l  - d h l
             - c h m  + h l m  + c g n  - g l n  - b g o  + a h o  + g
            k o  + c e p  - b f p  + f k p  - e l p  - c n p  + l n p
            + b o p  - k o p  - c d q  + a f q  + d l q  + c m q  - l m q
             - a o q  + b d r  - a e r  - d k r  - b m r  + k m r  + a
            n r  + f h v  - h o v  - f q v  + o q v  + e r v  - n r v
            - b d x  + a e x  + d k x  + b m x  - k m x  - a n x  - e v x
             + n v x
            )

            z^2 *
            (
            f g k  - e g l  + d h l  + c h m  - 2 h l m  -
            c g n  + 2 g l n  + b g o  - a h o  - 2 g k o  - c e p  + b f
            p  - 2 f k p  + 2 e l p  + 2 c n p  - 3 l n p  - 2 b o p  + 3
            k o p  + c d q  - a f q  - 2 d l q  - 2 c m q  + 3 l m q  + 2
            a o q  - b d r  + a e r  + 2 d k r  + 2 b m r  - 3 k m r  - 2
            a n r  - f h v  + 2 h o v  + 2 f q v  - 3 o q v  - 2 e r v  +
            3 n r v  - d k x  - b m x  + 2 k m x  + a n x  + e v x  - 2 n
            v x
            )

            z *
            (
            h l m  - g l n  + g k o  + f k p  - e l p  - c n p  + 3 l n p
            + b o p  - 3 k o p  + d l q  + c m q  - 3 l m q  - a o q  - d k r  - b m
            r  + 3 k m r  + a n r  - h o v  - f q v  + 3 o q v  + e r v  - 3 n r v
            - k m x  + n v x
            )

            1 *
            (
             - l n p + k o p + l m q - k m r - o q v + n r v
            )


            rewritten again:

            z * F0 + (1-z) * F1
            F0 = [[a,b,c],[d,e,f],[g,h,x]]
            F1 = [[v,k,l],[m,n,o],[p,q,r]]

            -----------------------------------------

            z^3 *
            (
            -F0(0,2)*F0(1,1)*F0(2,0)  + F0(0,1)*F0(1,2)*F0(2,0)  + F0(0,2)*F0(1,0)*F0(2,1)  - F0(0,0)*F0(1,2)*F0(2,1)  - F0(1,2)*F0(2,0)*F1(0,1)  + F0(1,1)*F0(2,0)*F1(0,2)  - F0(1,0)*F0(2,1)*F1(0,2)
             - F0(0,2)*F0(2,1)*F1(1,0)  + F0(2,1)*F1(0,2)*F1(1,0)  + F0(0,2)*F0(2,0)*F1(1,1)  - F0(2,0)*F1(0,2)*F1(1,1)  - F0(0,1)*F0(2,0)*F1(1,2)  + F0(0,0)*F0(2,1)*F1(1,2)  + F0(2,0)
            F1(0,1)*F1(1,2)  + F0(0,2)*F0(1,1)*F1(2,0)  - F0(0,1)*F0(1,2)*F1(2,0)  + F0(1,2)*F1(0,1)*F1(2,0)  - F0(1,1)*F1(0,2)*F1(2,0)  - F0(0,2)*F1(1,1)*F1(2,0)  + F1(0,2)*F1(1,1)*F1(2,0)
            + F0(0,1)*F1(1,2)*F1(2,0)  - F1(0,1)*F1(1,2)*F1(2,0)  - F0(0,2)*F0(1,0)*F1(2,1)  + F0(0,0)*F0(1,2)*F1(2,1)  + F0(1,0)*F1(0,2)*F1(2,1)  + F0(0,2)*F1(1,0)*F1(2,1)  - F1(0,2)*F1(1,0)*F1(2,1)
             - F0(0,0)*F1(1,2)*F1(2,1)  + F0(0,1)*F0(1,0)*F1(2,2)  - F0(0,0)*F0(1,1)*F1(2,2)  - F0(1,0)*F1(0,1)*F1(2,2)  - F0(0,1)*F1(1,0)*F1(2,2)  + F1(0,1)*F1(1,0)*F1(2,2)  + F0(0,0)
            F1(1,1)*F1(2,2)  + F0(1,2)*F0(2,1)*F1(0,0)  - F0(2,1)*F1(1,2)*F1(0,0)  - F0(1,2)*F1(2,1)*F1(0,0)  + F1(1,2)*F1(2,1)*F1(0,0)  + F0(1,1)*F1(2,2)*F1(0,0)  - F1(1,1)*F1(2,2)*F1(0,0)
            - F0(0,1)*F0(1,0)*F0(2,2)  + F0(0,0)*F0(1,1)*F0(2,2)  + F0(1,0)*F1(0,1)*F0(2,2)  + F0(0,1)*F1(1,0)*F0(2,2)  - F1(0,1)*F1(1,0)*F0(2,2)  - F0(0,0)*F1(1,1)*F0(2,2)  - F0(1,1)*F1(0,0)*F0(2,2)
             + F1(1,1)*F1(0,0)*F0(2,2)
            )

            z^2 *
            (
            F0(1,2)*F0(2,0)*F1(0,1)  - F0(1,1)*F0(2,0)*F1(0,2)  + F0(1,0)*F0(2,1)*F1(0,2)  + F0(0,2)*F0(2,1)*F1(1,0)  - 2*F0(2,1)*F1(0,2)*F1(1,0)  -
            F0(0,2)*F0(2,0)*F1(1,1)  + 2*F0(2,0)*F1(0,2)*F1(1,1)  + F0(0,1)*F0(2,0)*F1(1,2)  - F0(0,0)*F0(2,1)*F1(1,2)  - 2*F0(2,0)*F1(0,1)*F1(1,2)  - F0(0,2)*F0(1,1)*F1(2,0)  + F0(0,1)*F0(1,2)
            F1(2,0)  - 2*F0(1,2)*F1(0,1)*F1(2,0)  + 2*F0(1,1)*F1(0,2)*F1(2,0)  + 2*F0(0,2)*F1(1,1)*F1(2,0)  - 3*F1(0,2)*F1(1,1)*F1(2,0)  - 2*F0(0,1)*F1(1,2)*F1(2,0)  + 3
            F1(0,1)*F1(1,2)*F1(2,0)  + F0(0,2)*F0(1,0)*F1(2,1)  - F0(0,0)*F0(1,2)*F1(2,1)  - 2*F0(1,0)*F1(0,2)*F1(2,1)  - 2*F0(0,2)*F1(1,0)*F1(2,1)  + 3*F1(0,2)*F1(1,0)*F1(2,1)  + 2
            F0(0,0)*F1(1,2)*F1(2,1)  - F0(0,1)*F0(1,0)*F1(2,2)  + F0(0,0)*F0(1,1)*F1(2,2)  + 2*F0(1,0)*F1(0,1)*F1(2,2)  + 2*F0(0,1)*F1(1,0)*F1(2,2)  - 3*F1(0,1)*F1(1,0)*F1(2,2)  - 2
            F0(0,0)*F1(1,1)*F1(2,2)  - F0(1,2)*F0(2,1)*F1(0,0)  + 2*F0(2,1)*F1(1,2)*F1(0,0)  + 2*F0(1,2)*F1(2,1)*F1(0,0)  - 3*F1(1,2)*F1(2,1)*F1(0,0)  - 2*F0(1,1)*F1(2,2)*F1(0,0)  +
            3*F1(1,1)*F1(2,2)*F1(0,0)  - F0(1,0)*F1(0,1)*F0(2,2)  - F0(0,1)*F1(1,0)*F0(2,2)  + 2*F1(0,1)*F1(1,0)*F0(2,2)  + F0(0,0)*F1(1,1)*F0(2,2)  + F0(1,1)*F1(0,0)*F0(2,2)  - 2*F1(1,1)
            F1(0,0)*F0(2,2)
            )

            z *
            (
            F0(2,1)*F1(0,2)*F1(1,0)  - F0(2,0)*F1(0,2)*F1(1,1)  + F0(2,0)*F1(0,1)*F1(1,2)  + F0(1,2)*F1(0,1)*F1(2,0)  - F0(1,1)*F1(0,2)*F1(2,0)  - F0(0,2)*F1(1,1)*F1(2,0)  + 3*F1(0,2)*F1(1,1)*F1(2,0)
            + F0(0,1)*F1(1,2)*F1(2,0)  - 3*F1(0,1)*F1(1,2)*F1(2,0)  + F0(1,0)*F1(0,2)*F1(2,1)  + F0(0,2)*F1(1,0)*F1(2,1)  - 3*F1(0,2)*F1(1,0)*F1(2,1)  - F0(0,0)*F1(1,2)*F1(2,1)  - F0(1,0)*F1(0,1)*F1(2,2)  - F0(0,1)*F1(1,0)
            F1(2,2)  + 3*F1(0,1)*F1(1,0)*F1(2,2)  + F0(0,0)*F1(1,1)*F1(2,2)  - F0(2,1)*F1(1,2)*F1(0,0)  - F0(1,2)*F1(2,1)*F1(0,0)  + 3*F1(1,2)*F1(2,1)*F1(0,0)  + F0(1,1)*F1(2,2)*F1(0,0)  - 3*F1(1,1)*F1(2,2)*F1(0,0)
            - F1(0,1)*F1(1,0)*F0(2,2)  + F1(1,1)*F1(0,0)*F0(2,2)
            )

            1 *
            (
             - F1(0,2)*F1(1,1)*F1(2,0) + F1(0,1)*F1(1,2)*F1(2,0) + F1(0,2)*F1(1,0)*F1(2,1) - F1(0,1)*F1(1,0)*F1(2,2) - F1(1,2)*F1(2,1)*F1(0,0) + F1(1,1)*F1(2,2)*F1(0,0)
            )

         */
    }


};


#endif //SFM_FUNDAMENTALMATRIXFITTER_H
