from unittest import TestCase
from nose.plugins.attrib import attr
import numpy as np
from spectavi import mvg

__ntestcases__ = 100

np.random.seed(0xdeadbeef)


def skew_symmetric_matrix(s):
    mat = np.zeros((3, 3))
    mat[0, 1] = -s[2]
    mat[0, 2] = s[1]
    mat[1, 0] = s[2]
    mat[1, 2] = -s[0]
    mat[2, 0] = -s[1]
    mat[2, 1] = s[0]
    return mat


@attr(speed='fast')
class MultiViewGeometryTests(TestCase):

    def test_ransac_essential_reconstruction(self):
        """
        Simulate a essential matrix and attempt to reconstruct it.
        """

        C0 = (np.random.randn(3) + 1.) * 50.
        C1 = (np.random.randn(3) - 1.) * 50.

        def rot(a, b):
            v = np.cross(a, b)
            sk = skew_symmetric_matrix(v)
            return np.eye(3) + sk + np.dot(sk, sk) * 1. / (1+np.dot(a, b))

        def normalize(x):
            return x / np.linalg.norm(x)

        canon = np.asarray([1., 0., 0.])
        R0 = rot(canon, normalize(-C0))
        R1 = rot(canon, normalize(-C1))

        P0 = np.hstack((R0, np.dot(R0, -C0).reshape(-1, 1)))
        P1 = np.hstack((R1, np.dot(R1, -C1).reshape(-1, 1)))

        npt = 200
        X = np.hstack(((np.random.randn(npt, 3)), np.ones((npt, 1))))
        x0 = np.dot(X, P0.T)
        x1 = np.dot(X, P1.T)

        ransac_options = {'required_percent_inliers': .9,
                          'reprojection_error_allowed': .5,
                          'maximum_tries': 200,
                          'find_best_even_in_failure': False,
                          'singular_value_ratio_allowed': 3e-2,
                          'progressbar': False}
        ransac = mvg.ransac_fitter(x0, x1, options=ransac_options)
        assert ransac['success']
        rE = ransac['essential']

        _, s, _ = np.linalg.svd(rE)
        rE = rE / s[0]

        e = np.dot(P1, np.hstack((C0, (1.,))))
        invP0 = np.dot(P0.T, np.linalg.inv(np.dot(P0, P0.T)))
        E = np.dot(np.dot(skew_symmetric_matrix(e), P1), invP0)
        _, s, _ = np.linalg.svd(E)
        E = E / s[0]

        if np.std(rE / E) >= 1e-2:
            print 'Test failed, debug info:'
            print np.std(rE / E)
            print E
            print rE
            print rE / E

        assert np.std(rE / E) < 1e-2

    def test_dlt_reprojection_error(self):
        """
        Simulates and tests that reprojection error is zero on perfect case.
        """
        for _ in range(__ntestcases__):
            P0 = np.random.randn(3, 4)
            P1 = np.random.randn(3, 4)
            X0 = np.random.randn(4)
            x0 = np.dot(P0, X0)
            x1 = np.dot(P1, X0)
            err = mvg.dlt_reprojection_error(P0, P1, x0, x1)
            assert abs(err) < 1e-3

    def test_dlt_post_conditions(self):
        """
        Simulates and tests that simulated results match; checks for cross
        product condition at the end.
        """
        for _ in range(__ntestcases__):
            P0 = np.random.randn(3, 4)
            P1 = np.random.randn(3, 4)
            X0 = np.random.randn(4)
            x0 = np.dot(P0, X0)
            x1 = np.dot(P1, X0)
            X = mvg.dlt_triangulate(P0, P1, x0, x1).ravel()
            X /= X[3]
            X0 /= X0[3]
            assert np.allclose(X, X0)
            rx0 = np.dot(P0, X)
            rx1 = np.dot(P1, X)
            assert np.allclose(np.cross(rx0, x0), np.zeros(3))
            assert np.allclose(np.cross(rx1, x1), np.zeros(3))

    def test_seven_point_algorithm_conditions(self):
        """
        Tests that the seven point algorithm always produces Fundamental
        Matrices which result in a perfect fit.
        """
        for _ in range(__ntestcases__):
            x0 = np.random.randn(7, 3)
            x1 = np.random.randn(7, 3)
            FF = mvg.seven_point_algorithm(x0, x1)
            assert FF.shape[0] % 3 == 0
            nF = FF.shape[0]/3
            for i in range(nF):
                F = FF[3*i:3*(i+1)]
                xpTFx = np.sum(np.dot(x1, F) * x0, axis=1)
                assert np.max(np.abs(xpTFx)) < 1e-10

    def test_seven_point_algorithm_reconstruction(self):
        """
        Tests that the seven point algorithm always finds a simulated
        Fundamental Matrix.
        """
        for _ in range(__ntestcases__):
            P0 = np.hstack((np.eye(3), np.zeros(3).reshape(-1, 1)))
            P1 = np.random.randn(3, 4)
            e = P1.T[-1]  # NOTE: This works only in the canonical case!
            invP0 = np.dot(P0.T, np.linalg.inv(np.dot(P0, P0.T)))
            F0 = np.dot(np.dot(skew_symmetric_matrix(e), P1), invP0)
            X = np.random.randn(7, 4)
            x0 = np.dot(X, P0.T)
            x1 = np.dot(X, P1.T)
            FF = mvg.seven_point_algorithm(x0, x1)
            assert FF.shape[0] % 3 == 0
            nF = FF.shape[0]/3
            assert any([np.std(FF[3*i:3*(i+1)]/F0) < 1e-8 for i in range(nF)])
