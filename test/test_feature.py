from unittest import TestCase
from nose.plugins.attrib import attr
import numpy as np
from spectavi import feature
import os

np.random.seed(0xdeadbeef)


def brute_force_nn_batched(x, y, k=2):
    """Bruteforce nearest neighbour computation."""
    bs = 1000
    res = list()
    yrows = y.shape[0]
    dim = y.shape[1]
    for i in range(0, yrows, bs):
        dist = np.sum(np.square(x.reshape(-1, 1, dim) -
                                y[i:i + bs].reshape(1, -1, dim)), axis=-1)
        gt_nni = np.argsort(dist, axis=0)[:k].T
        res.append(gt_nni)
    return np.vstack(res)


@attr(speed='fast')
class FeatureTests(TestCase):

    def sift_comparison_test(self):
        """
        Compute SIFT features using spectavi, and compare to sift features
        pre-computed using vlfeat's binary implementation. Ensure that the two
        match.
        """
        # Load pre-computed SIFT features
        oneup = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        datapath = os.path.join(oneup, 'data', 'sift-test', 'sur-ogre.sift')
        precomputed_sf = np.loadtxt(datapath)
        # Compute SIFT features using spectavi
        impath = os.path.join(oneup, 'data', 'sift-test', 'sur-ogre.npz')
        im = np.load(impath)['im']
        sf = feature.sift_filter(im)
        # Do the comparison
        self.assertTrue(np.allclose(sf, precomputed_sf))

    def ann_hnswlib_test(self):
        """
        Compute nearest neighbour matchings using ANN hnswlib library. Ensure
        that error is capped at about 30%, given the randomly generated data.
        """
        xrows = 1000
        dim = 132
        yrows = 1000
        k = 2
        x = np.random.randn(xrows, dim).astype('float32')
        y = np.random.randn(yrows, dim).astype('float32')
        nni = feature.ann_hnswlib(x, y, k)
        gt_nni = brute_force_nn_batched(x, y, k)
        max_diff_count = np.sum(
            np.abs(gt_nni - nni) > 0)
        allowed_diff = k * round(.3 * yrows)
        self.assertLessEqual(max_diff_count, allowed_diff)

    def nn_bruteforce_test(self):
        """
        Compute nearest neighbour matchings using nn_bruteforce.
        """
        xrows = 1000
        dim = 132
        yrows = 1000
        k = 2
        x = np.random.randn(xrows, dim).astype('float32')
        y = np.random.randn(yrows, dim).astype('float32')
        nni, nnd = feature.nn_bruteforce(x, y, k=k, p=2., mu=0)
        gt_nni = brute_force_nn_batched(x, y, k)
        max_diff_count = np.sum(
            np.abs(gt_nni - nni) > 0)
        self.assertLessEqual(max_diff_count, 0)

    def kmedians_test(self):
        """
        Compute that kmedians used for Nearest Neigbours gets somewhat
        sensible results. Overall performance is very disappointing.
        """
        xrows = 500
        dim = 132
        yrows = xrows
        nn_k = 2
        x = np.random.randn(xrows, dim).astype('float32')
        y = np.random.randn(yrows, dim).astype('float32')
        y = x.copy()
        c = 30  # number of cluster we search
        nni, _ = feature.nn_kmedians(x, y, nn_k, c)
        nni_bf, _ = feature.nn_bruteforce(x, y, k=nn_k, p=1., mu=0)
        max_diff_count = np.sum(np.abs(nni - nni_bf) > 0)
        allowed_diff = 2*round(.4 * yrows)
        self.assertLessEqual(max_diff_count, allowed_diff)
