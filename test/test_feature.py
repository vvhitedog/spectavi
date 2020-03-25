from unittest import TestCase
from nose.plugins.attrib import attr
import numpy as np
from spectavi import feature
import os

np.random.seed(0xdeadbeef)


def brute_force_nn_batched(x, y, k=2, p=2, get_dist=False):
    """Bruteforce nearest neighbour computation."""
    p_norm = np.abs if p == 1 else np.square
    bs = 1000
    res = list()
    resd = list()
    yrows = y.shape[0]
    dim = y.shape[1]
    for i in range(0, yrows, bs):
        dist = np.sum(p_norm(x.reshape(-1, 1, dim) -
                             y[i:i + bs].reshape(1, -1, dim)), axis=-1)
        gt_nni = np.argsort(dist, axis=0)[:k].T
        res.append(gt_nni)
        if get_dist:
            gt_nnd = np.sort(dist, axis=0)[:k].T
            resd.append(gt_nnd)
    return np.vstack(res) if not get_dist else [np.vstack(res), np.vstack(resd)]


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
        im = np.load(impath)['im'].astype('float32')
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

    def nn_bruteforcel1k2_test(self):
        """
        Compute nearest neighbour matchings using nn_bruteforcel1k2. Basic
        benchmarking shows its about 10x faster than other brute-force
        approaches.
        """
        xrows = 200
        dim = 144
        yrows = 200
        k = 2
        x = np.random.uniform(
            low=0, high=256, size=(xrows, dim)).astype('uint8')
        y = np.random.uniform(
            low=0, high=256, size=(yrows, dim)).astype('uint8')
        _, nnd = feature.nn_bruteforcel1k2(x, y)
        _, gt_nnd = brute_force_nn_batched(
            x.astype('int32'), y.astype('int32'), k, p=1, get_dist=True)
        max_diff_count = np.sum(
            np.abs(gt_nnd - nnd) > 0)
        self.assertLessEqual(max_diff_count, 0)

    def nn_cascading_hash_test(self):
        """
        Smoke test for cascading hash
        """
        xrows = 10000
        dim = 144
        yrows = 1000
        k = 2
        #x = np.random.randn(xrows, dim).astype('float32')
        oneup = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        datapath = os.path.join(oneup, 'data', 'sift-test', 'sur-ogre.sift')
        precomputed_sf = np.loadtxt(datapath)
        x = precomputed_sf.astype('float32')
        xrows = x.shape[0]
        x = x - np.mean(x,axis=0,keepdims=True)
        x = x / np.std(x,axis=0,keepdims=True)
        drange = np.max(x,axis=0,keepdims=True) - np.min(x,axis=0,keepdims=True)
        #x = np.round((x - np.min(x,axis=0,keepdims=True)) / drange)
        x0 = np.round((x- np.min(x,axis=0,keepdims=True)) / drange * 128 )
        x0 = x0 - np.mean(x0,axis=0,keepdims=True)
        x1 = np.round((x) / drange * 127 )
        #x0 = x0 - 128
        x = x0
        print np.max(x0,axis=0), np.min(x0,axis=0)
        print np.max(x1,axis=0), np.min(x1,axis=0)
        _x = np.zeros([xrows,dim],dtype='float32')
        #_x = np.random.uniform(size=[xrows, dim]).astype('float32')
        _x[:,:x.shape[1]] = x

        #_x -= .5
        #_x *= 255
        #_x -= 128
        x = _x
        #y = np.random.randn(yrows, dim).astype('float32')
        y = _x
        nni, nnd = feature.nn_cascading_hash(x, y, k=k)
        print nni
        print nnd
