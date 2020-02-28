"""
``spectavi.feature``
================
The feature detection/descriptor library of the spectavi.
"""

from spectavi.__libspectavi import clib
from cndarray.ndarray import NdArray
import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np


"""
==================================================================================
sift_filter
==================================================================================
"""


_sift_filter = clib.sift_filter
_sift_filter.restype = None
_sift_filter.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                         ct.c_int,
                         ct.c_int,
                         ct.POINTER(NdArray), ]


def sift_filter(im):
    """
    Detect SIFT features and compute descriptors for a given image `im`.

    Parameters
    ----------
    im : ndarray
        A (single channel/grayscale) image (2d).

    Returns
    -------
    kps : float ndarray
        A matrix of size `nkp` by 132 which contains the SIFT descriptor for
        each keypoint, where `nkp` is the number of keypoints found.

    """
    if len(im.shape) != 2:
        raise TypeError("Only 2d images are supported.")
    sift_ret = NdArray(dtype='float32')
    hgt, wid = im.shape
    _sift_filter(im, wid, hgt, sift_ret)
    return sift_ret.asarray()


"""
==================================================================================
ann_hnswlib
==================================================================================
"""

_ann_hnswlib = clib.ann_hnswlib
_ann_hnswlib.restype = None
_ann_hnswlib.argtypes = [ndpointer(ct.c_float, flags="C_CONTIGUOUS"),
                         ndpointer(ct.c_float, flags="C_CONTIGUOUS"),
                         ct.c_int,
                         ct.c_int,
                         ct.c_int,
                         ct.c_int,
                         ct.POINTER(NdArray), ]


def ann_hnswlib(x, y, k=2):
    """
    Run the Approximate Nearest Neighbour (ann) algorithm (using the NSWlib implementation.)

    Parameters
    ----------
    x : float32 ndarray
        The database to query against.
    y : float32 ndarray
        The query.
    k : int, optional
        The amount of nearest neighbours to calculate.

    Returns
    -------
    ann: uint64 ndarray
        The array of size `y.shape[0]` by `k` which is the index into `x`
        describing the `k` nearest neighbours for each entry of `y` in
        ascending order of distance.

    """
    xrows, xdim = x.shape
    yrows, ydim = y.shape
    assert ydim == xdim
    dim = xdim
    ann_ret = NdArray(dtype='uint64')
    _ann_hnswlib(x, y, xrows, yrows, dim, k, ann_ret)
    return ann_ret.asarray()


"""
==================================================================================
nn_bruteforce
==================================================================================
"""

_nn_bruteforce = clib.nn_bruteforce
_nn_bruteforce.restype = None
_nn_bruteforce.argtypes = [ndpointer(ct.c_float, flags="C_CONTIGUOUS"),
                           ndpointer(ct.c_float, flags="C_CONTIGUOUS"),
                           ct.c_int,
                           ct.c_int,
                           ct.c_int,
                           ct.c_int,
                           ct.c_float,
                           ct.c_float,
                           ct.POINTER(NdArray),
                           ct.POINTER(NdArray), ]

_nn_bruteforcei = clib.nn_bruteforcei
_nn_bruteforcei.restype = None
_nn_bruteforcei.argtypes = [ndpointer(ct.c_int, flags="C_CONTIGUOUS"),
                            ndpointer(ct.c_int, flags="C_CONTIGUOUS"),
                            ct.c_int,
                            ct.c_int,
                            ct.c_int,
                            ct.c_int,
                            ct.c_float,
                            ct.c_float,
                            ct.POINTER(NdArray),
                            ct.POINTER(NdArray), ]

_nn_bruteforcel1k2 = clib.nn_bruteforcel1k2
_nn_bruteforcel1k2.restype = None
_nn_bruteforcel1k2.argtypes = [ndpointer(ct.c_ubyte, flags="C_CONTIGUOUS"),
                               ndpointer(ct.c_ubyte, flags="C_CONTIGUOUS"),
                               ct.c_int,
                               ct.c_int,
                               ct.c_int,
                               ct.c_int,
                               ct.POINTER(NdArray),
                               ct.POINTER(NdArray), ]


def nn_bruteforce(x, y, p=.5, mu=0., k=2, use_int=False):
    """
    Nearest Neighbour algorithm using a pseudo-bruteforce technique.

    Can be used to compute nearest neighbours for any p-norm.

    Parameters
    ----------
    x : float32 ndarray
        The database to query against.
    y : float32 ndarray
        The query.
    p : float, optional
        p-value of norm.
    mu : float, optional
        Approximation value, when `mu`=0 computation is exact.
    k : int, optional
        The amount of nearest neighbours to calculate.

    Returns
    -------
    nn_idx: uint64 ndarray
        The array of size `y.shape[0]` by `k` which is the index into `x`
        describing the `k` nearest neighbours for each entry of `y` in
        ascending order of distance.
    nn_idx: float32 ndarray
        The array of size `y.shape[0]` by `k` which is the distance of the
        nearest neighbours in ascening order.

    """
    xrows, xdim = x.shape
    yrows, ydim = y.shape
    assert ydim == xdim
    dim = xdim
    nn_idx = NdArray(dtype='uint64')
    if not use_int:
        nn_dist = NdArray(dtype='float32')
        _nn_bruteforce(x, y, xrows, yrows, dim, k, p, mu, nn_idx, nn_dist)
    else:
        nn_dist = NdArray(dtype='int32')
        xi = np.round(100 * x).astype('int32')
        yi = np.round(100 * y).astype('int32')
        _nn_bruteforcei(xi, yi, xrows, yrows, dim, k, p, mu, nn_idx, nn_dist)
    return nn_idx.asarray(), nn_dist.asarray()


def nn_bruteforcel1k2(x, y, nthreads=1):
    """
    Highly optimized L1 NN with k=2, and inputs must be 8-unsigned bytes
    aligned to 16-byte boundaries.
    """
    xrows, xdim = x.shape
    yrows, ydim = y.shape
    assert ydim == xdim
    dim = xdim
    nn_idx = NdArray(dtype='uint64')
    nn_dist = NdArray(dtype='int32')
    _nn_bruteforcel1k2(x, y, xrows, yrows, dim, nthreads, nn_idx, nn_dist)
    return nn_idx.asarray(), nn_dist.asarray()


"""
==================================================================================
kmedians
==================================================================================
"""

_nn_kmedians = clib.nn_kmedians
_nn_kmedians.restype = None
_nn_kmedians.argtypes = [ndpointer(ct.c_float, flags="C_CONTIGUOUS"),
                         ndpointer(ct.c_float, flags="C_CONTIGUOUS"),
                         ct.c_int,
                         ct.c_int,
                         ct.c_int,
                         ct.c_int,
                         ct.c_int,
                         ct.c_int,
                         ct.c_int,
                         ct.POINTER(NdArray),
                         ct.POINTER(NdArray), ]


def nn_kmedians(x, y, k, c=5):
    xrows, dim = x.shape
    yrows, ydim = y.shape
    nmx = int(np.round(np.sqrt(xrows / c) * c))
    nmy = int(np.round(np.sqrt(yrows / c) * c))
    assert ydim == dim
    nn_idx = NdArray(dtype='uint64')
    nn_dist = NdArray(dtype='float32')
    _nn_kmedians(x, y, xrows, yrows, dim, nmx, nmy, c, k, nn_idx, nn_dist)
    return nn_idx.asarray(), nn_dist.asarray()


"""
==================================================================================
"""
