"""
``spectavi.feature``
================
The feature detection/descriptor library of the spectavi.
"""

from spectavi.__libspectavi import clib
from cndarray.ndarray import NdArray
import ctypes as ct
from numpy.ctypeslib import ndpointer


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
        describing the `k` nearest neighbours for each entry of `y` in descending order.

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
"""
