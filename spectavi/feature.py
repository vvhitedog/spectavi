"""
``spectavi.feature``
================
The feature detection/descriptor library of the spectavi.
"""

from __libspectavi import clib
from cndarray.ndarray import NdArray
import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np


"""
==================================================================================
sift_filter
==================================================================================
void sift_filter(const double *im, int wid, int hgt, NdArray *out) {
"""

_sift_filter = clib.sift_filter
_sift_filter.restype = None
_sift_filter.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                   ct.c_int,
                                   ct.c_int,
                                   ct.POINTER(NdArray),]


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
    sift_ret = NdArray()
    hgt,wid = im.shape
    _sift_filter(im,wid,hgt,sift_ret)
    return sift_ret.asarray()


"""
==================================================================================
"""
