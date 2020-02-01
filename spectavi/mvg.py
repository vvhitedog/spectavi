"""
``spectavi.mvg``
================
The muti-view-geometry library of the spectavi project.
"""

from __libspectavi import clib
from cndarray.ndarray import NdArray
import ctypes as ct
from numpy.ctypeslib import ndpointer
import numpy as np



def hnormalize(x):
    """
    Normalization points from euclidean to homogenous.
    """
    return x[..., :-1] / np.expand_dims(x[..., -1], axis=-1)


"""
==================================================================================
image_pair_rectification 
==================================================================================
"""

_image_pair_rectification = clib.image_pair_rectification
_image_pair_rectification.restype = None
_image_pair_rectification.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                      ndpointer(
                                          ct.c_double, flags="C_CONTIGUOUS"),
                                      ndpointer(
                                          ct.c_double, flags="C_CONTIGUOUS"),
                                      ndpointer(
                                          ct.c_double, flags="C_CONTIGUOUS"),
                                      ct.c_int,
                                      ct.c_int,
                                      ct.c_int,
                                      ct.c_double,
                                      ct.POINTER(NdArray),
                                      ct.POINTER(NdArray),
                                      ct.POINTER(NdArray),
                                      ct.POINTER(NdArray),
                                      ]


def image_pair_rectification(P0, P1, im0, im1, sampling_factor=1.2, crop_invalid=True):
    """
    Rectifying an image pair given their two camera matrices.

    Parameters
    ----------
    P0 : ndarray
        First (left) input image's camera matrix.
    P1 : ndarray
        Second (right) input image's camera matrix.
    im0 : ndarray
        First (left) input image.
    im1 : ndarray
        Second (right) input image.
    sampling_factor : float, optional
        The oversampling ratio to use per line when rectifying.
    crop_invalid : bool, optional
        If true, output images are cropped to valid data only.

    Returns
    -------
    r0 : ndarray
        Rectified (left) first image.
    r1 : ndarray
        Rectified (right) second image.
    r0i : int ndarray
        Rectified (left) first image's coordinates used in resampling.
    r1i : int ndarray
        Rectified (right) second image's coordinates used in resampling.

    Notes
    -----
    The coordinate matricies returned give a direct mapping between the
    rectified output and the original images. This is clearly important for any
    work that is done downstream with the rectified images.


    """
    if np.any(im0.shape != im1.shape):
        raise TypeError("Input images must have same size.")
    if len(im0.shape) == 2:
        nchan = 1
        hgt, wid = im0.shape
    else:
        hgt, wid, nchan = im0.shape
    r0 = NdArray()
    r1 = NdArray()
    ri0 = NdArray(dtype='int32')
    ri1 = NdArray(dtype='int32')
    _image_pair_rectification(P0, P1, im0, im1, wid, hgt, nchan, sampling_factor,
                              ct.byref(r0), ct.byref(r1), ct.byref(ri0), ct.byref(ri1))
    r0 = r0.asarray()
    r1 = r1.asarray()
    ri0 = ri0.asarray()
    ri1 = ri1.asarray()
    if crop_invalid:
        idx = (ri0 != -1) | (ri1 != -1)
        y, x = np.where(idx)
        lowy, highy = np.min(y), np.max(y)
        lowx, highx = np.min(x), np.max(x)
        r0 = r0[lowy:highy + 1, lowx:highx + 1, ...]
        r1 = r1[lowy:highy + 1, lowx:highx + 1, ...]
        ri1 = ri1[lowy:highy + 1, lowx:highx + 1]
        ri0 = ri0[lowy:highy + 1, lowx:highx + 1]
    return r0, r1, ri0, ri1


"""
==================================================================================
ransac_fitter 
==================================================================================
"""

_ransac_fitter = clib.ransac_fitter
_ransac_fitter.restype = None
_ransac_fitter.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                           ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                           ct.c_int,
                           ct.c_double,
                           ct.c_double,
                           ct.c_int,
                           ct.c_bool,
                           ct.c_double,
                           ct.c_bool,
                           ct.POINTER(ct.c_bool),
                           ct.POINTER(NdArray),
                           ct.POINTER(NdArray),
                           ct.POINTER(ct.c_double),
                           ct.POINTER(NdArray), ]


def ransac_fitter(x0, x1, options={'required_percent_inliers': .9,
                                   'reprojection_error_allowed': .5,
                                   'maximum_tries': 500,
                                   'find_best_even_in_failure': True,
                                   'singular_value_ratio_allowed': 3e-2,
                                   'progressbar': False}):
    """
    Fit a two view geometery to two sets of tentatively corresponding points.

    Parameters
    ----------
    x0 : ndarray
        The first set of points in homogenous (?) normalized coordinates.
    x1 : ndarray
        The second set of points in homogenous (?) normalized coordinates.
    options : dict, optional
        A set of parameters of the RANSAC process listed in detail below.
    required_percent_inliers: float, optional
        The minimum percent of required inliers for RANSAC to be considered a
        success.
    reprojection_error_allowed: float, optional
        The allowed error in reprojection for point match to be considered an
        inlier of the model.
    maximum_tries: int, optional
        The number of trials RANSAC is allowed to run for.
    find_best_even_in_failure: bool, optional
        In case of failure, return the best model.
    singular_value_ratio_allowed: float, optional
        The required ratio of the difference of the first two singular values
        to the sum of them. This is to enforce fitting an essential matrix.
    progressbar: bool, optional
        If true, a progressbar is drawn to stdout.

    Returns
    -------
    ret: dict
        A dictionary with the following entries.
    success : bool
        True RANSAC found a viable solution given parameters.
    essential : ndarray
        The essential matrix corresponding to the two sets of points.
    camera : ndarray
        A camera matrix for the second set of points (the first camera can be
        assumed to be the identity.)
    inlier_percent: float
        Percent of points that were deemed inliers of this model.
    inlier_idx: float

    """

    success = ct.c_bool()
    essential = NdArray()
    camera = NdArray()
    inlier_idx = NdArray(dtype='int32')
    inlier_percent = ct.c_double()

    npt = x0.shape[0]
    assert (x0.shape[0] == x1.shape[0])
    _ransac_fitter(x0, x1, npt, options['required_percent_inliers'],
                   options['reprojection_error_allowed'],
                   options['maximum_tries'],
                   options['find_best_even_in_failure'],
                   options['singular_value_ratio_allowed'],
                   options['progressbar'],
                   ct.byref(success),
                   ct.byref(essential),
                   ct.byref(camera),
                   ct.byref(inlier_percent),
                   ct.byref(inlier_idx))

    success = success.value
    essential = essential.asarray()
    camera = camera.asarray()
    inlier_percent = inlier_percent.value
    inlier_idx = inlier_idx.asarray()

    ret = {'success': success,
           'essential': essential,
           'camera': camera,
           'inlier_percent': inlier_percent,
           'inlier_idx': inlier_idx, }

    return ret


"""
==================================================================================
seven_point_algorithm 
==================================================================================
"""

_seven_point_algorithm = clib.seven_point_algorithm
_seven_point_algorithm.restype = None
_seven_point_algorithm.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                   ndpointer(
                                       ct.c_double, flags="C_CONTIGUOUS"),
                                   ct.POINTER(ct.c_int),
                                   ndpointer(ct.c_double, flags="C_CONTIGUOUS")]


def seven_point_algorithm(x, xp):
    if not (x.shape[0] == 7 and xp.shape[0] == 7):
        raise TypeError('Must be 7 points.')
    if not (x.shape[1] == 2 and xp.shape[1] == 2):
        # raise TypeError('Coords must be euclidean not homogenous.')
        x, xp = hnormalize(x), hnormalize(xp)
    dst = np.empty((3, 3, 3))
    nroot = ct.c_int()
    _seven_point_algorithm(x, xp, ct.byref(nroot), dst)
    nroot = nroot.value
    return np.vstack(dst[:nroot])


"""
==================================================================================
dlt_triangulate 
==================================================================================
"""


_dlt_triangulate = clib.dlt_triangulate
_dlt_triangulate.restype = None
_dlt_triangulate.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                             ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                             ct.c_int,
                             ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                             ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                             ndpointer(ct.c_double, flags="C_CONTIGUOUS"), ]


_dlt_reprojection_error = clib.dlt_reprojection_error
_dlt_reprojection_error.restype = None
_dlt_reprojection_error.argtypes = [ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                    ndpointer(
                                        ct.c_double, flags="C_CONTIGUOUS"),
                                    ct.c_int,
                                    ndpointer(
                                        ct.c_double, flags="C_CONTIGUOUS"),
                                    ndpointer(
                                        ct.c_double, flags="C_CONTIGUOUS"),
                                    ndpointer(ct.c_double, flags="C_CONTIGUOUS"), ]


def dlt_triangulate(P0, P1, x, xp, ret_error=False):
    if not (P0.shape == (3, 4) and P1.shape == (3, 4)):
        raise TypeError('P0,P1 must be camera matrices.')
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    if len(xp.shape) == 1:
        xp = np.expand_dims(xp, axis=0)
    if not (x.shape[0] == xp.shape[0]):
        raise TypeError('Must be same # points or shape.')
    if not (len(x.shape) == 2 and len(xp.shape) == 2):
        raise TypeError('Wrong dimensionality of input.')
    if not (x.shape[1] == 3 and xp.shape[1] == 3):
        raise TypeError('Coords must be homogenous.')
    npt = x.shape[0]
    if ret_error:
        dst = np.empty((npt, 1))
        _dlt_reprojection_error(P0, P1, npt, x, xp, dst)
    else:
        dst = np.empty((npt, 4))
        _dlt_triangulate(P0, P1, npt, x, xp, dst)
    return dst


def dlt_reprojection_error(P0, P1, x, xp):
    return dlt_triangulate(P0, P1, x, xp, ret_error=True)

"""
==================================================================================
"""
