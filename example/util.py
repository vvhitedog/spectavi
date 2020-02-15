import numpy as np
from matplotlib import image as mimage
from time import time


class Timer(object):
    """A simple timer context-manager, taken from 
    https://blog.usejournal.com/how-to-create-your-own-timing-context-manager-in-python-a0e944b48cf8
    """

    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = time()

    def __exit__(self, type, value, traceback):
        self.end = time()
        print("{desc}: {time}s".format(
            desc=self.description, time=(self.end - self.start)))


def rgb_to_gray(rgb):
    """Convert color image to grayscale.

    Parameters
    ----------
    rgb : ndarray
        Three dimensional array, last dimension being at least 3 in size.

    Returns
    -------
    gray: ndarray
        Grayscale image.
    """
    if len(rgb.shape) < 3:
        return rgb.squeeze()
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def imread(filename, dtype='float64', force_grayscale=False):
    """Read a file from disk.

    Parameters
    ----------
    filename : str
        Filename on disk.
    dtype : str, optional
        Data-type of returned array, by default 'float64'
    force_grayscale : bool, optional
        If true, a grayscale image is returned only works if input is rgb, by default False

    Returns
    -------
    im: ndarray
        Loaded image.
    """
    im = mimage.imread(filename)
    if force_grayscale:
        im = rgb_to_gray(im)
    im = im.astype(dtype)
    if dtype == 'float32' or dtype == 'float64':
        im /= np.max(im)
    return im


def read_txt_matrix(txtf, header=False):
    """
    Reads an matrix encoded in ASCII into memory as numpy matrix.
    """
    return np.asarray([map(float, line.strip().split()) for iline, line in
                       enumerate(open(txtf, 'r').readlines()) if (iline > 0 or not header)])
