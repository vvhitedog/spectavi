"""This example composes several key steps into a pipeline that can estimate
the Essential matrix between a pair of images, given an estimate of the
intrinsics of the camera (that took those images.)

There are three main steps:
1. compute key points for finding correspondences
2. estimate tentative correspondences between image pair
3. robustly estimate the essential matrix between image pair

Each of these three main steps can be run, with this script stopping at any
point and visualizing the results. This is meant to be an instructuve
example.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from util import imread, Timer
from spectavi.feature import sift_filter, nn_bruteforcel1k2
import argparse
import multiprocessing


def normalize_to_ubyte(x):
    """Simple range normalization to an unsigned byte."""
    xrows, dim = x.shape
    new_dim = int(np.ceil(dim / 16.)*16)
    xx = np.zeros([xrows, new_dim])
    xx[:, :dim] = x
    x = xx
    col_mins = np.min(x, axis=0)
    col_maxs = np.max(x, axis=0)
    col_range = col_maxs - col_mins
    col_range[col_range == 0] = 1  # avoid division by zero
    return (255. * (x - col_mins.reshape(1, -1)) / col_range.reshape(1, -1)).astype('uint8')


def step1_sift_detect(args):
    """Run SIFT key-point detection and descriptors on images."""
    ims = [imread(image_filename, force_grayscale=True)
           for image_filename in args.images]
    with Timer('step1-computation'):
        siftkps = [sift_filter(im) for im in ims]
    # Begin Visualize
    c_im = np.hstack(ims)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(c_im, cmap='gray', interpolation='nearest')
    x0, y0 = siftkps[0][:, :2].T
    x1, y1 = siftkps[1][:, :2].T
    shift = ims[0].shape[1]
    ax.plot(x0, y0, 'rx', markersize=1)
    ax.plot(x1+shift, y1, 'bx', markersize=1)
    ax.autoscale()
    ax.set_title('Step1: SIFT Keypoints Detected')
    # End Visualize
    return siftkps


def step2_match_keypoints(args, step1_out):
    """Using output of step1, find likely matches."""
    x, y = step1_out
    with Timer('step2-computation'):
        _x = normalize_to_ubyte(x)
        _y = normalize_to_ubyte(y)
        nn_idx, nn_dist = nn_bruteforcel1k2(
            _x, _y, nthreads=multiprocessing.cpu_count())
    ratio = nn_dist[:, 1] / nn_dist[:, 0].astype('float64')
    pass_idx = ratio >= args.min_ratio
    idx0, _ = nn_idx.T
    xd = x[idx0[pass_idx]]
    yd = y[pass_idx]
    # Begin Visualize
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im0, im1 = imread(args.images[0]), imread(args.images[1])
    c_im = np.hstack([im0, im1])
    ax.imshow(c_im, cmap='gray', interpolation='nearest')
    x0, y0 = xd[:, :2].T
    x1, y1 = yd[:, :2].T
    shift = im0.shape[1]
    x1 = x1.copy() + shift
    # plot points
    ax.plot(x0, y0, 'rx', markersize=3)
    ax.plot(x1, y1, 'bx', markersize=3)
    lines = np.asarray(zip(zip(x0, y0), zip(x1, y1)))
    # randomize line colors
    rand_idx = np.random.randint(lines.shape[0], size=int(
        lines.shape[0]*args.percent_to_show))
    lines = lines[rand_idx]
    lc = mc.LineCollection(lines, cmap=plt.cm.gist_ncar, linewidths=1)
    lc.set_array(np.random.random(lines.shape[0]))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_title('Step2: Match SIFT Keypoints')
    # End Visualize
    return xd, yd


def run(args):
    step1_out = step1_sift_detect(args)
    step2_out = step2_match_keypoints(args, step1_out)
    plt.show(block=True)


example_text = '''example:
    python ex01_essential_estimation.py ../data/castle/01.jpg ../data/castle/02.jpg
'''

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Pipeline to estimate essential matrix'
                                     ' between image pair',
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', metavar='IM', type=str, nargs=2,
                        help='images to compute SIFT')
    parser.add_argument('--min_ratio', default=2., type=float, action='store',
                        help='min-ratio of second min distance to min distance that is accepted (default=2.)')
    parser.add_argument('--percent_to_show', default=.1, type=float, action='store',
                        help='percent of matches to show (for legibility) (default=.1)')
    _args = parser.parse_args()
    run(_args)
