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
from spectavi.mvg import ransac_fitter, dlt_triangulate, image_pair_rectification
import argparse
import multiprocessing


def write_ply(plyfile, data, rgb=None):
    """Write a basic ply file using an ndarray of 3d points."""
    with open(plyfile, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % data.shape[0])
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if rgb is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')
        if rgb is None:
            for p in data:
                f.write('%f %f %f\n' % (p[0], p[1], p[2]))
        else:
            for p, c in zip(data, rgb):
                f.write('%f %f %f %d %d %d\n' %
                        (p[0], p[1], p[2], c[0], c[1], c[2]))


def homogeneous(x):
    """Transforms a matrix of 2d points to 3d homogenous coordinates."""
    return np.hstack((x, np.ones((x.shape[0], 1))))


def normalize_to_ubyte(x):
    """Simple range normalization to an unsigned byte."""
    xrows, dim = x.shape
    new_dim = int(np.ceil(dim / 16.) * 16)
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
    ax.plot(x1 + shift, y1, 'bx', markersize=1)
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
        lines.shape[0] * args.percent_to_show))
    lines = lines[rand_idx]
    lc = mc.LineCollection(lines, cmap=plt.cm.gist_ncar, linewidths=1)
    lc.set_array(np.random.random(lines.shape[0]))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_title('Step2: Match SIFT Keypoints')
    # End Visualize
    return xd, yd


def step3_estimate_essential_matrix(args, step2_out):
    """Estimate an essential matrix using a robust algorithm (RANSAC) with
    matched keypoints."""
    xd, yd = step2_out
    K = np.loadtxt(fname=args.K)
    iK = np.linalg.inv(K)
    x0 = np.dot(homogeneous(xd[..., :2]), iK.T)
    x1 = np.dot(homogeneous(yd[..., :2]), iK.T)
    with Timer('step3-computation'):
        ransac_options = {'required_percent_inliers': .7,
                          'reprojection_error_allowed': 4e-4,
                          'maximum_tries': 50000,
                          'find_best_even_in_failure': True,
                          'singular_value_ratio_allowed': 1e-2,
                          'progressbar': True}
        ransac = ransac_fitter(x0, x1, options=ransac_options)
    # assert ransac['success']
    rE = ransac['essential']
    print (' Percent of inliers: ', ransac['inlier_percent'])
    _, s, _ = np.linalg.svd(rE)
    rE = rE / s[0]
    print (' Fundamental Matrix Singular Values: ', s)
    print (' Singular Values ratio score: ',
           np.abs(s[0] - s[1]) / np.abs(s[0] + s[1]))
    return ransac, x0, x1


def step4_traingulate_points(args, step3_out):
    """Triangulate the points detected as inliers from the previous step."""
    ransac, x0, x1 = step3_out
    idx = ransac['inlier_idx']
    P1 = ransac['camera']
    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    with Timer('step4-computation'):
        RX = dlt_triangulate(P0, P1, x0[idx], x1[idx])
    RX = RX[..., :] / RX[..., -1].reshape(-1, 1)
    write_ply("ex.ply", RX)
    return RX, ransac


def step5_rectify_images(args, step4_out):
    """Rectify images based on RANSAC fit of essential matrix."""
    _, ransac = step4_out
    P1 = ransac['camera']
    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    K = np.loadtxt(fname=args.K)
    P1 = np.dot(K, P1)
    P0 = np.dot(K, P0)
    im0, im1 = imread(args.images[0]), imread(args.images[1])
    rsf = 1.
    with Timer('step5-computation'):
        r0, r1, ri0, ri1 = image_pair_rectification(
            P0, P1, im0, im1, sampling_factor=rsf)
    plt.imsave("im0-rect.png", r0)
    plt.imsave("im1-rect.png", r1)
    plt.imsave("im0-idx-rect.png", ri0)
    plt.imsave("im1-idx-rect.png", ri1)


def run(args):
    step1_out = step1_sift_detect(args)
    step2_out = step2_match_keypoints(args, step1_out)
    step3_out = step3_estimate_essential_matrix(args, step2_out)
    step4_out = step4_traingulate_points(args, step3_out)
    step5_rectify_images(args, step4_out)
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
    parser.add_argument('K', metavar='K', type=str,
                        help='intrinsics for camera (assumption is one camera taking two images')
    parser.add_argument('--min_ratio', default=2., type=float, action='store',
                        help='min-ratio of second min distance to min distance that is accepted (default=2.)')
    parser.add_argument('--percent_to_show', default=.1, type=float, action='store',
                        help='percent of matches to show (for legibility) (default=.1)')
    _args = parser.parse_args()
    run(_args)
