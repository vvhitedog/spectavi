"""This example composes several key steps into a pipeline that can estimate
the Essential matrix between a pair of images, given an estimate of the
intrinsics of the camera (that took those images.) The estimated essential
matrix is used to triangulate a set of seed points (to create a sparse 3D
point cloud,) and to derive a suitable pair of camera matrices (P0,P1) to
perform a rectification step.

There are three main steps and 2 post-processing steps:
1. compute key points for finding correspondences
2. estimate tentative correspondences between image pair
3. robustly estimate the essential matrix between image pair
4. triangulate a set of sparse points deemed inliers to create a sparse 3D
   point cloud
5. use the essential matrix to derive camera matrices and use those to
   rectify the image pair

This scripts showcases the outputs of each of these distinct steps.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from util import imread, Timer
from spectavi.feature import nn_bruteforcel1k2,nn_cascading_hash
from spectavi.feature import normalize_to_ubyte_and_multiple_16_dim
from spectavi.feature import sift_filter, sift_filter_batch, sift_filter_striped
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


def step1_sift_detect(args):
    """Run SIFT key-point detection and descriptors on images."""
    ims = [imread(image_filename, dtype='float32',
            force_grayscale=True)
           for image_filename in args.images]
    with Timer('step1-computation'):
        if args.use_sift_striped:
            siftkps = [ sift_filter_striped(im,
                nthread=args.cpu_count) for im in ims ] 
        else:
            siftkps = sift_filter_batch(ims)
    print ('sift 1 #: ', siftkps[0].shape[0] )
    print ('sift 2 #: ', siftkps[1].shape[0] )
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
    _x = normalize_to_ubyte_and_multiple_16_dim(x)
    _y = normalize_to_ubyte_and_multiple_16_dim(y)
    with Timer('step2-computation'):
        if args.matching_method == 'bruteforce':
            nn_idx, nn_dist = nn_bruteforcel1k2(
                (_x+128).astype('uint8'),
                (_y+128).astype('uint8'),
                nthreads=args.cpu_count)
        elif args.matching_method == 'cascading-hash':
            nn_idx, nn_dist = nn_cascading_hash(_x, _y)
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
        ransac_quality = {'low': .6, 'medium': .7,
                          'high': .75, 'ultra': .8, 'uber': .9}
        ransac_options = {'required_percent_inliers':
                          ransac_quality[args.ransac_quality],
                          'reprojection_error_allowed': 3.35e-4,
                          'maximum_tries': 10000000,
                          'find_best_even_in_failure': False,
                          'singular_value_ratio_allowed': 1e-3,
                          'progressbar': False}
        ransac = ransac_fitter(x0, x1, options=ransac_options)
    # assert ransac['success']
    rE = ransac['essential']
    print (' Number of keypoints: ', xd.shape[0])
    print (' Percent of inliers: ', ransac['inlier_percent'])
    _, s, _ = np.linalg.svd(rE)
    rE = rE / s[0]
    print (' Fundamental Matrix Singular Values: ', s)
    print (' Singular Values ratio score: ',
           np.abs(s[0] - s[1]) / np.abs(s[0] + s[1]))
    return ransac, x0, x1, xd, yd


def step4_triangulate_points(args, step3_out):
    """Triangulate the points detected as inliers from the previous step."""
    ransac, x0, x1, xd, yd = step3_out
    idx = ransac['inlier_idx']
    P1 = ransac['camera']
    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    with Timer('step4-computation'):
        RX = dlt_triangulate(P0, P1, x0[idx], x1[idx])
    RX = RX[..., :] / RX[..., -1].reshape(-1, 1)
    xy0 = xd[idx, :2].astype('int32')
    xy1 = yd[idx, :2].astype('int32')
    im0, im1 = imread(args.images[0]), imread(args.images[1])
    im0v = im0[xy0[:, 1], xy0[:, 0]]
    im1v = im1[xy1[:, 1], xy1[:, 0]]
    rgb = np.round(255*(im0v + im1v)/2.).astype('uint8')
    write_ply(os.path.join(args.outdir, "sparse_inliers.ply"), RX, rgb=rgb)
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
    with Timer('step5-computation'):
        r0, r1, ri0, ri1 = image_pair_rectification(
            P0, P1, im0, im1, sampling_factor=args.rsf)
    plt.imsave(os.path.join(args.outdir, "rect-" +
                            os.path.basename(args.images[0])), r0)
    plt.imsave(os.path.join(args.outdir, "rect-" +
                            os.path.basename(args.images[1])), r1)
    ri0.tofile(os.path.join(args.outdir, "rect-idx-" +
                            os.path.basename(args.images[0])).split('.')[0]
               + '.bin')
    ri1.tofile(os.path.join(args.outdir, "rect-idx-" +
                            os.path.basename(args.images[1])).split('.')[0]
               + '.bin')


def try_open3d_viz(args):
    """Try to visualize sparse 3d point cloud using open3d"""
    try: 
        from open3d import visualization as viz
        from open3d import io
        ply_file = os.path.join(args.outdir, "sparse_inliers.ply")
        pc = io.read_point_cloud(ply_file)
        viz.draw_geometries([pc])
    except ImportError as err:
        print ("Failed to import `open3d` package, can not visualize"
                " point-cloud, try installing open3d or use meshlab to visualize"
                " ply file.")


def load_cache(args):
    """Loads a cache if it exists."""
    filename = os.path.join(args.outdir, 'cache.npz')
    if os.path.exists(filename):
        data = np.load(filename)
        return data['xd'], data['yd']
    else:
        return None


def save_cache(args, step2_out):
    """Saves a cache."""
    filename = os.path.join(args.outdir, 'cache.npz')
    xd, yd = step2_out
    np.savez_compressed(filename, xd=xd, yd=yd)


def run(args):
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if args.cache:
        cache = load_cache(args)
    else:
        cache = None
    if cache is None:
        step1_out = step1_sift_detect(args)
        step2_out = step2_match_keypoints(args, step1_out)
    else:
        step2_out = cache
    if cache is None and args.cache:
        save_cache(args, step2_out)
    step3_out = step3_estimate_essential_matrix(args, step2_out)
    step4_out = step4_triangulate_points(args, step3_out)
    step5_rectify_images(args, step4_out)
    plt.show(block=True)
    try_open3d_viz(args)


example_text = '''example:
    python ex01_essential_estimation.py ../data/castle/01.jpg ../data/castle/02.jpg ../data/castle/K.txt
'''

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Pipeline to estimate essential matrix'
                                     ' between image pair; later perform'
                                     ' triangulation & rectification',
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', metavar='IM', type=str, nargs=2,
                        help='images to estimate essential matrix')
    parser.add_argument('K', metavar='K', type=str,
                        help='intrinsics for camera (assumption is one camera taking two images')
    parser.add_argument('--min_ratio', default=1.75, type=float, action='store',
                        help='min-ratio of second min distance to min distance that is accepted (default=1.75)')
    parser.add_argument('--percent_to_show', default=.1, type=float, action='store',
                        help='percent of matches to show (for legibility) (default=.1)')
    parser.add_argument('--ransac_quality', default='ultra', choices=['low', 'medium', 'high', 'ultra', 'uber'], action='store',
                        help='quality of ransac fit to perform (default=ultra)')
    parser.add_argument('--matching_method', default='cascading-hash', choices=['bruteforce', 'cascading-hash'], action='store',
                        help='which method to use, bruteforce = brute force matching, cascading-hash = variant'+\
                        ' on cascading hash method (default=cascading-hash)')
    parser.add_argument('--outdir', default='ex01_out', type=str,
                        help='output is placed in this directory (default="ex01_out")')
    parser.add_argument('--rsf', default=1., type=float, action='store',
                        help='resampling factor (along epipolar lines) when performing rectification (default=1.)')
    parser.add_argument('--cache', action='store_true',
                        help='cache the keypoint matches per session, if a cached output exists, execution starts at step 3 (default=False)')
    parser.add_argument('--use_sift_striped', action='store_true',
                        help='use striped version of SIFT keypoint computation, may result in slightly different results, but is more efficient (default=False)')
    parser.add_argument('--cpu_count', default=8, type=int, action='store',
                        help='number of cpus to use for multi-threaded code (default=8)')
    _args = parser.parse_args()
    run(_args)
