"""An example showing the matching of SIFT keypoints using ANN.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from util import imread, Timer
from spectavi.feature import sift_filter, ann_hnswlib, nn_bruteforcel1k2
import argparse

example_text = '''example:
    python ex02_sift_match.py ../data/castle/01.jpg ../data/castle/02.jpg --use_l1_norm 
'''


def process_image(image_filename):
    """Process an image file for this example.
    """
    with Timer('sift-process-image'):
        im = imread(image_filename, force_grayscale=True)
        sift_kps = sift_filter(im)
    return sift_kps


def normalize_to_ubyte(x):
    """Simple range normalization to an unsigned byte.
    """
    xrows,dim = x.shape
    new_dim = int(np.ceil(dim / 16.)*16)
    xx = np.zeros([xrows,new_dim])
    xx[:,:dim] = x
    x = xx
    col_mins = np.min(x, axis=0)
    col_maxs = np.max(x, axis=0)
    col_range = col_maxs - col_mins
    col_range[col_range == 0] = 1 # avoid division by zero
    return (255. * (x - col_mins.reshape(1, -1)) / col_range.reshape(1, -1)).astype('uint8')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='SIFT keypoint matching using approximate nearest'
                                     ' neighbour (ANN) example.',
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', metavar='IM', type=str, nargs=2,
                        help='images to compute SIFT')
    parser.add_argument('--min_ratio', default=2., type=float, action='store',
                        help='min-ratio of second min distance to min distance that is accepted (default=2.)')
    parser.add_argument('--percent_to_show', default=.1, type=float, action='store',
                        help='percent of matches to show (for legibility) (default=.1)')
    parser.add_argument('--use_l1_norm', action='store_true',
                        help='use l1 norm for nearest neighbours (default=l2 norm)')
    args = parser.parse_args()
    # process each image and store
    x, y = [process_image(imagefn) for imagefn in args.images]
    print('1st image # SIFT features: {n1}'.format(n1=x.shape[0]))
    print('2nd image # SIFT features: {n2}'.format(n2=y.shape[0]))
    with Timer('ann-compute'):
        if not args.use_l1_norm:
            # compute for each entry in `y` the index in `x` that is L2 nearest
            nn_idx = ann_hnswlib(x, y)
        else:
            nn_idx, nn_dist = nn_bruteforcel1k2(normalize_to_ubyte(x),
                                                normalize_to_ubyte(y))
    # get ratio of the distance of closest and second closest points
    idx0, idx1 = nn_idx.T
    p_norm = np.abs if args.use_l1_norm else np.square
    dist0 = np.sum(p_norm(x[idx0] - y), axis=-1)
    dist1 = np.sum(p_norm(x[idx1] - y), axis=-1)
    ratio = dist1 / dist0
    # get idx that pass ratio test
    pass_idx = ratio >= args.min_ratio
    xd = x[idx0[pass_idx]]
    yd = y[pass_idx]
    # draw concatenated images (re-read color images)
    im0, im1 = imread(args.images[0]), imread(args.images[1])
    c_im = np.hstack([im0, im1])
    shift = im0.shape[1]
    # draw images
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(c_im, interpolation='nearest')
    # draw sift points
    x0, y0 = xd[:, :2].T
    x1, y1 = yd[:, :2].T
    x1 += shift
    ax.plot(x0, y0, 'rx')
    ax.plot(x1, y1, 'bx')
    # plot the connections (a random subset for legibility)
    lines = np.asarray(zip(zip(x0, y0), zip(x1, y1)))
    # randomize the colors of the lines
    rand_idx = np.random.randint(lines.shape[0], size=int(
        lines.shape[0]*args.percent_to_show))
    lines = lines[rand_idx]
    lc = mc.LineCollection(lines, cmap=plt.cm.gist_ncar, linewidths=1)
    lc.set_array(np.random.random(lines.shape[0]))
    ax.add_collection(lc)
    ax.autoscale()
    # show
    plt.show(block=True)
