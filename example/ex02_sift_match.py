"""An example showing the matching of SIFT keypoints using ANN.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from util import imread, Timer
from spectavi.feature import sift_filter, ann_hnswlib
import argparse

example_text = '''example:
    python ex02_sift_match.py ../data/castle/01.jpg ../data/castle/02.jpg
'''


def process_image(image_filename):
    """Process an image file for this example.
    """
    with Timer('sift-process-image'):
        im = imread(image_filename, force_grayscale=True)
        sift_kps = sift_filter(im)
    return sift_kps


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='SIFT keypoint matching using approximate nearest'
                                     ' neighbour (ANN) example.',
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', metavar='IM', type=str, nargs=2,
                        help='images to compute SIFT')
    parser.add_argument('--min_ratio', default=5., type=float, action='store',
                        help='min-ratio of second min distance to min distance that is accepted (default=5.)')
    parser.add_argument('--percent_to_show', default=.05, type=float, action='store',
                        help='percent of matches to show (for legibility) (default=.2)')
    args = parser.parse_args()
    # process each image and store
    x, y = [process_image(imagefn) for imagefn in args.images]
    print('1st image # SIFT features: {n1}'.format(n1=x.shape[0]))
    print('2nd image # SIFT features: {n2}'.format(n2=y.shape[0]))
    with Timer('ann-compute'):
        # compute for each entry in `y` the index in `x` that is L2 nearest
        nn_idx = ann_hnswlib(x, y)
    # get ratio of the distance of closest and second closest points
    idx0, idx1 = nn_idx.T
    dist0 = np.sum(np.square(x[idx0] - y), axis=-1)
    dist1 = np.sum(np.square(x[idx1] - y), axis=-1)
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
    ax = fig.add_subplot(1,1,1)
    ax.imshow(c_im, interpolation='nearest')
    # draw sift points
    x0, y0 = xd[:, :2].T
    x1, y1 = yd[:, :2].T
    x1 += shift
    ax.plot(x0, y0, 'rx')
    ax.plot(x1, y1, 'bx')
    # plot the connections (a random subset for legibility)
    lines = np.asarray(zip(zip(x0,y0),zip(x1,y1)))
    # randomize the colors of the lines
    rand_idx = np.random.randint(lines.shape[0],size=int(lines.shape[0]*args.percent_to_show))
    lines = lines[rand_idx]
    lc = mc.LineCollection(lines,cmap=plt.cm.gist_ncar,linewidths=1)
    lc.set_array(np.random.random(lines.shape[0]))
    ax.add_collection(lc)
    ax.autoscale()
    # show
    plt.show(block=True)
