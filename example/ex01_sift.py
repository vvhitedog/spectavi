"""An example showing the computation of SIFT keypoints and descriptors.
"""
from matplotlib import pyplot as plt
from util import imread
from spectavi.feature import sift_filter
import argparse

example_text = '''example:
    python ex01_sift.py ../data/castle/01.jpg ../data/castle/02.jpg
'''

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='SIFT keypoint example.',
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', metavar='IM', type=str, nargs='+',
                        help='images to compute SIFT')
    args = parser.parse_args()
    # process each image and show
    for imagefn in args.images:
        # only grayscale supported in SIFT
        im = imread(imagefn, force_grayscale=True)
        # compute SIFT
        sift_kps = sift_filter(im)
        # get x and y coordinates of SIFT features
        x, y = sift_kps[:, :2].T
        plt.figure()
        # plot image
        plt.imshow(im, cmap='gray')
        # plot keypoints as red x's
        plt.plot(x, y, 'rx')
    plt.show(block=True)
