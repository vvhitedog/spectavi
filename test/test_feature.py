from unittest import TestCase
from nose.plugins.attrib import attr
import numpy as np
from spectavi import feature
import os

@attr(speed='fast')
class FeatureTests(TestCase):


    def sift_comparison_test(self):
        """
        Compute SIFT features using spectavi, and compare to sift features
        pre-computed using vlfeat's binary implementation. Ensure that the two
        match.
        """
        # Load pre-computed SIFT features
        oneup = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        datapath = os.path.join(oneup,'data','sift-test','sur-ogre.sift')
        precomputed_sf = np.loadtxt(datapath)
        # Compute SIFT features using spectavi
        impath = os.path.join(oneup,'data','sift-test','sur-ogre.npz')
        im = np.load(impath)['im']
        sf = feature.sift_filter(im)
        # Do the comparison
        assert np.allclose(sf,precomputed_sf)



