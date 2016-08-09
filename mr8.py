#!/usr/bin/env python
'''
Utility to generate MR8 Filter Bank
'''
__author='Ameya Joshi'
__email = 'ameya@sigtuple.com'

from skimage import filters
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
from itertools import product, chain
from scipy.misc import face
#from sklearn.externals.joblib import Parallel, delayed

class MR8_FilterBank():

    def __init__(sigma=[1,2,4], n_orienatations=6):
        return
    
    def make_gaussian_filter(self, x, sigma, order=0):
        if order > 2:
            raise ValueError("Only orders up to 2 are supported")
        # compute unnormalized Gaussian response
        response = np.exp(-x ** 2 / (2. * sigma ** 2))
        if order == 1:
            response = -response * x
        elif order == 2:
            response = response * (x ** 2 - sigma ** 2)
        # normalize
        response /= np.abs(response).sum()
        return response

    def makefilter(self, scale, phasey, pts, sup):
        gx = self.make_gaussian_filter(pts[0, :], sigma=3 * scale)
        gy = self.make_gaussian_filter(pts[1, :], sigma=scale, order=phasey)
        f = (gx * gy).reshape(sup, sup)
        # normalize
        f /= np.abs(f).sum()
        return f
    
    def makeRFSfilters(self, radius=24, sigmas=[1, 2, 4], n_orientations=6):
        """ Generates filters for RFS filterbank.

        Parameters
        ----------
        radius : int, default 28
            radius of all filters. Size will be 2 * radius + 1

        sigmas : list of floats, default [1, 2, 4]
            define scales on which the filters will be computed

        n_orientations : int
            number of fractions the half-angle will be divided in

        Returns
        -------
        edge : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
            Contains edge filters on different scales and orientations
        bar : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
            Contains bar filters on different scales and orientations
        rot : ndarray (2, 2*radius+1, 2*radius+1)
            contains two rotation invariant filters, Gaussian and Laplacian of
            Gaussian
        """

        support = 2 * radius + 1
        x, y = np.mgrid[-radius:radius + 1, radius:-radius - 1:-1]
        orgpts = np.vstack([x.ravel(), y.ravel()])

        rot, edge, bar = [], [], []
        for sigma in sigmas:
            for orient in xrange(n_orientations):
                # Not 2pi as filters have symmetry
                angle = np.pi * orient / n_orientations
                c, s = np.cos(angle), np.sin(angle)
                rotpts = np.dot(np.array([[c, -s], [s, c]]), orgpts)
                edge.append(self.makefilter(sigma, 1, rotpts, support))
                bar.append(self.makefilter(sigma, 2, rotpts, support))
        length = np.sqrt(x ** 2 + y ** 2)
        rot.append(self.make_gaussian_filter(length, sigma=10))
        rot.append(self.make_gaussian_filter(length, sigma=10, order=2))

        # reshape rot and edge
        edge = np.asarray(edge)
        edge = edge.reshape(len(sigmas), n_orientations, support, support)
        #print 'edge shape',edge.shape
        bar = np.asarray(bar).reshape(edge.shape)
        #print 'bar shape',bar.shape
        rot = np.asarray(rot)[:, np.newaxis, :, :]
        #print rot.shape
        return edge, bar, rot


    def apply_filterbank(self, img, filterbank):
        from scipy.ndimage import convolve
        result = []
        #print img
        for battery in filterbank:
            #print 'Hi'
            #print len(battery), battery[0].shape
            response = [convolve(img, filt, mode='reflect') for filt in battery]
            #response = Parallel(n_jobs=5)(
                    #delayed(convolve)(img, filt) for filt in battery)

            #print len(response)
            max_response = np.max(response, axis=0)
            result.append(max_response)
            print("battery finished")
        #print len(result)
        return result



if __name__ == "__main__":
    sigmas = [1, 2, 4]
    n_sigmas = len(sigmas)
    n_orientations = 6
    mr8 = MR8_FilterBank() 
    edge, bar, rot = mr8.makeRFSfilters(sigmas=sigmas,
            n_orientations=n_orientations)

    n = n_sigmas * n_orientations

    # plot filters
    # 2 is for bar / edge, + 1 for rot
    fig, ax = plt.subplots(n_sigmas * 2 + 1, n_orientations)
    for k, filters in enumerate([bar, edge]):
        for i, j in product(xrange(n_sigmas), xrange(n_orientations)):
            row = i + k * n_sigmas
            ax[row, j].imshow(filters[i, j, :, :], cmap=plt.cm.gray)
            ax[row, j].set_xticks(())
            ax[row, j].set_yticks(())
    ax[-1, 0].imshow(rot[0, 0], cmap=plt.cm.gray)
    ax[-1, 0].set_xticks(())
    ax[-1, 0].set_yticks(())
    ax[-1, 1].imshow(rot[1, 0], cmap=plt.cm.gray)
    ax[-1, 1].set_xticks(())
    ax[-1, 1].set_yticks(())
    for i in xrange(2, n_orientations):
        ax[-1, i].set_visible(False)

    # apply filters to lena
    img = face(gray=True).astype(np.float)
    print(img.shape)
    filterbank = chain(edge, bar, rot)
    n_filters = len(edge) + len(bar) + len(rot)
    print('[applying]:%d'%n_filters)
    response = mr8.apply_filterbank(img, filterbank)
    # plot responses
    fig2, ax2 = plt.subplots(3, 3)
    for axes, res in zip(ax2.ravel(), response):
        axes.imshow(res, cmap=plt.cm.gray)
        axes.set_xticks(())
        axes.set_yticks(())
    ax2[-1, -1].set_visible(False)
    plt.show()



    
    

