#!/usr/bin/env python
'''
Code to generate fisher_vectors from images. 
'''

import numpy as np
import scipy as sp
import cv2
from sklearn import cluster
from sklearn import mixture
from matplotlib import pyplot as plt
import mr8
from sklearn.externals import joblib
import os,sys
from itertools import product, chain
import argparse


class Fisher_Vector():

    def __init__(self, save_flag=False, out_dir='.'):
        self.outdir = out_dir
        
    #Applies the 38 Filters from MR8 FB and applies Max response on each scale of the edge and bar filters to create 8 filter responses
    def create_mr8_features(self, img, sigmas = [1,2,4], n_ort = 6):
        if img == None:
            raise Exception('No image found')
        self.mr8bank = mr8.MR8_FilterBank()
        self.edge, self.bar, self.rot= self.mr8bank.makeRFSfilters(sigmas=sigmas, n_orientations = n_ort)
        self.filterbank = chain(self.edge, self.bar, self.rot)
        self.n_filters = len(self.edge)+len(self.bar)+len(self.rot)
        self.filterbank = chain(self.edge,self.bar, self.rot)
        self.responses = self.mr8bank.apply_filterbank(img, self.filterbank)
        return self.responses
    
    def create_feature_vectors(self, img):
        if img == None:
            raise Exception('No image found')
        responses = self.create_mr8_features(img)
        out_img = np.zeros((img.shape[0], img.shape[1], 11), dtype = np.float32)
        for i in xrange(img.shape[0]):
            for j in xrange(img.shape[1]):
                vec = np.zeros((8,), dtype=np.float32)
                for r in xrange(len(responses)):
                    vec[r] = responses[r][i,j]
                l2_norm = np.linalg.norm(vec)
                ##to combat 0 norm at edges
                if l2_norm == 0:
                    l2_norm = 1

                vec = vec * ( np.log(1 + l2_norm)/0.03) / l2_norm
                new_vec = np.zeros((11,), dtype = np.float32)
                new_vec[:8] = vec[:]
                cent_dist = np.linalg.norm(np.asarray([i,j]) - np.asarray([128.,128.]))/128.
                angle = np.arctan(abs(i-128)/abs(j-128)) if j-128 != 0 else 0
                cent_flag = np.sign(j-128)
                new_vec[8:] = np.asarray([cent_dist, angle, cent_flag])[:]
                #print new_vec.shape
                for m in xrange(11):
                    out_img[i,j,m] = new_vec[m]
                #print out_img[i,j] 
        #for i in xrange(11):
        #    plt.figure();plt.imshow(out_img[:,:,i], cmap='gray')
        #plt.show()
        return out_img

    def create_vocabulary(self,features, num_clusters=800, save_flag=True, outpath=''):
        self.gmm = mixture.GMM(num_clusters)
        print('Creating GMM for the vocabulary')
        self.gmm.fit(features)
        print('GMM Created.')
        if save_flag :
            joblib.dump(self.gmm, outpath+'_vocabulary.pkl')
        return self.gmm

    def create_fv(self, ftr, gmm, alpha=0.5):
        #if read_gmm == True:
        #    self.gmm = joblib.load('vocabulary.pkl')
        self.gmm = gmm
        means = self.gmm.means_
        covar = self.gmm.covars_
        weights = self.gmm.weights_
        n_comps = self.gmm.n_components
        out = np.zeros((n_comps, 2*ftr.shape[1]),dtype=np.float32)
        probs = self.gmm.predict_proba(ftr)
        #print probs.shape
        for k in xrange(n_comps):
            sum_k1 = 0
            sum_k2 = 0
            T = ftr.shape[0]
            check_flag = 0
            for i in xrange(ftr.shape[0]):
                #print i
                #skip very small probabiity to reduce computation
                if probs[i,k] < 0.001:
                    check_flag+=1
                    continue
                sum_k1 += (( ftr[i,:] - means[k] )/covar[k] ) * probs[i,k]
                sum_k2 += (((( ftr[i,:] - means[k] ) ** 2)/ covar[k] ** 2 ) - 1 ) * probs[i,k]
            #Condition to handle no updates to sum if GMM is highly biased. 
            #Should never go into this for proper functionality
            if check_flag==ftr.shape[0]:
                sum_k1 = np.zeros(ftr[0].shape,dtype=np.float32)
                sum_k2 = np.zeros(ftr[0].shape, dtype=np.float32)



            sum_k1 = (1/( T* np.sqrt(weights[k]+0.00001))) * sum_k1
            #print sum_k1
            sum_k2 = (1/( T* np.sqrt(2*weights[k]+0.00001))) * sum_k2
            
            sum_out = np.concatenate((sum_k1, sum_k2),axis=0)
            sum_out = np.sign(sum_out) * np.abs(sum_out) ** alpha
            sum_out = sum_out / (np.linalg.norm(sum_out)+0.00001)
            out[k,:] = sum_out

        print('Out:',out.shape)
        return out.flatten()
    

            
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir',help='<File path to directory with input images')
    parser.add_argument('-m','--mode',help='<Mode to use script in : loc_desc | vocab | fisher_vec >',choices=['loc_desc','vocab','fisher_vec'])
    parser.add_argument('-o','--out',help='<Output file_name>')

    args = parser.parse_args()

    input_dir = args.input_dir
    mode = args.mode
    outpath = args.out

    fv = Fisher_Vector()
    if mode == '--loc_desc':
        loc_dict = dict()
        for img_name in os.listdir(input_dir):
            img = cv2.imread(input_dir+'/'+img_name, 0)
            print('Creating Features for %s'%img_name)
            #plt.imshow(img,cmap='gray');plt.show()
            ftrs = []
            temp = fv.create_feature_vectors(img)
            for i in xrange(temp.shape[0]):
                for j in xrange(temp.shape[1]):
                    ftrs.append(temp[i,j,:])
            #print ftrs
            local_descriptors = np.asarray(ftrs)
            #print local_descriptors
            if np.NaN in local_descriptors:
                raise Exception('WTF')
            loc_dict[img_name] = local_descriptors
        joblib.dump(loc_dict, outpath+'_feature_dict.pkl')
        print('Features for images saved.')
    
    elif mode == '--vocab':
        img_dict = joblib.load(outpath+'_feature_dict.pkl')
        ftr = []
        for img_name in img_dict.viewkeys():
            ftr.append(img_dict[img_name])
        
        ftrs_in = np.asarray(ftr)[0,:,:]
        #print ftrs_in
        vocab = fv.create_vocabulary(ftrs_in, 800, True, outpath)
        print vocab
    
    elif mode == '--fisher_vec':
        fvectors = dict()
        img_dict = joblib.load(outpath+'_feature_dict.pkl')
        vocab = joblib.load(outpath+'_vocabulary.pkl')
        for img_name in img_dict.viewkeys():
            print('Creating Fisher Vectors for %s'%img_name)
            fvectors[img_name] = fv.create_fv(ftr=img_dict[img_name],gmm=vocab)
        print len(fvectors.viewkeys())
        joblib.dump(fvectors, outpath+'_fisher_vectors.pkl')
    
    elif mode =='--test_code':
        img_name = input_dir
        img = cv2.imread(img_name,0)

        temp = fv.create_feature_vectors(img)
        for i in xrange(11):
            plt.figure()
            plt.imshow(temp[:,:,i])
        plt.show()

     




