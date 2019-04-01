# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:24:26 2019

@author: kangzhiq
"""

from skimage.feature import daisy
import numpy as np
from skimage import io
import os
from skimage.color import rgb2gray 
import time
import scipy as sp


i = io.imread("./imageS/000.jpg")
i_g = rgb2gray(image)
io.imshow(i_g)

res = daisy(i_g)

descriptors = np.array((res.shape[0]*res.shape[1], 
                            res.shape[2]))
for i in range(200):
    for j in range(res.shape[0]):
        n = res.shape[1]
        
    
desc_num = np.zeros(300)
descriptors = np.empty((0,200))
ind = 0
for i in os.listdir("./images"):
    image = io.imread("./images/" + i)
    image_g = rgb2gray(image)
    image_d = daisy(image_g, step=16, radius=32, rings=3, histograms=8, 
                    orientations=8)
    descriptor = np.reshape(image_d, 
                            (image_d.shape[0]*image_d.shape[1], 
                             image_d.shape[2]),
                             order='F')
    descriptors = np.concatenate((descriptors, descriptor))
    desc_num[ind] = len(descriptors)
    ind += 1
    print(ind)
    
from sklearn.cluster import KMeans 

clusters = 20
kmeans = KMeans(n_clusters=clusters).fit(descriptors)   
res = kmeans.predict(descriptors)    

discr_histy = np.zeros((300,clusters))
for i in range(300):
    print(i)
    i_start = int(desc_num[i-1]) if i>0 else 0
    i_end = int(desc_num[i])
    discr_histy[i,:] = np.bincount(res[i_start:i_end], minlength=clusters)
    discr_histy[i,:] /= np.sum(discr_histy[i,:])

descr0 #histo de l'image 
d = sp.spatial.distance_matrix(descr0,discr_histy)
d_i = np.argsort(d[0,:])