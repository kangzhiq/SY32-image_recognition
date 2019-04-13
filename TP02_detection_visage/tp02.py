# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:26:35 2019

@author: zhiqi
"""

#Q1
import numpy as np
import matplotlib as plt
from skimage import io
import os
from skimage.color import rgb2gray 

labels = np.loadtxt("./label.txt")

img = io.imread("./train/0001.jpg")
img_g = rgb2gray(img)
io.imshow(img_g)



ligne = int(labels[0,1])
colonne = int(labels[0,2])
h = int(labels[0,3])
l = int(labels[0,4])
face = np.zeros((h, l))

for i in np.arange(ligne, ligne+h):
    for j in np.arange(colonne, colonne+l):
        face[i-ligne, j-colonne] = img_g[i, j]
io.imshow(face)
io.imshow(img_g)
io.imsave("./pos/0001.jpg", face)

labels = np.loadtxt("label.txt")

h_moyen = np.mean(labels[:, 3])
l_moyen = np.mean(labels[:, 4])

#Q2

def extract_face(img, coor, index):
    ligne = int(coor[1])
    colonne = int(coor[2])
    h = int(coor[3])
    l = int(coor[4])
    face = np.zeros((h, l))
    for i in np.arange(ligne, ligne+h):
        for j in np.arange(colonne, colonne+l):
            face[i-ligne, j-colonne] = img[i, j]
    io.imsave('./pos/%04d.jpg'%(index), face)
    return 0

def extract_neg(img, coor):
    return 0

def save_training_data():        
    labels = np.loadtxt("label.txt")
    index = 0
    for i in np.arange(1,10):
        print(i)
        img = rgb2gray(io.imread('./train/%04d.jpg'%(i)))
        img_g = rgb2gray(img)
        while labels[index,0]==i:    
            extract_face(img_g, labels[index], index)
            #extract_neg(img_g, labels[index,:])
            index += 1
    return 0

save_training_data()    


