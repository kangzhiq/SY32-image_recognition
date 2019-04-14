# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:26:35 2019

@author: zhiqi
"""
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize, rescale

#Q1
labels = np.loadtxt("label.txt") #load loabels
ratio = np.sum(labels[:, 3])/np.sum(labels[:, 4]) # Get ratio of h and l

#Q2 
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(labels[:,3], 50)
plt.show() #height>=75 is good
n, bins, patches = plt.hist(labels[:,4], 50)
plt.show() #Width>=50 is good

w_h = 50 #Height of window
w_w = 75 #Width of window

def extract_face(img, coor, index):
    ligne = int(coor[1])
    colonne = int(coor[2])
    h = int(coor[3])
    l = int(coor[4])
    face = np.zeros((h, l))
    for i in np.arange(ligne, ligne+h):
        for j in np.arange(colonne, colonne+l):
            face[i-ligne, j-colonne] = img[i, j]
    #Perform Gaussian smoothing to avoid aliasing artifacts
    face = resize(face, (w_h, w_w)) 
    io.imsave('./pos/%04d.jpg'%(index), face)
    return 0

def save_pos_data():        
    labels = np.loadtxt("label.txt")
    index = 0
    for i in np.arange(1,1000):
        print(i)
        img = rgb2gray(io.imread('./train/%04d.jpg'%(i)))
        img_g = rgb2gray(img)
        while labels[index,0]==i:    
            extract_face(img_g, labels[index], index)
            index += 1
    return 0
save_pos_data()

#Q3
import random
def check_IoU(labels_img, line, column):
    isValide = True
    #Check IoU for each face
    for i in len(labels_img):
        #Get values of face
        face_l = labels_img[1]
        face_c = labels_img[2]
        face_h = labels_img[3]
        face_w = labels_img[4]
        
        #Intersection of lines
        start_l = min(face_l, line)
        end_l = max(face_l+face_h, line+w_h)
        intersec_h =  face_h + w_h - (end_l-start_l)
        
        #Intersection of columne
        start_c = min(face_c, column)
        end_c = max(face_c+face_w, column+w_w)
        intersec_w = face_w+w_w-(end_c-start_c)
        
        #If Intersection area>0 then calculate ratio of IoU
        if intersec_h > 0 and intersec_w > 0:
            intersec_area = intersec_h * intersec_w
            face_area = face_h*face_w
            window_area = w_h*w_w
            ratio = intersec_area/(face_area+window_area-intersec_area)
            if ratio > 1/2:
                isValide = False
    return isValide

def random_init_window(labels_img, H, L):
    valide = False
    while not valide:
        line = int(random.random()*H)
        if line+w_h > H:
            continue
        column = int(random.random()*L)
        if column+w_w >L:
            continue
        if check_IoU(labels_img, line, column):
            valide = True
    return line, column

def cut_window_and_save(line, column, img, iname):
    face = np.zeros((w_h, w_w))
    for i in np.arange(line, line+w_h):
        for j in np.arange(column, column+w_w):
            face[i-line, j-column] = img[i, j]
    #Perform Gaussian smoothing to avoid aliasing artifacts 
    io.imsave('./neg/%s.jpg'%(iname), face)
    return 0


def save_neg_data():
    labels = np.loadtxt("label.txt")
    index = 0
    for i in np.arange(1,1000):
        print(i)
        img = rgb2gray(io.imread('./train/%04d.jpg'%(i)))
        img_g = rgb2gray(img)
        labels_img = np.empty((0,5))
        for j in range(10):
            rescale_rate = random.random()*5 #rescale rate between 0 and 5
            H = int(img_g.shape[0])
            L = int(img_g.shape[1])
            while labels[index,0]==i:    
                labels_img = np.concatenate(labels_img, labels[index])
                index += 1
            labels_img[:,1:] *= rescale_rate # rescale the face label value
            labels_img = labels_img.astype(int)
            win_l, win_c = random_init_window(labels_img, H, L)
            cut_window_and_save(win_l, win_c, img_g, i+"_"+index+"_"+j)
            
            
            
            
            
            
            

            
        
        
        
        
        
        
        
        
        
