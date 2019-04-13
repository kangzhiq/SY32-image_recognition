# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:12:57 2019

@author: zhiqi
"""
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import numpy as np

model = VGG16(weights='imagenet')

print(model.summary())

def prepro_img(path):    
    img = load_img(path, target_size=(224, 224))  
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

img = prepro_img("kangaroo.jpg")
y = model.predict(img)
label = decode_predictions(y)    
print(label)


img = prepro_img("crab.jpg")
y = model.predict(img)
label = decode_predictions(y)    
print(label)

from keras.models import Model
model_feat = Model(inputs=model.input, outputs=model.get_layer('fc2').output)

img = prepro_img("kangaroo.jpg")
y_feat = model_feat.predict(img)

import os
from skimage import io
from skimage.color import rgb2gray

descriptors = np.zeros((300,4096))
ind = 0
for i in os.listdir("./images"):
    image = prepro_img("./images/" + i)
    img_d = model_feat.predict(image)
    descriptors[ind,:] = img_d
    print(ind)
    ind += 1

from sklearn.cluster import KMeans 

clusters = 20
kmeans = KMeans(n_clusters=clusters).fit(descriptors)  
res = kmeans.predict(descriptors)

img = prepro_img("kangaroo.jpg")
y = model_feat.predict(img)
img_lable = kmeans.predict(y)






