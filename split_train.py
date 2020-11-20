#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:07:11 2020

@author: laura
"""

import numpy as np
from utils import load_image
from skimage.measure import label, regionprops
from pred_map import array2raster
from osgeo import gdal
import os


labels_tr = load_image("/home/laura/Projects/Trabalho-Camile/Data/samples_A1_train2tif.tif")
refrence_img = gdal.Open("/home/laura/Projects/Trabalho-Camile/Data/samples_A1_train2tif.tif")

labels_obj = label(labels_tr)
uniq, count = np.unique(labels_obj, return_counts=True)
dictionary = dict(zip(uniq, count))
dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=False)
segmented = np.zeros(labels_tr.shape, dtype='uint8')
new_train = labels_tr.copy()

list_class = np.unique(labels_tr)[1:]
list_class = list_class.tolist()

for lb, c in dictionary:
    class_obj = np.unique(labels_tr[labels_obj==lb])[0]
    try:
        list_class.remove(class_obj)
        segmented[labels_obj==lb] = class_obj
        new_train[labels_obj==lb] = 0
        if class_obj not in np.unique(new_train):
            new_train[labels_obj==lb] = class_obj
    except:
        pass
        
array2raster(os.path.join('new_train.tif'), refrence_img, new_train, 'Byte')
array2raster(os.path.join('new_dev.tif'), refrence_img, segmented, 'Byte')
