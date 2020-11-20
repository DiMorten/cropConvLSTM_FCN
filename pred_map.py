#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 09:46:03 2018

@author: laura
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from skimage.util.shape import view_as_windows
import tensorflow as tf
import os
from utils import check_folder, load_image
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
import os.path
import re

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

import glob
from skimage.measure import label

def hysteresis(img, weak, strong=0.9):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09, w = 0.4, s = 0.9):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.float)
    
    weak = np.float(w)
    strong = np.float(s)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def array2raster(newRasterfn, dataset, array, dtype):
    """
    save GTiff file from numpy.array
    input:
        newRasterfn: save file name
        dataset : original tif file
        array : numpy.array
        dtype: Byte or Float32.
    """
    cols = array.shape[1]
    rows = array.shape[0]
    originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform() 

    driver = gdal.GetDriverByName('GTiff')

    # set data type to save.
    GDT_dtype = gdal.GDT_Unknown
    if dtype == "Byte": 
        GDT_dtype = gdal.GDT_Byte
    elif dtype == "Float32":
        GDT_dtype = gdal.GDT_Float32
    else:
        print("Not supported data type.")

    # set number of band.
    if array.ndim == 2:
        band_num = 1
    else:
        band_num = array.shape[2]

    outRaster = driver.Create(newRasterfn, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    # Loop over all bands.
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b + 1)
        # Read in the band's data into the third dimension of our array
        if band_num == 1:
            outband.WriteArray(array)
        else:
            outband.WriteArray(array[:,:,b])

    # setteing srs from input tif file.
    prj=dataset.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/v3',
                        help="Experiment directory containing params.json")
    parser.add_argument('--img_data', default='/home/laura/Projects/Trabalho-Camile/Data/orthoimages/ortoA1_25tiff.tif', 
                    help="Path of the images dataset")
    parser.add_argument('--gt_dir', default='/home/laura/Projects/Trabalho-Camile/Data/samples_A1_test2tif.tif', help="Path of the gt dataset")
    parser.add_argument('--gt_tr', default='/home/laura/Projects/Trabalho-Camile/Data/samples_A1_train2tif.tif', help="Path of the gt dataset")
    parser.add_argument('--mask', default='/home/laura/Projects/Trabalho-Camile/Data/mask.tif', 
                        help="Path of the refrence train image")    
    args = parser.parse_args()
    
    
    for i in range(1):    
        for j in range(1):    
            tf.compat.v1.reset_default_graph()        
            
            model_dir_k = os.path.join(args.model_dir,str(i))    
    
            refrence_img = gdal.Open(args.img_data)
            image = refrence_img.ReadAsArray()
            image = np.rollaxis(image,0,3)
            pred_prob1 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.1.npy'))
            pred_prob2 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3.npy'))
            pred_prob3 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.5.npy'))
            prob_mean = (pred_prob1+pred_prob2+pred_prob3)/3
            pred_data = np.argmax(prob_mean, axis = -1)
            
            pred_depth1 = np.load(os.path.join(model_dir_k, 'pred_depth_128_0.1.npy'))
            pred_depth2 = np.load(os.path.join(model_dir_k, 'pred_depth_128_0.3.npy'))
            pred_depth3 = np.load(os.path.join(model_dir_k, 'pred_depth_128_0.5.npy'))
            pred_depth_mean = (pred_depth1+pred_depth2+pred_depth3)/3

            # Back to original labels
            labels_tr = load_image(args.gt_tr).astype('uint8')
            cl = np.max(np.unique(labels_tr))
            labels_tm = labels_tr-1
            labels_tm[labels_tm==255] = cl
            classes = np.unique(labels_tm)
            new_labels2labels = dict((i, c) for i, c in enumerate(classes))
            
            lbl_tmp = pred_data.copy()
            classes_pred = np.unique(pred_data)
            for j in range(len(classes_pred)):
                pred_data[lbl_tmp == classes_pred[j]] = new_labels2labels[classes_pred[j]] 
                
            pred_data = pred_data+1
            
            array2raster(os.path.join(model_dir_k,'predicted_class_map.tif'), refrence_img, pred_data, 'Byte')
            
            # refrence_img = gdal.Open(args.gt_dir)
            prob2save = np.amax(prob_mean, axis = -1)
            
            array2raster(os.path.join(model_dir_k,'predicted_prob_map.tif'), refrence_img, prob2save, 'Float32')
               
            array2raster(os.path.join(model_dir_k,'predicted_depth.tif'), refrence_img, pred_depth_mean, 'Float32')

            array2raster(os.path.join(model_dir_k,'image.tif'), refrence_img, image[:,:,0:3], 'Float32')



            # # load mask image
            # mask = load_image(args.mask)
            # mask = mask.astype('uint8')
            
            # # load mask image
            # treshold = 0.95
            # testlab = load_image(args.gt_dir)
            # testlab = testlab.astype('uint8')
            # prob2save[mask != 99] = 0            
            # (res, weak, strong) = threshold(prob2save, w = 0.7, s = treshold)
            # prob_treshold = hysteresis(res, weak, strong=treshold)
            
            # predicted_treshold = np.zeros(pred_data.shape)
            # predicted_treshold[prob_treshold>=treshold] = pred_data[prob_treshold>=treshold]
            # predicted_treshold[labels_tr>0] = labels_tr[labels_tr>0]
            # predicted_treshold[mask != 99] = 0
            # predicted_treshold[testlab>0] = 0
            
            # pizel_crown = [4725,3399,3636,3492,3767,4874,1385,2495,6029,1406,11320,2065,2246,3042]
            
            # labels_obj = label(predicted_treshold)
            # uniq, count = np.unique(labels_obj, return_counts=True)
            # segmented = np.zeros(predicted_treshold.shape)
            # count_tresh = 1000
            # for lb, c in zip(uniq[1:],count[1:]):
            #     if c > count_tresh:
            #         segmented[labels_obj==lb] = predicted_treshold[labels_obj==lb] 
            
            # array2raster(os.path.join(model_dir_k,'tr_self_labeling_double_tresh.tif'), refrence_img, segmented, 'Byte')

