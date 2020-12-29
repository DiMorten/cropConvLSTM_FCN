#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:40:13 2020

@author: laura
"""

"""Train the model"""


import argparse
import logging
import os
import random
from utils import load_image, metrics, add_padding
from utils import set_logger, save_dict_to_json 
from utils import Params, check_folder, Results
from utils import extract_patches_coord
import glob
import numpy as np
from generator import DataGenerator
from models import cnn, Monitor, f1_mean
from keras.utils import plot_model
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
import keras
from sklearn.cluster import KMeans
import gc
from loss import categorical_focal_loss, masked_mse, accuracy_mask, f_score
from collections import Counter, OrderedDict
from sklearn import preprocessing as pp
import joblib
import pandas as pd
import seaborn as sns
#from deeplabv3 import Deeplabv3
#from deeplab_versions import DeepLabVersions
import tensorflow as tf
import pdb
from pathlib import Path
import colorama
from dataSource import CampoVerdeSAR
import deb

colorama.init()

def im_load(path):
    im_names=['20151029_S1','20151110_S1','20151122_S1','20151204_S1','20151216_S1','20160121_S1','20160214_S1','20160309_S1','20160321_S1','20160508_S1','20160520_S1','20160613_S1','20160707_S1','20160731_S1']
    out=[]
    for im_name in im_names:
        im=np.load(path/(im_name+'.npy')).astype(np.float16)
        out.append(im)
    out=np.asarray(out).astype(np.float16)
    print(out.shape)
    t_len, row, col, bands = out.shape
    out=np.moveaxis(out,0,-1)
    out=np.reshape(out,(row,col,bands*t_len))
    print(out.shape)
    
    return out
#    pdb.set_trace()
    # transpose, concatenate with bands


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='../results',
                    help="Experiment directory containing params.json")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")
parser.add_argument('--img_data', default='../data/cv/sar/', 
                    help="Path of the train RGB image")
parser.add_argument('--gt_tr', default='../data/cv/labels/20160731_S1.tif', 
                    help="Path of the refrence train image")
parser.add_argument('--train_test_mask', default='../data/cv/TrainTestMask.tif', 
                    help="Path of the refrence train image")                    
#parser.add_argument('--gt_val', default='new_dev.tif', 
#                    help="Path of the refrence train image")
#parser.add_argument('--gt_test', default='/home/laura/Projects/Trabalho-Camile/Data/samples_A1_test2tif.tif', 
#                    help="Path of the refrence train image")
#parser.add_argument('--depth', default='/home/laura/Projects/Trabalho-Camile/FCN_MTL_v2/data/imgTrain_depth.tif', 
# parser.add_argument('--depth', default='/home/laura/Projects/Trabalho-Camile/Data/train_edt.tif',
#                    help="Path of the refrence train image")
parser.add_argument('--mask', default='../data/cv/TrainTestMask.tif', 
                    help="Path of the refrence train image")
parser.add_argument('--mode', default="Train", 
                    help="If True use a dev set, if False, use training set to monitor the metrics")

if __name__ == '__main__':

    val_mode = False
    # define dataset
#    ds = CampoVerdeSAR()
#    image = ds.loadIms()

    # Load the parameters from json file
    args = parser.parse_args()

    
    args.img_data=Path(args.img_data)
#    json_path = os.path.join(args.model_dir, 'params.json')
    json_path = 'params.json'
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
           
    # load train image
    #image = np.load(args.img_data)
    image = im_load(args.img_data)
    print(image.shape)
    #image = np.rollaxis(image,0,3)
    row,col,bands = image.shape
    params.channels = bands

    # load label image for train
    labels = load_image(args.gt_tr)
    mask = load_image(args.train_test_mask)


    if val_mode==True:
        # load label image for dev
        labels_dev = labels.copy()
        labels_dev[labels_dev!=3]=0
        labels_dev = labels_dev.astype('uint8')


    if args.mode == "Train":
        labels[mask!=1]=0
    else:
        labels[mask!=2]=0
#        labels = load_image(args.gt_test)
        
    labels = labels.astype('uint8')
    deb.prints(labels.shape)
    deb.prints(np.unique(labels,return_counts=True))
    
    
    # load depth image
#    depth = load_image(args.depth)
    
    # Start training or evaluation
    for i in range(1):
        model_k = os.path.join(args.model_dir, str(i))
        check_folder(model_k)
        
        if args.mode != "Eval":
            
            # get training coords from normalize data
            coords = np.where(labels!=0)
            
            if args.mode == "Train":
                img_tmp = image[coords]
                scaler = pp.StandardScaler().fit(img_tmp)
                img_tmp = []                
                scaler_filename = os.path.join(args.model_dir,"scaler.save")
                joblib.dump(scaler, scaler_filename)
            else:
                scaler_filename = os.path.join(args.model_dir,"scaler.save")
                scaler = joblib.load(scaler_filename)
        
            image = image.reshape(row*col,bands)
            image = scaler.transform(image)
            image = image.reshape(row,col,bands)
            
            if args.mode == "Test":
                params.ovrl = params.ovrl_test
            
            image_tr, stride, step_row, step_col, overlap = add_padding(image, params.patch_size, params.ovrl)
        
            labels_tr, _, _, _, _ = add_padding(labels, params.patch_size, params.ovrl)
##            labels_val, _, _, _, _ = add_padding(labels_dev, params.patch_size, params.ovrl)
        
            # get coords
            coords_tr = extract_patches_coord(labels_tr, params.patch_size, stride, train = True)
##            coords_val = extract_patches_coord(labels_val, params.patch_size, stride, train = False)
            
            coords_tr = np.rollaxis(coords_tr,1,0)
            print(coords_tr.shape)
##            pdb.set_trace()
##            coords_val = np.rollaxis(coords_val,1,0)
              
            #depth_tr, _, _, _, _ = add_padding(depth, params.patch_size, params.ovrl)
        
            #depth_tr = (depth_tr-np.min(depth_tr))/(np.max(depth_tr)-np.min(depth_tr))
            
            # convert original classes to ordered classes
            classes = np.unique(labels_tr)
            deb.prints(classes)
            labels_tr = labels_tr-1
##            labels_val = labels_val-1
            params.classes = len(classes)-1
            labels_tr[labels_tr==255] = params.classes
            classes = np.unique(labels_tr)
            deb.prints(classes)

##            labels_val[labels_val==255] = params.classes
            tmp_tr = labels_tr.copy()
##            tmp_val = labels_val.copy()

            deb.prints(labels_tr.shape)
            deb.prints(np.unique(labels_tr,return_counts=True))  
            labels2new_labels = dict((c, i) for i, c in enumerate(classes))
            new_labels2labels = dict((i, c) for i, c in enumerate(classes))
            for j in range(len(classes)):
                labels_tr[tmp_tr == classes[j]] = labels2new_labels[classes[j]]
##                labels_val[tmp_val == classes[j]] = labels2new_labels[classes[j]]
            deb.prints(labels_tr.shape)
            deb.prints(np.unique(labels_tr,return_counts=True))            
        if args.mode == "Train":       
            # Set the logger
            count_cl = dict(sorted(Counter(labels_tr[labels_tr!=params.classes]).items()))
            prop = np.array((list(count_cl.values())))/np.sum(np.array((list(count_cl.values()))))
            ratio = np.max(np.array((list(count_cl.values()))))/np.array((list(count_cl.values())))
            print("Prop Data ---> {}".format(prop))
            print("Ratio Data ---> {}".format(ratio))
            np.save(os.path.join(args.model_dir,'ratio'),ratio)
            
            set_logger(os.path.join(model_k, 'train.log'))
            # Check that we are not overwriting some previous experiment
            # Comment these lines if you are developing your model and don't care about overwritting
            file_output = os.path.join(model_k,'bestmodel_{}.hdf5'.format(i))
            model_dir_has_best_weights = os.path.isfile(file_output)
            overwritting = model_dir_has_best_weights and args.restore_from is None
            assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"
               
            # Shuflle
            index_tr = np.random.choice(len(coords_tr[0]), len(coords_tr[0]), replace=False)
            
            # index validation
##            index_val = np.array(range(len(coords_val[0])))

            # Define generators
            dim = (params.patch_size, params.patch_size, params.channels)

            training_generator = DataGenerator(image_tr, labels_tr, coords_tr, index_tr, params.channels, 
                                               params.patch_size, params.batch_size, dim,  
                                               samp_per_epoch= 1965, shuffle=True, 
                                               use_augm = params.use_augm)
##samp_per_epoch= params.samp_per_epoch

##            validation_generator = DataGenerator(image_tr, labels_val, coords_val, index_val, params.channels, 
##                                                 params.patch_size, params.batch_size, dim,  shuffle=True)
            
            # validation_generator = DataGenerator(image_tr, labels_val, coords_val, index_val, params.channels, 
            #                                      params.patch_size, params.batch_size, dim,
            #                                      samp_per_epoch = params.samp_epoch_val, shuffle=True, 
            #                                      use_augm = params.use_augm)
            
            # Define optimazer
            optimizer = Adam(lr=params.learning_rate)
            # Define model
            if params.model == "custom":
                model = cnn(img_shape=dim, nb_classes=params.classes)   
            
            else:
                model = DeepLabVersions(dim, params)
                        
            ##plot_model(model, to_file=os.path.join(model_k,'model.png'), show_shapes=True)
            
            cl_ind = [x for x in range(params.classes)]
            losses = {"cl_output": categorical_focal_loss(depth=np.int(params.classes+1), alpha=[ratio.tolist()],class_indexes=cl_ind)}
            # losses = {"cl_output": categorical_focal_loss(depth=np.int(params.classes+1), alpha=[ratio.tolist()],class_indexes=cl_ind), 
            #           "reg_output": masked_mse()}
            lossWeights = {"cl_output": 1.0}
            
            custom_metrics = {"cl_output": accuracy_mask()}
            
            # Compile model
            # model.compile(loss=crossentropy_mask, 
            #               metrics=["accuracy", f1_mean], optimizer=optimizer)
            model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights,
                          metrics=['accuracy']) #custom_metrics
            
            # Train model on dataset
            history = model.fit_generator(generator=training_generator,
##                                validation_data=validation_generator,
                                epochs = params.num_epochs)
                                #steps_per_epoch = params.samp_per_epoch,
##                                callbacks = [Monitor(validation=training_generator,patience = 15,
##                                                    model_dir=model_k, classes=params.classes)]) 
            model.save(file_output)
            
            # save training history        
            filename = os.path.join(model_k,'history_plot_acc.png')     
            Results(history.history['cl_output_f_acc'], 
                    history.history['val_cl_output_f_acc'],
                    history.history['cl_output_loss'],
                    history.history['val_cl_output_loss'],
                    filename).show_result()
            
            filename = os.path.join(model_k,'history_plot_f1batch.png')     
            Results(history.history['mean_f1'], 
                    history.history['val_cl_output_f_acc'],
                    history.history['reg_output_loss'],
                    history.history['val_reg_output_loss'],
                    filename).show_result()
            
            
        elif args.mode == "Test":
            ratio = np.load(os.path.join(args.model_dir,'ratio.npy'))
            if not os.path.isfile(os.path.join(model_k, 'pred_prob_{}_{}.npy'.format(params.patch_size, params.ovrl))):
                # Load moodel
                cl_ind = [x for x in range(params.classes)]
                file_model = os.path.join(model_k,'bestmodel_{}.hdf5'.format(i))
                model = load_model(file_model, custom_objects={"f_cat": categorical_focal_loss(depth=np.int(params.classes+1), alpha=[ratio.tolist()],class_indexes=cl_ind),
                                                              #"f_reg": masked_mse(),
                                                              "f_acc": accuracy_mask(),
                                                              "tf": tf})
             
                row,col,bands = image_tr.shape
                cl_img = np.zeros((row,col,params.classes))
                cl_img = cl_img.astype('float16')
                #reg_img = np.zeros((row,col))
                #reg_img = reg_img.astype('float16')
                
                for m in range(params.patch_size//2,row-params.patch_size//2,stride): 
                    for n in range(params.patch_size//2,col-params.patch_size//2,stride):
                        patch = image_tr[m-params.patch_size//2:m+params.patch_size//2 + params.patch_size%2,
                                  n-params.patch_size//2:n+params.patch_size//2 + params.patch_size%2]
                        patch = patch.reshape((1,params.patch_size,params.patch_size,bands))
                        pred_cl = model.predict(patch)
                        _, x, y, c = pred_cl.shape
                          
                        cl_img[m-stride//2:m+stride//2,n-stride//2:n+stride//2,:] = pred_cl[0,overlap//2:x-overlap//2,overlap//2:y-overlap//2,:]
                        #reg_img[m-stride//2:m+stride//2,n-stride//2:n+stride//2] = pred_reg[0,overlap//2:x-overlap//2,overlap//2:y-overlap//2,0]
            
            
                cl_img = cl_img[overlap//2:-step_row,overlap//2:-step_col,:]
                #reg_img = reg_img[overlap//2:-step_row,overlap//2:-step_col]
                
                np.save(os.path.join(model_k, 'pred_prob_{}_{}'.format(params.patch_size, params.ovrl)), cl_img)
                #np.save(os.path.join(model_k, 'pred_depth_{}_{}'.format(params.patch_size, params.ovrl)), reg_img)
        
                gc.collect()
                del model
                
        elif args.mode == "Eval":
            aa_list = list()
            pre_list = list()
            rec_list = list()
            f_list = list()
            oa_list = list()
            kapp_list = list()
            
            classes = np.unique(labels)
            new_labels2labels = dict((i, c) for i, c in enumerate(classes))
            
            p,r,f,oa,kapp,aa = metrics(params, labels, model_k, new_labels2labels)
            aa_list.append(aa)
            pre_list.append(p)
            rec_list.append(r)
            f_list.append(f)
            oa_list.append(oa)
            kapp_list.append(kapp)

