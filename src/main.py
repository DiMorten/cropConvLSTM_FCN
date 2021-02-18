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
from deeplab_versions import DeepLabVersions, DeepLabConvLSTM
from keras.utils import plot_model
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
import keras
from sklearn.cluster import KMeans
import gc
from loss import categorical_focal_loss, masked_mse, accuracy_mask, f_score, categorical_focal_ignoring_last_label
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
import matplotlib.pyplot as plt
import time, datetime
import cv2
colorama.init()
from utils import cmap, plot_figures_test
def getTimeDelta(im_names):
    time_delta=[]
    for im in im_names:
        date=im[:8]
        print(date)
        time_delta.append(time.mktime(datetime.datetime.strptime(date, 
                                            "%Y%m%d").timetuple()))
    print(time_delta)
    return np.asarray(time_delta)
im_names=['20151029_S1','20151110_S1','20151122_S1','20151204_S1','20151216_S1','20160121_S1',
    '20160214_S1','20160309_S1','20160321_S1','20160508_S1','20160520_S1','20160613_S1',
    '20160707_S1','20160731_S1']    
def im_load(path,im_names):

    out=[]
    for im_name in im_names:
        im=np.load(path/(im_name+'.npy')).astype(np.float16)
        out.append(im)
    out=np.asarray(out).astype(np.float16)
    print("Loaded images shape", out.shape)
    t_len, row, col, bands = out.shape
    out=np.moveaxis(out,0,-1)
    out=np.reshape(out,(row,col,bands*t_len))
    print("Loaded images shape after t and band concatenation", out.shape)
    
    return out
#    pdb.set_trace()
    # transpose, concatenate with bands
def plot_input(im,mask,time_delta):
    time_delta = np.concatenate((time_delta,time_delta),axis=0)
    print(time_delta)
    averageTimeseries = []
    print("Input shape",im.shape,mask.shape)
    for band_id in range(im.shape[-1]):
        im_flat = im[...,band_id].flatten()
        mask_flat = mask.flatten()
        print("t shape",im_flat.shape,mask_flat.shape)

        im_flat = im_flat[mask_flat==1]
        averageTimeseries.append(np.average(im_flat))
    fig, ax = plt.subplots()
    ax.plot(time_delta,averageTimeseries,marker=".")
    ax.set(xlabel='time ID', ylabel='band',title='Image average over time')
    plt.grid()
    plt.show()
        



parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='results',
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
parser.add_argument('--mode', default="Eval", 
                    help="If True use a dev set, if False, use training set to monitor the metrics")

if __name__ == '__main__':

    val_mode = False
    # define dataset
#    ds = CampoVerdeSAR()
#    image = ds.loadIms()

    # Load the parameters from json file
    args = parser.parse_args()
#    args.exp_id = 'first'

#    args.exp_id = 'fourth'
#    args.exp_id = 'third'
####    args.exp_id = 'fifth_moreval'
#    args.exp_id = 'sixth_moreval'
#    args.exp_id = 'seventh_moreval'
 ##   args.exp_id = 'deeplab_rep1'
    args.exp_id = 'deeplab_convlstm_rep1'

    args.dataset = 'cv'

    
    args.img_data=Path(args.img_data)
#    json_path = os.path.join(args.model_dir, 'params.json')
    json_path = 'params.json'
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
           
    # load train image
    #image = np.load(args.img_data)
    image = im_load(args.img_data,im_names)
    time_delta = getTimeDelta(im_names)
    deb.prints(image.shape)
    deb.prints(time_delta)

    #image = np.rollaxis(image,0,3)
    row,col,bands = image.shape
    params.channels = bands

    # load label image for train
    labels = load_image(args.gt_tr)
    mask = load_image(args.train_test_mask)
#    plot_input(image,mask,time_delta)

    if val_mode==True:
        # load label image for dev
        labels_dev = labels.copy()
        labels_dev[labels_dev!=3]=0
        labels_dev = labels_dev.astype('uint8')


    if args.mode == "Train":
        labels[mask!=1]=0
    else:
#       
        test_on_train_set = False
        if test_on_train_set==False:
            labels[mask!=2]=0
        else:
            labels[mask!=1]=0

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
                print("Train. Saving scaler",scaler_filename)                
                joblib.dump(scaler, scaler_filename)
            else:
                scaler_filename = os.path.join(args.model_dir,"scaler.save")
                print("Test. Loading scaler",scaler_filename)
                scaler = joblib.load(scaler_filename)
            
            print("Scaler parameters",scaler.mean_, scaler.var_)
        
            image = image.reshape(row*col,bands)
            image = scaler.transform(image)
            image = image.reshape(row,col,bands)
            
            if args.mode == "Test":
#                params.ovrl_test = 0.5
                params.ovrl_test = 0.3
                params.ovrl = params.ovrl_test
            
            #if args.plot_histogram == True:
            #    plot_histogram(image)

            image_tr, stride, step_row, step_col, overlap = add_padding(image, params.patch_size, params.ovrl)
        
            labels_tr, _, _, _, _ = add_padding(labels, params.patch_size, params.ovrl)
##            labels_val, _, _, _, _ = add_padding(labels_dev, params.patch_size, params.ovrl)
        
            # get coords
            coords_tr = extract_patches_coord(labels_tr, params.patch_size, stride, train = True)
##            coords_val = extract_patches_coord(labels_val, params.patch_size, stride, train = False)
            deb.prints(coords_tr.shape)
            coords_tr_idx = range(coords_tr.shape[0])
            val_n = round(coords_tr.shape[0]*0.01)
            val_n = round(coords_tr.shape[0]*0.0005)
            val_n = round(coords_tr.shape[0]*0.0015)

            deb.prints(val_n)
            deb.prints(len(coords_tr_idx))

            coords_val_idx = np.random.choice(coords_tr_idx,val_n,replace=False)
            coords_val = coords_tr[coords_val_idx]
            if args.mode== "Train":
                coords_tr = np.delete(coords_tr, coords_val_idx, axis = 0)

            print("val coords were extracted")
            deb.prints(coords_tr.shape)
            deb.prints(coords_val.shape)
            coords_tr = np.rollaxis(coords_tr,1,0)
            coords_val = np.rollaxis(coords_val,1,0)

            deb.prints(coords_tr.shape)

			

##            pdb.set_trace()
              
            #depth_tr, _, _, _, _ = add_padding(depth, params.patch_size, params.ovrl)
        
            #depth_tr = (depth_tr-np.min(depth_tr))/(np.max(depth_tr)-np.min(depth_tr))
            
            # convert original classes to ordered classes
            classes = np.unique(labels_tr)


##            labels_val[labels_val==255] = params.classes
##            tmp_val = labels_val.copy()

            if args.mode=="Train":
                print(" debuging class change to 0")
                deb.prints(classes)
                deb.prints(np.unique(labels_tr,return_counts=True))
                
                #classes_label_original = np.unique(labels_tr)
                deb.prints(classes)
                labels_tr = labels_tr-1
    ##            labels_val = labels_val-1
                deb.prints(np.unique(labels_tr,return_counts=True))  
                labels_tr[labels_tr==255] = classes[-1]
                deb.prints(np.unique(labels_tr,return_counts=True))  
                classes = np.unique(labels_tr)
                deb.prints(classes)
                print("end debuging class change to 0") 
                
                
            print(" debugging class change to consecutive values")


            deb.prints(labels_tr.shape)
            deb.prints(np.unique(labels_tr,return_counts=True))  
            labels2new_labels = dict((c, i) for i, c in enumerate(classes))
            new_labels2labels = dict((i, c) for i, c in enumerate(classes))

            deb.prints(labels2new_labels)
            deb.prints(new_labels2labels)

            params.classes = len(classes)-1
            deb.prints(params.classes)  

            if args.mode=="Train":
                deb.prints(classes)
                tmp_tr = labels_tr.copy()
                
                for j in range(len(classes)):
                    print("class, new class",classes[j], labels2new_labels[classes[j]])
                    labels_tr[tmp_tr == classes[j]] = labels2new_labels[classes[j]]
    ##                labels_val[tmp_val == classes[j]] = labels2new_labels[classes[j]]
                print("Classes to newclasses ok")
                deb.prints(labels_tr.shape)
                deb.prints(np.unique(labels_tr,return_counts=True))  
                #pdb.set_trace()
                  

                  
                cv2.imwrite('labels_sample_full.png',labels_tr*20)   
  
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
            file_output = os.path.join(model_k,'bestmodel_{}_{}.hdf5'.format(args.dataset, args.exp_id))
            deb.prints(file_output)
            model_dir_has_best_weights = os.path.isfile(file_output)
            overwritting = model_dir_has_best_weights and args.restore_from is None
            #assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"
               
            # Shuflle
            index_tr = np.random.choice(len(coords_tr[0]), len(coords_tr[0]), replace=False)
            
            # index validation
            index_val = np.array(range(len(coords_val[0])))

            # Define generators
            dim = (params.patch_size, params.patch_size, params.channels)

            training_generator = DataGenerator(image_tr, labels_tr, coords_tr, index_tr, params.channels, 
                                               params.patch_size, params.batch_size, dim,  
                                               samp_per_epoch= params.samp_per_epoch, shuffle=True,  #samp_per_epoch= 1965
                                               use_augm = params.use_augm)
##samp_per_epoch= params.samp_per_epoch

            validation_generator = DataGenerator(image_tr, labels_tr, coords_val, index_val, params.channels, 
                                                 params.patch_size, params.batch_size, dim,  shuffle=True)
            
            # validation_generator = DataGenerator(image_tr, labels_val, coords_val, index_val, params.channels, 
            #                                      params.patch_size, params.batch_size, dim,
            #                                      samp_per_epoch = params.samp_epoch_val, shuffle=True, 
            #                                      use_augm = params.use_augm)
            
            # Define optimazer
            optimizer = Adam(lr=params.learning_rate)
            # Define model
            params.add_reg = False
            if params.model == "custom":
                model = cnn(img_shape=dim, nb_classes=params.classes)   
            
            else:
                # model = DeepLabVersions(dim, params)
                model = DeepLabConvLSTM(dim, params)
            print(model.summary())
            
            ##plot_model(model, to_file=os.path.join(model_k,'model.png'), show_shapes=True)
            
            cl_ind = [x for x in range(params.classes)]
            losses = {"cl_output": categorical_focal_loss(depth=np.int(params.classes+1), alpha=[ratio.tolist()],class_indexes=cl_ind)}
            #losses = {"cl_output": categorical_focal_ignoring_last_label(alpha=0.25,gamma=2)}
            
            
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
                                validation_data=validation_generator,
                                epochs = params.num_epochs,
                                #steps_per_epoch = params.samp_per_epoch,
                                callbacks = [Monitor(validation=validation_generator,patience = 15,
                                                    model_dir=model_k, classes=params.classes)]) 
            model.save(file_output)
            
            # save training history        
            filename = os.path.join(model_k,'history_plot_acc.png')     
            
            try:
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
            except:
                print("Results object didn't work")

            
        elif args.mode == "Test":
            print("TEST MODE =====================")
            ratio = np.load(os.path.join(args.model_dir,'ratio.npy'))
#            if not os.path.isfile(os.path.join(model_k, 'pred_prob_{}_{}.npy'.format(params.patch_size, params.ovrl))):
            if True:
                # Load moodel
                cl_ind = [x for x in range(params.classes)]

                file_model = os.path.join(model_k,'bestmodel_{}_{}.hdf5'.format(args.dataset, args.exp_id))
                deb.prints(file_model)
                model = load_model(file_model, custom_objects={"f_cat": categorical_focal_loss(depth=np.int(params.classes+1), alpha=[ratio.tolist()],class_indexes=cl_ind),
                                                              #"f_reg": masked_mse(),
                                                              "f_acc": accuracy_mask(),
                                                              "tf": tf})
             
                row,col,bands = image_tr.shape
                cl_img = np.zeros((row,col,params.classes))
                #cl_img = cl_img.astype('float16')
                #reg_img = np.zeros((row,col))
                #reg_img = reg_img.astype('float16')
                plot_sample=False
                
                for m in range(params.patch_size//2,row-params.patch_size//2,stride): 
                    for n in range(params.patch_size//2,col-params.patch_size//2,stride):
                        patch = image_tr[m-params.patch_size//2:m+params.patch_size//2 + params.patch_size%2,
                                  n-params.patch_size//2:n+params.patch_size//2 + params.patch_size%2]
                        patch = patch.reshape((1,params.patch_size,params.patch_size,bands))
                        pred_cl = model.predict(patch)
                        _, x, y, c = pred_cl.shape

                        # plot
                        if plot_sample==True:
                            label_patch = labels_tr[m-params.patch_size//2:m+params.patch_size//2 + params.patch_size%2,
                                    n-params.patch_size//2:n+params.patch_size//2 + params.patch_size%2]

                            mask_patch = labels_tr[m-params.patch_size//2:m+params.patch_size//2 + params.patch_size%2,
                                    n-params.patch_size//2:n+params.patch_size//2 + params.patch_size%2]

                            pred_int = pred_cl.argmax(axis=-1).astype(np.uint8)
                            class_n = len(np.unique(labels_tr))-1
                            #deb.prints(class_n)
                            #deb.prints(np.average(label_patch))
                            if np.average(label_patch)!=class_n:
                                plot_figures_test(patch.astype(np.float32), label_patch, pred_int, pred_cl, model_k, 9, 'test')
                        #pdb.set_trace()
                        cl_img[m-stride//2:m+stride//2,n-stride//2:n+stride//2,:] = pred_cl[0,overlap//2:x-overlap//2,overlap//2:y-overlap//2,:]
                        #reg_img[m-stride//2:m+stride//2,n-stride//2:n+stride//2] = pred_reg[0,overlap//2:x-overlap//2,overlap//2:y-overlap//2,0]
            
            
                cl_img = cl_img[overlap//2:-step_row,overlap//2:-step_col,:]
                labels_tr = labels_tr[overlap//2:-step_row,overlap//2:-step_col]

                print("==== Predictions were applied to the entire image")
                #reg_img = reg_img[overlap//2:-step_row,overlap//2:-step_col]
                pred_save_path = os.path.join(model_k, 'pred_prob_{}_{}_{}'.format(params.patch_size, params.ovrl, args.exp_id))
                np.save(pred_save_path, cl_img)
                print("Saved in ",pred_save_path)
                print(cl_img.min(), np.average(cl_img), cl_img.max())
                deb.prints(cl_img.shape)
                deb.prints(np.unique(labels_tr,return_counts=True))
                deb.prints(np.unique(cl_img.argmax(axis=-1),return_counts=True))
                
                #labels_tr_mask =labels_tr.copy()
                #labels_tr_mask[mask==0]=9
                labels_tr_copy = labels_tr.copy()
                cl_int =cl_img.argmax(axis=-1).astype(np.uint8)

                classes_prediction = np.unique(cl_int)
                tmp_cl_int = cl_int.copy()
                
                classes_labels_unshifted = np.unique(labels_tr)
                labels_shifted = labels_tr - 1
                labels_shifted[labels_shifted==255] = classes_labels_unshifted[-1]
                classes_labels_shifted = np.unique(labels_shifted)
                new_labels2labels = dict((i, c) for i, c in enumerate(classes_labels_shifted))
                deb.prints(new_labels2labels)
                for j in range(len(classes_prediction)):
#                    labels_tr[tmp_tr == classes[j]] = labels2new_labels[classes[j]]
                    cl_int[tmp_cl_int == classes_prediction[j]] = new_labels2labels[classes_prediction[j]]
                deb.prints(np.unique(cl_int,return_counts=True)) 
                print("Adding 1 to classification result")
                cl_int = cl_int + 1
                #t_class = np.unique(labels_tr)[-1]
                #cl_int[cl_int == last_class] = 0
                deb.prints(np.unique(cl_int,return_counts=True))                
                print("Masking classification result")
                
                cl_int[mask==0]=0
                if test_on_train_set==False:
                    cl_int[mask==1]=0
                    labels_tr[mask==1]=0
                else:
                    cl_int[mask==2]=0
                    labels_tr[mask==2]=0
                

                deb.prints(np.unique(cl_int,return_counts=True))                
                deb.prints(np.unique(labels_tr,return_counts=True))

                '''
                print("delete background") 
                
                cl_int_mask = cl_int_mask[labels_tr_mask<bcknd_id]
                labels_tr_mask = labels_tr_mask[labels_tr_mask<bcknd_id]
                deb.prints(np.unique(labels_tr_mask,return_counts=True))
                deb.prints(np.unique(cl_int_mask,return_counts=True))                
                '''
                #pdb.set_trace()
                print("cl_int_mask stats:")
                print(cl_int.min(), np.average(cl_int), cl_int.max())
                
                cv2.imwrite("result.png",cl_int*25)

                cv2.imwrite("result_test.png",cl_int*25)

                cv2.imwrite("result_gt.png",labels_tr*25)
                #pdb.set_trace()

                im = plt.imshow(cl_int, cmap=cmap,vmin=0, vmax=len(np.unique(labels_tr)))
                plt.axis('off')
                set_name = 'test'
                plt.savefig(os.path.join(model_k, set_name + 'predict.png'), dpi = 3000, format='png', bbox_inches = 'tight')
                plt.clf()
                plt.close()

                im = plt.imshow(labels_tr, cmap=cmap,vmin=0, vmax=len(np.unique(labels_tr)))
                plt.axis('off')
                set_name = 'test'
                plt.savefig(os.path.join(model_k, set_name + 'label.png'), dpi = 3000, format='png', bbox_inches = 'tight')
                plt.clf()
                plt.close()


                # metrics
#                cl_flat = cl_int[mask==2]


#                labels_flat = labels_tr_mask.flatten()
#                mask_flat = mask.flatten()
                if test_on_train_set==False:
                    cl_flat = cl_int[mask==2]
                    labels_flat = labels_tr[mask==2] # hadnt i kept only train areas?
                else:
                    cl_flat = cl_int[mask==1]
                    labels_flat = labels_tr[mask==1] # hadnt i kept only train areas?
                
                print("Unique before metrics")
                deb.prints(np.unique(labels_flat,return_counts=True))
                deb.prints(np.unique(cl_flat,return_counts=True))                



                from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
                
                f1 = f1_score(labels_flat,cl_flat,average=None)
                print("f1",f1)
                f1 = f1_score(labels_flat,cl_flat,average='macro')
                print("f1",f1)

                oa = accuracy_score(labels_flat,cl_flat)
                print("oa",oa)
                


                f1 = np.round(f1_score(labels_flat, cl_flat, average=None)*100,2)
                precision = np.round(precision_score(labels_flat, cl_flat, average=None)*100,2)
                recall= np.round(recall_score(labels_flat, cl_flat, average=None)*100,2)


                confusion_matrix = confusion_matrix(labels_flat,cl_flat)
                print(confusion_matrix)
                #update the logs dictionary:
                mean_f1 = np.sum(f1)/len(classes_prediction)

                print(f' — val_f1: {f1}\n — val_precision: {precision}\n — val_recall: {recall}')
                print(f' — mean_f1: {mean_f1}')
                
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
            
            #classes = np.unique(labels)

                
            classes_labels_unshifted = np.unique(labels)
            labels_shifted = labels - 1
            labels_shifted[labels_shifted==255] = classes_labels_unshifted[-1]
            classes_labels_shifted = np.unique(labels_shifted)
            new_labels2labels = dict((i, c) for i, c in enumerate(classes_labels_shifted))

            #new_labels2labels = dict((i, c) for i, c in enumerate(classes))
            
            p,r,f,oa,kapp,aa = metrics(params, labels, model_k, new_labels2labels)
            aa_list.append(aa)
            pre_list.append(p)
            rec_list.append(r)
            f_list.append(f)
            oa_list.append(oa)
            kapp_list.append(kapp)

