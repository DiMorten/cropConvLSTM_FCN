
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:45:38 2018

@author: laura
"""


import json
import logging
import os
import shutil
import numpy as np
import sys
import errno
from osgeo import gdal
import glob
import multiprocessing
import subprocess, signal
import gc
from sklearn import preprocessing as pp
import joblib
import pandas as pd
from itertools import groupby
from collections import Counter
from sklearn.metrics import accuracy_score, cohen_kappa_score,precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from icecream import ic
plt.rcParams.update({'font.size': 5})

import deb
colormap_list = np.array([[40/255.0, 255/255.0, 40/255.0],
          [166/255.0, 206/255.0, 227/255.0],
          [31/255.0, 120/255.0, 180/255.0],
          [178/255.0, 223/255.0, 138/255.0],
          [51/255.0, 160/255.0, 44/255.0],
          [251/255.0, 154/255.0, 153/255.0],
          [227/255.0, 26/255.0, 28/255.0],
          [253/255.0,191/255.0,111/255.0],
          [255/255.0, 127/255.0, 0/255.0],
          [202/255.0, 178/255.0, 214/255.0],
          [106/255.0, 61/255.0, 154/255.0],
          [255/255.0,255/255.0,153/200.0],
          [255/255.0, 40/255.0, 255/255.0],
          [255/255.0, 146/255.0, 36/255.0],
          [177/255.0, 89/255.0, 40/255.0],
          [255/255.0, 255/255.0, 0/255.0],
          [0/255.0, 0/255.0, 0/255.0]])
custom_colormap = np.array([[255,146,36],
                [255,255,0],
                [164,164,164],
                [255,62,62],
                [0,0,0],
                [172,89,255],
                [0,166,83],
                [40,255,40],
                [187,122,83],
                [217,64,238],
                [0,113,225],
                [128,0,0],
                [114,114,56],
                [53,255,255]])
#custom_colormap = custom_colormap / 255.0                

cmap = ListedColormap(colormap_list)
#cmap = ListedColormap(custom_colormap)

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
    
class Results(object):
    def __init__(self, acc, val_acc, loss, val_loss, filename):
        self.acc = acc
        self.val_acc = val_acc
        self.loss = loss
        self.val_loss = val_loss
        self.filename = filename
        
    def show_result(self):

        f, axarr = plt.subplots(1 , 2)
#        f.set_figheight(15)
        f.set_figwidth(10)

       
        # Accuracia
        axarr[0].plot(self.acc)
        axarr[0].plot(self.val_acc)
        axarr[0].set_title('model accuracy' if 'acc' in self.filename else 'model F1 score')
        axarr[0].set_ylabel('accuracy' if 'acc' in self.filename else 'F1 score')
        axarr[0].set_xlabel('epoch')
        axarr[0].legend(['train', 'valid'], loc='upper left')

        # Função de Perda
        axarr[1].plot(self.loss)
        axarr[1].plot(self.val_loss)
        axarr[1].set_title('model loss')
        axarr[1].set_ylabel('loss')
        axarr[1].set_xlabel('epoch')
        axarr[1].legend(['train', 'valid'], loc='upper left')
                        
        f.savefig(self.filename, dpi = 500, format='png', bbox_inches='tight')
        
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def check_folder(folder_dir):
    '''Create folder if not available
    '''
    if not os.path.exists(folder_dir):
        try:
            os.makedirs(folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
                

def load_image(patch):
    # Read Image
    print ('loading im in path',patch)
    gdal_header = gdal.Open(patch)
    # get array
    img = gdal_header.ReadAsArray()
    return img


def add_padding(img, psize, overl):
    '''Function to padding image
        input:
            patches_size: psize
            stride: stride
            img: image (row,col,bands)
    '''    

    try:
        row, col, bands = img.shape
    except:
        bands = 0
        row, col = img.shape
        
    # Percent of overlap between consecutive patches.
    # The overlap will be multiple of 2
    overlap = int(round(psize * overl))
    overlap -= overlap % 2
    stride = psize - overlap

    # Add Padding to the image to match with the patch size and the overlap
    row += overlap//2
    col += overlap//2
    step_row = (stride - row % stride) % stride
    step_col = (stride - col % stride) % stride
    
    if bands>0:
        npad_img = ((overlap//2, step_row), (overlap//2, step_col),(0,0))
    else:        
        npad_img = ((overlap//2, step_row), (overlap//2, step_col))  

    deb.prints(img.shape)
    deb.prints(np.pad(img, npad_img, mode='symmetric').shape)

    # padd with symetric (espelhado)    
    pad_img = np.pad(img, npad_img, mode='symmetric')
    deb.prints(pad_img.shape)

    # Number of patches: k1xk2
    k1, k2 = (row+step_row)//stride, (col+step_col)//stride
    print('Number of patches: %d' %(k1 * k2))

    return pad_img, stride, step_row, step_col, overlap

def seq_add_padding(img, psize, overl):
    '''Function to padding image
        input:
            patches_size: psize
            stride: stride
            img: image (row,col,bands)
    '''    

    try:
        t_len, row, col, bands = img.shape
    except:
        bands = 0
        t_len, row, col = img.shape
        
    # Percent of overlap between consecutive patches.
    # The overlap will be multiple of 2
    overlap = int(round(psize * overl))
    overlap -= overlap % 2
    stride = psize - overlap

    # Add Padding to the image to match with the patch size and the overlap
    row += overlap//2
    col += overlap//2
    step_row = (stride - row % stride) % stride
    step_col = (stride - col % stride) % stride
    
    if bands>0:
        npad_img = ((overlap//2, step_row), (overlap//2, step_col),(0,0))
    else:        
        npad_img = ((overlap//2, step_row), (overlap//2, step_col))  
    
    pad_img_shape = np.pad(img[0], npad_img, mode='symmetric').shape
    pad_img = np.zeros((t_len,) + pad_img_shape)
    deb.prints(pad_img_shape)
    deb.prints(pad_img.shape)
    
    # padd with symetric (espelhado)    
    for t_step in range(t_len):
        deb.prints(img[t_step].shape)
        deb.prints(np.pad(img[t_step], npad_img, mode='symmetric').shape)
        
        pad_img[t_step] = np.pad(img[t_step], npad_img, mode='symmetric')

    # Number of patches: k1xk2
    k1, k2 = (row+step_row)//stride, (col+step_col)//stride
    print('Number of patches: %d' %(k1 * k2))

    return pad_img, stride, step_row, step_col, overlap


def balance_coords(img_gt, class_list, samples_per_class, random = True):
    '''Function to balance coords for data augmentation
        input:
            labels: class for each sample

    '''

    # add padding to gt raster
    coords = np.where(img_gt!=0)    
    # get classes
    labels = img_gt[img_gt!=0]
    classes = np.unique(labels)
        
    num_total_samples = len(class_list)*samples_per_class
    
    coordsx_tr = np.zeros(num_total_samples, dtype='int')
    coordsy_tr = np.zeros(num_total_samples, dtype='int')
    
    k = 0
    for clss in classes:
        if clss in class_list:
            # get total samples of class = clss
            clss_labels = labels[labels == clss].copy()
            index = np.where(labels == clss)
            num_samples = len(clss_labels)
            
            if num_samples > samples_per_class:
                # if num_samples > samples_per_class choose samples randomly
                index = np.random.choice(index[0], samples_per_class, replace=False)
           
            else:
                index = np.random.choice(index[0], samples_per_class, replace=True)
                             
            
            coordsx_tr[k*samples_per_class:(k+1)*samples_per_class] = coords[0][index]
            coordsy_tr[k*samples_per_class:(k+1)*samples_per_class] = coords[1][index]
            
            k += 1
        
    # Permute samples randomly
    if random:
        idx = np.random.permutation(num_total_samples)
        coordsx_tr = coordsx_tr[idx]
        coordsy_tr = coordsy_tr[idx]

    return np.array([coordsx_tr, coordsy_tr])

def metrics(params, label, model_dir_k,new_labels2labels):
    
#    pred_prob1 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.1.npy'))
#    pred_prob2 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3.npy'))
#    pred_prob3 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.5.npy'))

#    pred_prob1 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3_fifth_moreval.npy'))
#    pred_prob2 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3_fifth_moreval.npy'))
#    pred_prob3 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3_fifth_moreval.npy'))


#    pred_prob1 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3_seventh_moreval.npy'))
#    pred_prob2 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3_seventh_moreval.npy'))
#    pred_prob3 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3_seventh_moreval.npy'))

    pred_prob1 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3_deeplab_rep1.npy'))
    pred_prob2 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3_deeplab_rep1.npy'))
    pred_prob3 = np.load(os.path.join(model_dir_k, 'pred_prob_128_0.3_deeplab_rep1.npy'))

    prob_mean = (pred_prob1+pred_prob2+pred_prob3)/3
    predict_mask_test = np.argmax(pred_prob3, axis = -1)
    prob_mean = []
            
    # evaluate test  
    print ('*' * 50)
    print ('Classification done!!!!')
    print ('*' * 50)    


    lbl_tmp = predict_mask_test.copy()
    classes_pred = np.unique(predict_mask_test)
    for j in range(len(classes_pred)):
        predict_mask_test[lbl_tmp == classes_pred[j]] = new_labels2labels[classes_pred[j]]      
    
    predict_mask_test = predict_mask_test+1
    
    # get classification report
    coords = np.where(label!=0)
    cohen_score = cohen_kappa_score(label[coords], predict_mask_test[coords])
    acc = accuracy_score(label[coords], predict_mask_test[coords])
    
    # save report in csv
    clf_rep = precision_recall_fscore_support(label[coords], predict_mask_test[coords])
    out_dict = {
                              "precision" :clf_rep[0].round(4)
                            ,"recall" : clf_rep[1].round(4)
                            ,"f1-score" : clf_rep[2].round(4)
                            ,"support" : clf_rep[3]                    
                            }
    
    out_df = pd.DataFrame(out_dict, index = np.unique(label[coords]))
    avg_tot = (out_df.apply(lambda x: round(x.mean(), 2) if x.name!="support" else  round(x.sum(), 2)).to_frame().T)
    oa_acc = (out_df.apply(lambda x: round(acc, 4) if x.name=="precision" else  round(x.sum()*0, 1)).to_frame().T)
    kappa = (out_df.apply(lambda x: round(cohen_score, 4) if x.name=="precision" else  round(x.sum()*0, 1)).to_frame().T)
    avg_tot.index = ["avg/total"]
    oa_acc.index = ['OA']
    kappa.index = ['Kappa']
    out_df = out_df.append(avg_tot)
    out_df = out_df.append(oa_acc)
    out_df = out_df.append(kappa)
    
    out_df.to_csv(os.path.join(model_dir_k,'val_report_' + str(params.patch_size) + '.csv'), sep='\t')
    
    y_true = pd.Series(label[coords].tolist())
    y_pred = pd.Series(predict_mask_test[coords].tolist())
    df_confusion = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    df_confusion.to_csv(os.path.join(model_dir_k,'val_conf_matrix_' + str(params.patch_size) + '.csv'), index_label = 'True|Predicted', sep='\t')
    
    classes = np.unique(label[coords])
    cm = confusion_matrix(label[coords], predict_mask_test[coords], labels =classes, normalize='true')
    avr_acc = round(np.sum(cm.diagonal())/len(classes)*100,2)
    cm = np.round(cm,2)
    
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cbar=0, cmap="YlGnBu", linewidths=.5, linecolor= 'black'); #annot=True to annotate cells        
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Average class accuracy: ' + str(avr_acc)) 
    ax.xaxis.set_ticklabels([str(i) for i in classes]); ax.yaxis.set_ticklabels([str(i) for i in classes])
    plt.savefig(os.path.join(model_dir_k,'conf_mat_' + str(params.patch_size) + '.png'), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    
    return clf_rep[0].round(4), clf_rep[1].round(4), clf_rep[2].round(4), acc, cohen_score, cm.diagonal()

def extract_patches_coord(img_gt, psize, stride, train = False):
    '''Function to extract patches coordinates from rater images
        input:
            img: raster image 
            gt: shpafile raster
            psize: image patch size
            ovrl: overlap to extract patches
            model: model type

    '''
    # add padding to gt raster
    row,col = img_gt.shape
    
    # get classes
    labels = img_gt[img_gt!=0]
    unique_class = np.unique(labels)

    # loop over x,y coordinates and extract patches
    coord_list = list()
    classes = list()

    if train:
        for m in range(psize//2,row-psize//2,stride): 
            for n in range(psize//2,col-psize//2,stride):
                coord = [m,n]
                class_patch = img_gt[m,n]

                if class_patch in unique_class: # if any of class_patch in unique_class
                    coord_list.append(coord)                    
                    classes.append(class_patch)
                            
                elif class_patch == 0: 
                    lab_p = img_gt[coord[0]-psize//2:coord[0]+psize//2 + psize%2,coord[1]-psize//2:coord[1]+psize//2 + psize%2]
                    no_class = np.sum(lab_p>0)
                    if no_class>0.05*psize**2: # >0.05*psize**2*t_len
                        coord_list.append(coord)
                        cnt = Counter(lab_p[lab_p>0])
                        classes.append(max(cnt, key=cnt.get))
                                        
                 
        count_cl = Counter(classes)              
        samples_per_class = int(round(np.max(list(count_cl.values()))))
        num_total_samples = len(np.unique(classes))*samples_per_class
        coordsx_tr = np.zeros((num_total_samples,2), dtype='int')
        labels_tr = np.zeros(num_total_samples, dtype='int')
        
        k = 0
        coord_list = np.array(coord_list)
        classes = np.array(classes)
        for key in count_cl.keys():
            # get total samples per class
            index = np.where(classes == key)
            num_samples = count_cl[key]
            
            if num_samples >= samples_per_class:
                # if num_samples > samples_per_class choose samples randomly
                index = np.random.choice(index[0], samples_per_class, replace=False)
           
            else:
                index = np.random.choice(index[0], samples_per_class, replace=True)
                             
            
            coordsx_tr[k*samples_per_class:(k+1)*samples_per_class,:] = coord_list[index,:]
            labels_tr[k*samples_per_class:(k+1)*samples_per_class] = classes[index]
            k += 1
    
        # Permute samples randomly
        idx = np.random.permutation(num_total_samples)
        coordsx_tr = coordsx_tr[idx,:]
        labels_tr = labels_tr[idx]    

    else:
        cont = 0
        for m in range(psize//2,row-psize//2,stride): 
            for n in range(psize//2,col-psize//2,stride):
                coord = [m,n]
                class_patch = img_gt[m,n]
                if class_patch in unique_class:
                    coord_list.append(coord)                    
                    classes.append(class_patch)
                    
                elif class_patch == 0 and cont%4 == 0:
                    lab_p = img_gt[coord[0]-psize//2:coord[0]+psize//2 + psize%2,coord[1]-psize//2:coord[1]+psize//2 + psize%2]
                    no_class = np.sum(lab_p>0)
                    if no_class>0.05*psize**2:
                        coord_list.append(coord)
                        cnt = Counter(lab_p[lab_p>0])
                        classes.append(max(cnt, key=cnt.get))
                        
                cont+=1

                            
        coordsx_tr = np.array(coord_list)
        labels_tr = np.array(classes)
    
    return coordsx_tr
    
    
    
#def plot_figures(img,cl,pred,prob,pred_d,d_map,model_dir,epoch, nb_classes,set_name):
'''
plot_figures(self.validation[batch_index][0],val_targ,val_predict,
                             val_prob,self.model_dir,epoch, 
                             self.classes,'val')
'''
def plot_figures(img,cl,pred,prob,model_dir,epoch, nb_classes,set_name):

    '''
        img: input
        cl: val label
        pred: val predict
        prob: val prob

        
    '''
    batch = 8 # columns 8
#    nrows = 6
    nrows = 4

    img = img[:8,:,:,-1]
    cl = cl[:8,:,:]
    pred = pred[:8,:,:]
    prob = np.amax(prob[:8,:,:],axis=-1)
#        pred_d = pred_d[:8,:,:,0]
#        d_map = d_map[:8,:,:,0]
                        
    fig, axes = plt.subplots(nrows=nrows, ncols=batch, figsize=(9, 6))
    
#    imgs = [img,cl,pred,prob,d_map,pred_d]
    imgs = [img,cl,pred,prob]
     
    cont = 0
    cont_img = 0
    cont_bacth = 0
    print("dtype img,cl,pred,prob", img.dtype,cl.dtype,pred.dtype,prob.dtype)
    print("shape img,cl,pred,prob", img.shape,cl.shape,pred.shape,prob.shape)

    for ax in axes.flat:
        ax.set_axis_off()
        if cont_img < batch:
#            im = ax.imshow(imgs[cont][cont_bacth], cmap = 'gray')
            im = ax.imshow(imgs[cont][cont_bacth], cmap=cmap,vmin=0, vmax=nb_classes)

        elif cont_img >= batch and cont_img < 3*batch:
            im = ax.imshow(imgs[cont][cont_bacth], cmap=cmap,vmin=0, vmax=nb_classes)
        elif cont_img >= 3*batch and cont_img < 4*batch:
            im = ax.imshow(imgs[cont][cont_bacth], cmap='OrRd', interpolation='nearest')
        elif cont_img >= 4*batch:
            im = ax.imshow(imgs[cont][cont_bacth], cmap='winter', interpolation='nearest')

        cont_img+=1
        cont_bacth+=1
        if cont_img%batch==0:
            cont+=1
            cont_bacth=0

    
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    # cbar = fig.colorbar(im, cax=cb_ax)
    
    # set the colorbar ticks and tick labels
    # cbar.set_ticks(np.arange(0, 1, nb_classes))
    
    plt.axis('off')
    plt.savefig(os.path.join(model_dir, set_name + str(epoch) + '.png'), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()

def plot_figures_timedistributed(img,cl,pred,prob,model_dir,epoch, nb_classes,set_name):

    '''
        img: input
        cl: val label
        pred: val predict
        prob: val prob

        
    '''
    batch = 8 # columns 8
#    nrows = 6
    nrows = 5

#    img = img[:8,-1,:,:,0]

    img2 = img[:8,-1,:,:,0].copy()

    img = img[:8,0,:,:,0]
    ic(np.unique(img2, return_counts = True))
    ic(np.unique(img, return_counts = True))

    cl = cl[:8,:,:]
    pred = pred[:8,:,:]
    prob = np.amax(prob[:8,:,:],axis=-1)
#        pred_d = pred_d[:8,:,:,0]
#        d_map = d_map[:8,:,:,0]
                        
    fig, axes = plt.subplots(nrows=nrows, ncols=batch, figsize=(9, 6))
    
#    imgs = [img,cl,pred,prob,d_map,pred_d]
    imgs = [img,img,cl,pred,prob]
     
    cont = 0
    cont_img = 0
    cont_bacth = 0
    print("dtype img,cl,pred,prob", img.dtype,cl.dtype,pred.dtype,prob.dtype)
    print("shape img,cl,pred,prob", img.shape,cl.shape,pred.shape,prob.shape)

    for ax in axes.flat:
        ax.set_axis_off()
        if cont_img < batch:
#            im = ax.imshow(imgs[cont][cont_bacth], cmap = 'gray')
            im = ax.imshow(imgs[cont][cont_bacth], cmap=cmap,vmin=0, vmax=nb_classes)

        elif cont_img >= batch and cont_img < 4*batch:
            im = ax.imshow(imgs[cont][cont_bacth], cmap=cmap,vmin=0, vmax=nb_classes)
        elif cont_img >= 4*batch and cont_img < 5*batch:
            im = ax.imshow(imgs[cont][cont_bacth], cmap='OrRd', interpolation='nearest')
        elif cont_img >= 5*batch:
            im = ax.imshow(imgs[cont][cont_bacth], cmap='winter', interpolation='nearest')

        cont_img+=1
        cont_bacth+=1
        if cont_img%batch==0:
            cont+=1
            cont_bacth=0

    
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    # cbar = fig.colorbar(im, cax=cb_ax)
    
    # set the colorbar ticks and tick labels
    # cbar.set_ticks(np.arange(0, 1, nb_classes))
    
    plt.axis('off')
    plt.savefig(os.path.join(model_dir, set_name + str(epoch) + '.png'), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()
def plot_figures_test(img,cl,pred,prob,model_dir,nb_classes,set_name):

    '''
        img: input
        cl: val label
        pred: val predict
        prob: val prob

        
    '''
    print("shape img,cl,pred,prob",img.shape, cl.shape, pred.shape, prob.shape)

    batch = 1 # columns 8
#    nrows = 6
    nrows = 4
    deb.prints(img.shape)
#    pdb.set_trace()
    img = img[0,-1,:,:,0]
    #cl = cl[0,:,:]
    pred = pred[0,:,:]
    prob = np.amax(prob[0,:,:],axis=-1)
    deb.prints(img.shape)
    #pdb.set_trace()
#    print("dtype img,cl,pred,prob",img.dtype, cl.dtype, pred.dtype, prob.dtype) #float16, 

#        pred_d = pred_d[:8,:,:,0]
#        d_map = d_map[:8,:,:,0]
                        
    fig, axes = plt.subplots(nrows=nrows, ncols=batch, figsize=(9, 6))
    
#    imgs = [img,cl,pred,prob,d_map,pred_d]
    imgs = [img,cl,pred,prob]
     
    cont = 0
    cont_img = 0
    cont_bacth = 0
    print("dtype img,cl,pred,prob", img.dtype,cl.dtype,pred.dtype,prob.dtype)
    print("shape img,cl,pred,prob", img.shape,cl.shape,pred.shape,prob.shape)

    for ax in axes.flat:
        ax.set_axis_off()
        if cont_img == 0:
            im = ax.imshow(imgs[cont_img], cmap = 'gray')
#            im = ax.imshow(imgs[cont_img], cmap=cmap,vmin=0, vmax=nb_classes)

        elif cont_img == 1 or cont_img == 2:
            im = ax.imshow(imgs[cont_img], cmap=cmap,vmin=0, vmax=nb_classes)
        elif cont_img == 3:
            im = ax.imshow(imgs[cont_img], cmap='OrRd', interpolation='nearest')
#        elif cont_img == 3:
#            im = ax.imshow(imgs[cont_img], cmap='winter', interpolation='nearest')

        cont_img+=1
        cont_bacth+=1
        
        #if cont_img%batch==0:
        #    cont+=1
        #    cont_bacth=0
    #pdb.set_trace()
    
    #fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
    #                    wspace=0.02, hspace=0.02)
    
    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    # cbar = fig.colorbar(im, cax=cb_ax)
    
    # set the colorbar ticks and tick labels
    # cbar.set_ticks(np.arange(0, 1, nb_classes))
    plt.show()
    
#    plt.axis('off')
#    plt.savefig(os.path.join(model_dir, set_name + '.png'), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()


def plot_figures_test_timedistributed(img,cl,pred,prob,model_dir,nb_classes,set_name):

    '''
        img: input
        cl: val label
        pred: val predict
        prob: val prob

        
    '''
    print("shape img,cl,pred,prob",img.shape, cl.shape, pred.shape, prob.shape)

    batch = 1 # columns 8
#    nrows = 6
    nrows = 4
    deb.prints(img.shape)
#    pdb.set_trace()
    img = img[0,:,:,0]
    #cl = cl[0,:,:]
    pred = pred[0,:,:]
    prob = np.amax(prob[0,:,:],axis=-1)
    deb.prints(img.shape)
    #pdb.set_trace()
#    print("dtype img,cl,pred,prob",img.dtype, cl.dtype, pred.dtype, prob.dtype) #float16, 

#        pred_d = pred_d[:8,:,:,0]
#        d_map = d_map[:8,:,:,0]
                        
    fig, axes = plt.subplots(nrows=nrows, ncols=batch, figsize=(9, 6))
    
#    imgs = [img,cl,pred,prob,d_map,pred_d]
    imgs = [img,cl,pred,prob]
     
    cont = 0
    cont_img = 0
    cont_bacth = 0
    print("dtype img,cl,pred,prob", img.dtype,cl.dtype,pred.dtype,prob.dtype)
    print("shape img,cl,pred,prob", img.shape,cl.shape,pred.shape,prob.shape)

    for ax in axes.flat:
        ax.set_axis_off()
        if cont_img == 0:
            im = ax.imshow(imgs[cont_img], cmap = 'gray')
#            im = ax.imshow(imgs[cont_img], cmap=cmap,vmin=0, vmax=nb_classes)
        elif cont_img == 1 or cont_img == 2:
            im = ax.imshow(imgs[cont_img], cmap=cmap,vmin=0, vmax=nb_classes)
        elif cont_img == 3:
            im = ax.imshow(imgs[cont_img], cmap='OrRd', interpolation='nearest')
#        elif cont_img == 3:
#            im = ax.imshow(imgs[cont_img], cmap='winter', interpolation='nearest')

        cont_img+=1
        cont_bacth+=1
        
        #if cont_img%batch==0:
        #    cont+=1
        #    cont_bacth=0
    #pdb.set_trace()
    
    #fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
    #                    wspace=0.02, hspace=0.02)
    
    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    # cbar = fig.colorbar(im, cax=cb_ax)
    
    # set the colorbar ticks and tick labels
    # cbar.set_ticks(np.arange(0, 1, nb_classes))
    plt.show()
    
#    plt.axis('off')
#    plt.savefig(os.path.join(model_dir, set_name + '.png'), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()