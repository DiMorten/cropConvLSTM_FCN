#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:45:05 2020

@author: laura
"""

import numpy as np
from time import time
import numpy as np
import keras.backend as K
import keras
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers import AveragePooling2D, Flatten, BatchNormalization, Dropout
from keras.models import Model
from keras.layers import ELU
from keras import layers
from keras import regularizers
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from keras.callbacks import Callback
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve
from collections import Counter, OrderedDict
from utils import plot_figures
elu_alpha = 0.1





def cnn(pretrained_weights = None, img_shape = (128,128,25),nb_classes=10):
    inputs = Input(shape=img_shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Classification branch
    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop3,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv2,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv1,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    classfier = Conv2D(nb_classes, 1, activation = 'softmax', name='cl_output')(conv8)
    
    
    # Regression branch
#    up61 = Conv2D(24, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#    merge61 = concatenate([drop3,up61], axis = 3)
#    conv61 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge61)
#    conv61 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv61)

#    up71 = Conv2D(24, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv61))
#    merge71 = concatenate([conv2,up71], axis = 3)
#    conv71 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge71)
#    conv71 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv71)

#    up81 = Conv2D(24, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv71))
#    merge81 = concatenate([conv1,up81], axis = 3)
#    conv81 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge81)
#    conv81 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv81)
#    regresor = Conv2D(1, 1, activation = 'sigmoid', name='reg_output')(conv81)


    model = Model(input = inputs, output = [classfier])
    print(model.summary())
    return model

def f1_mean(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.round(K.clip(y_true * y_pred, 0, 1))
        possible_positives = K.round(K.clip(y_true, 0, 1))
        recall_mean = K.sum(true_positives)/(K.sum(possible_positives) + K.epsilon())
        return recall_mean

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.round(K.clip(y_true * y_pred, 0, 1))
        predicted_positives = K.round(K.clip(y_pred, 0, 1))
        precision_mean = K.sum(true_positives)/(K.sum(predicted_positives) + K.epsilon())
        return precision_mean
    
    precision_mean  = precision(y_true, y_pred)
    recall_mean = recall(y_true, y_pred)
    f1_mean = 2*((precision_mean*recall_mean)/(precision_mean+recall_mean+K.epsilon()))
    return f1_mean


class Monitor(Callback):
    def __init__(self, validation, patience, model_dir, classes):   
        super(Monitor, self).__init__()
        self.validation = validation 
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.classes = classes
        self.model_dir = model_dir
        
    def on_train_begin(self, logs={}):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0

        
    def on_epoch_begin(self, epoch, logs={}):        
        self.pred = []
        self.targ = []

     
    def on_epoch_end(self, epoch, logs={}):
        # num = np.random.randint(0,len(self.validation),1)
        for batch_index in range(len(self.validation)):
            val_targ = self.validation[batch_index][1][0]   
            val_pred = self.model.predict(self.validation[batch_index][0])
            val_prob = val_pred[0]
            val_depth = val_pred[1]
            val_predict = np.argmax(val_prob,axis=-1)
            if batch_index == 0:
                plot_figures(self.validation[batch_index][0],val_targ,val_predict,
                             val_prob,val_depth,self.validation[batch_index][1][1],self.model_dir,epoch, 
                             self.classes,'val')
            
            val_predict = val_predict[val_targ<self.classes]
            val_targ = val_targ[val_targ<self.classes]
            self.pred.extend(val_predict)
            self.targ.extend(val_targ)
        
 
        f1 = np.round(f1_score(self.targ, self.pred, average=None)*100,2)
        precision = np.round(precision_score(self.targ, self.pred, average=None)*100,2)
        recall= np.round(recall_score(self.targ, self.pred, average=None)*100,2)
        
        #update the logs dictionary:
        mean_f1 = np.sum(f1)/self.classes
        logs["mean_f1"]=mean_f1

        print(f' — val_f1: {f1}\n — val_precision: {precision}\n — val_recall: {recall}')
        print(f' — mean_f1: {mean_f1}')
        
        current = logs.get("mean_f1")
        if np.less(self.best, current):
            self.best = current
            self.wait = 0
            print("Found best weights at epoch {}".format(epoch + 1))
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping".format(self.stopped_epoch + 1))

