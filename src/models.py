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
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv3D, Conv3DTranspose, AveragePooling3D
from keras.layers import AveragePooling2D, Flatten, BatchNormalization, Dropout, TimeDistributed, ConvLSTM2D
from keras.models import Model
from keras.layers import ELU, Lambda
from keras import layers
from keras import regularizers
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from keras.callbacks import Callback
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve
from collections import Counter, OrderedDict
from utils import plot_figures, plot_figures_timedistributed
import pdb
from keras.regularizers import l1,l2
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, AveragePooling2D, Bidirectional, Activation

elu_alpha = 0.1

import deb



def cnn(pretrained_weights = None, img_shape = (128,128,25),nb_classes=10):
    #img_shape = (32,32,2)
    inputs = Input(shape=img_shape)
    fs = 16
    conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
#    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv5 = Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv5 = Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#    drop5 = Dropout(0.5)(conv5)
    
    # Classification branch
    up6 = Conv2D(fs*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv3,up6], axis = 3)
    conv6 = Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(fs*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv2,up7], axis = 3)
    conv7 = Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(fs, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv1,up8], axis = 3)
    conv8 = Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    classfier = Conv2D(nb_classes, 1, activation = 'softmax', name='cl_output')(conv8)
    
    model = Model(inputs = inputs, outputs = [classfier])
    print(model.summary())
    return model


def cnn_t(pretrained_weights = None, img_shape = (128,128,25),nb_classes=10):
    img_shape = (14,32,32,2)
    inputs = Input(shape=img_shape)
    fs = 16
    conv1 = TimeDistributed(Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(inputs)
    conv1 = TimeDistributed(Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = TimeDistributed(Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool1)
    conv2 = TimeDistributed(Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    conv3 = TimeDistributed(Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool2)
    conv3 = TimeDistributed(Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv3)
#    drop3 = Dropout(0.5)(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv5 = TimeDistributed(Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool3)
    conv5 = TimeDistributed(Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv5)
#    drop5 = Dropout(0.5)(conv5)
    conv5 = Lambda(lambda n: n[:,-1])(conv5)
    # Classification branch
    up6 = Conv2D(fs*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    conv3 = Lambda(lambda n: n[:,-1])(conv3)
    merge6 = concatenate([conv3,up6], axis = -1)
    conv6 = Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(fs*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    conv2 = Lambda(lambda n: n[:,-1])(conv2)
    merge7 = concatenate([conv2,up7], axis = -1)
    conv7 = Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(fs, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    conv1 = Lambda(lambda n: n[:,-1])(conv1)
    merge8 = concatenate([conv1,up8], axis = -1)
    conv8 = Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    classfier = Conv2D(nb_classes, 1, activation = 'softmax', name='cl_output')(conv8)
    
    model = Model(inputs = inputs, outputs = [classfier])
    print(model.summary())
    return model


def cnn_rnn_t(pretrained_weights = None, img_shape = (128,128,25),nb_classes=10):
    #img_shape = (14,32,32,2)
    inputs = Input(shape=img_shape)
    fs = 16
    conv1 = TimeDistributed(Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(inputs)
    conv1 = TimeDistributed(Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = TimeDistributed(Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool1)
    conv2 = TimeDistributed(Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    conv3 = TimeDistributed(Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool2)
    conv3 = TimeDistributed(Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv3)
#    drop3 = Dropout(0.5)(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

    conv5 = TimeDistributed(Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool3)
    conv5 = TimeDistributed(Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv5)
#    drop5 = Dropout(0.5)(conv5)
#    conv5 = Lambda(lambda n: n[:,-1])(conv5)
    conv5 = ConvLSTM2D(64,3,return_sequences=False,
                padding="same")(conv5)
    # Classification branch
    up6 = Conv2D(fs*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    conv3 = Lambda(lambda n: n[:,-1])(conv3)
    merge6 = concatenate([conv3,up6], axis = -1)
    conv6 = Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(fs*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(fs*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    conv2 = Lambda(lambda n: n[:,-1])(conv2)
    merge7 = concatenate([conv2,up7], axis = -1)
    conv7 = Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(fs*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(fs, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    conv1 = Lambda(lambda n: n[:,-1])(conv1)
    merge8 = concatenate([conv1,up8], axis = -1)
    conv8 = Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(fs, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    classfier = Conv2D(nb_classes, 1, activation = 'softmax', name='cl_output')(conv8)
    
    model = Model(inputs = inputs, outputs = [classfier])
    print(model.summary())
    return model


def slice_tensor(x,output_shape):
    deb.prints(output_shape)
    deb.prints(K.int_shape(x))
#				res1 = Lambda(lambda x: x[:,:,:,-1], output_shape=output_shape)(x)
#				res2 = Lambda(lambda x: x[:,:,:,-1], output_shape=output_shape[1:])(x)
    res2 = Lambda(lambda x: x[:,-1])(x)

#				deb.prints(K.int_shape(res1))
    deb.prints(K.int_shape(res2))
    
    return res2
def UUnetConvLSTM(pretrained_weights = None, img_shape = (14,128,128,2),nb_classes=10):
    inputs = Input(shape=img_shape)
    conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(inputs)
    conv1 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool1)
    conv2 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    conv3 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool2)
    conv3 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(drop3)

    conv5 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool3)
    conv5 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv5)
    drop5 = Dropout(0.5)(conv5)
    drop5 = ConvLSTM2D(128,3,return_sequences=False,
					padding="same")(drop5)
    # Classification branch
    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    drop3 = slice_tensor(drop3, output_shape = K.int_shape(up6))
    merge6 = concatenate([drop3,up6], axis = 3)
    #merge6 = up6
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    conv2 = slice_tensor(conv2, output_shape = K.int_shape(up7))
    merge7 = concatenate([conv2,up7], axis = 3)
    #merge7 = up7
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    conv1 = slice_tensor(conv1, output_shape = K.int_shape(up8))
    merge8 = concatenate([conv1,up8], axis = 3)
    #merge8 = up8
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    classfier = Conv2D(nb_classes, 1, activation = 'softmax', name='cl_output')(conv8)
    


    model = Model(inputs = inputs, outputs = [classfier])
    print(model.summary())
    return model



weight_decay=1E-4
def dilated_layer(x,filter_size,dilation_rate=1, kernel_size=3):
    x = TimeDistributed(Conv2D(filter_size, kernel_size, padding='same',
        dilation_rate=(dilation_rate, dilation_rate)))(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                        beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    return x
def transpose_layer(x,filter_size,dilation_rate=1,
    kernel_size=3, strides=(2,2)):
    x = Conv2DTranspose(filter_size, 
        kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                                        beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    return x
def dilated_layer_Nto1(x,filter_size,dilation_rate=1, kernel_size=3):
    x = Conv2D(filter_size, kernel_size, padding='same',
        dilation_rate=(dilation_rate, dilation_rate))(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                        beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    return x

def BUnet4ConvLSTM(img_shape = (14,128,128,2),class_n=10):
    in_im = Input(shape=img_shape)

    concat_axis = 3

    #fs=32
    fs=16

    p1=dilated_layer(in_im,fs)			
    p1=dilated_layer(p1,fs)
    e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
    p2=dilated_layer(e1,fs*2)
    e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
    p3=dilated_layer(e2,fs*4)
    e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

    x = Bidirectional(ConvLSTM2D(128,3,return_sequences=False,
            padding="same"),merge_mode='concat')(e3)

    d3 = transpose_layer(x,fs*4)
    d3 = keras.layers.concatenate([d3, p3], axis=-1)
    d3=dilated_layer_Nto1(d3,fs*4)
    d2 = transpose_layer(d3,fs*2)
    d2 = keras.layers.concatenate([d2, p2], axis=-1)
    d2=dilated_layer_Nto1(d2,fs*2)
    d1 = transpose_layer(d2,fs)
    d1 = keras.layers.concatenate([d1, p1], axis=-1)
    out=dilated_layer_Nto1(d1,fs)
    out = Conv2D(class_n, (1, 1), activation=None,
                                padding='same', name='cl_output')(out)
    model = Model(in_im, out)
    print(model.summary())
    return model

def simplefcn(img_shape = (128,128,25),class_n=10):
    in_im = Input(shape=img_shape)
    x = Conv2D(30, 3, padding='same')(in_im)
    out = Conv2D(class_n, (1, 1), activation=None,
                                padding='same', name='cl_output')(x)
    model = Model(in_im, out)
    print(model.summary())
    return model

def simplefcn_t(img_shape = (14,128,128,2),class_n=10):
    in_im = Input(shape=img_shape[1:])
    x = Conv2D(30, 3, padding='same')(in_im)
    out = Conv2D(class_n, (1, 1), activation=None,
                                padding='same', name='cl_output')(x)
    model = Model(in_im, out)
    print(model.summary())
    return model
def UUnet4ConvLSTM(img_shape = (14,128,128,2),class_n=10):
    in_im = Input(shape=img_shape)
    concat_axis = 3

    #fs=32
    fs=16

    p1=dilated_layer(in_im,fs)			
    p1=dilated_layer(p1,fs)
    e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
    p2=dilated_layer(e1,fs*2)
    e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
    p3=dilated_layer(e2,fs*4)
    e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

    x = ConvLSTM2D(128,3,return_sequences=False,
            padding="same")(e3)

    d3 = transpose_layer(x,fs*4)
    p3 = slice_tensor(p3, output_shape = K.int_shape(d3))
    deb.prints(K.int_shape(p3))
    deb.prints(K.int_shape(d3))
    
    d3 = keras.layers.concatenate([d3, p3], axis=-1)
    d3=dilated_layer_Nto1(d3,fs*4)
    d2 = transpose_layer(d3,fs*2)
    p2 = slice_tensor(p2, output_shape = K.int_shape(d2))
    deb.prints(K.int_shape(p2))
    deb.prints(K.int_shape(d2))

    d2 = keras.layers.concatenate([d2, p2], axis=-1)
    d2=dilated_layer_Nto1(d2,fs*2)
    d1 = transpose_layer(d2,fs)
    p1 = slice_tensor(p1, output_shape = K.int_shape(d1))
    deb.prints(K.int_shape(p1))
    deb.prints(K.int_shape(d1))

    d1 = keras.layers.concatenate([d1, p1], axis=-1)
    out=dilated_layer_Nto1(d1,fs)
    out = Conv2D(class_n, (1, 1), activation=None,
                                padding='same', name='cl_output')(out)
    model = Model(in_im, out)
    print(model.summary())
    return model


def dilated_layer_3D(x,filter_size,dilation_rate=1, kernel_size=3):
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate, dilation_rate, dilation_rate)
    x = Conv3D(filter_size, kernel_size, padding='same',
        dilation_rate=dilation_rate)(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                        beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    return x


def transpose_layer_3D(x,filter_size,dilation_rate=1,
    kernel_size=3, strides=(1,2,2)):
    x = Conv3DTranspose(filter_size,
        kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay),
                                        beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    return x	

def UUnet3DConvLSTM(img_shape = (14,128,128,2),class_n=10):
    in_im = Input(shape=img_shape)
    concat_axis = 3

    #fs=32
    fs=16


    p1=dilated_layer_3D(in_im,fs,kernel_size=(7,3,3))
    e1 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p1)
    p2=dilated_layer_3D(e1,fs*2,kernel_size=(7,3,3))
    e2 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p2)
    p3=dilated_layer_3D(e2,fs*4,kernel_size=(7,3,3))
    e3 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p3)

    x = ConvLSTM2D(256,3,return_sequences=False,
            padding="same")(e3)
#    x = Lambda(lambda n: n[:,-1])(e3)
    
    d3 = transpose_layer(x,fs*4)
    p3 = slice_tensor(p3, output_shape = K.int_shape(d3))
    deb.prints(K.int_shape(p3))
    deb.prints(K.int_shape(d3))
    
    d3 = keras.layers.concatenate([d3, p3], axis=-1)
    d3=dilated_layer_Nto1(d3,fs*4)
    d2 = transpose_layer(d3,fs*2)
    p2 = slice_tensor(p2, output_shape = K.int_shape(d2))
    deb.prints(K.int_shape(p2))
    deb.prints(K.int_shape(d2))

    d2 = keras.layers.concatenate([d2, p2], axis=-1)
    d2=dilated_layer_Nto1(d2,fs*2)
    d1 = transpose_layer(d2,fs)
    p1 = slice_tensor(p1, output_shape = K.int_shape(d1))
    deb.prints(K.int_shape(p1))
    deb.prints(K.int_shape(d1))

    d1 = keras.layers.concatenate([d1, p1], axis=-1)
    out=dilated_layer_Nto1(d1,fs)
    out = Conv2D(class_n, (1, 1), activation=None,
                                padding='same', name='cl_output')(out)
    model = Model(in_im, out)
    print(model.summary())
    return model
def Unet3D(img_shape = (14,128,128,2),class_n=10):
    in_im = Input(shape=img_shape)

    #fs=32
    fs=16

    p1=dilated_layer_3D(in_im,fs,kernel_size=(7,3,3))
    e1 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p1)
    p2=dilated_layer_3D(e1,fs*2,kernel_size=(7,3,3))
    e2 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p2)
    p3=dilated_layer_3D(e2,fs*4,kernel_size=(7,3,3))
    e3 = AveragePooling3D((1, 2, 2), strides=(1, 2, 2))(p3)

    d3 = transpose_layer_3D(e3,fs*4)
    d3 = keras.layers.concatenate([d3, p3], axis=4)
    d3 = dilated_layer_3D(d3,fs*4,kernel_size=(7,3,3))
    d2 = transpose_layer_3D(d3,fs*2)
    d2 = keras.layers.concatenate([d2, p2], axis=4)
    d2 = dilated_layer_3D(d2,fs*2,kernel_size=(7,3,3))
    d1 = transpose_layer_3D(d2,fs)
    d1 = keras.layers.concatenate([d1, p1], axis=4)
    out = dilated_layer_3D(d1,fs,kernel_size=(7,3,3))
    out = Conv3D(self.class_n, (1, 1, 1), activation=None,
                                padding='same')(out)
    self.graph = Model(in_im, out)
    print(self.graph.summary())


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
        self.f1_history = []
        self.oa_history = []
        
        
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
        #deb.prints(range(len(self.validation)))
        # num = np.random.randint(0,len(self.validation),1)
        for batch_index in range(len(self.validation)):
            val_targ = self.validation[batch_index][1][0]   
            val_pred = self.model.predict(self.validation[batch_index][0])
##            deb.prints(val_pred.shape) # was programmed to get two outputs> classif. and depth
##            deb.prints(val_targ.shape) # was programmed to get two outputs> classif. and depth
##            deb.prints(len(self.validation[batch_index][1])) # was programmed to get two outputs> classif. and depth

            # val_prob = val_pred[0]
            val_prob = val_pred.copy()
            # val_depth = val_pred[1]
            val_predict = np.argmax(val_prob,axis=-1)
            if batch_index == 0:
                #plot_figures(self.validation[batch_index][0],val_targ,val_predict,
                #             val_prob,self.model_dir,epoch, 
                #             self.classes,'val')
                #plot_figures_timedistributed(self.validation[batch_index][0],val_targ,val_predict,
                #             val_prob,self.model_dir,epoch, 
                #             self.classes,'val')
                pass
            val_targ = np.squeeze(val_targ)
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

        self.f1_history.append(mean_f1)
        
        print(f' — val_f1: {f1}\n — val_precision: {precision}\n — val_recall: {recall}')
        print(f' — mean_f1: {mean_f1}')

        oa = np.round(accuracy_score(self.targ, self.pred)*100,2)
        print("oa",oa)        
        self.oa_history.append(oa)

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
        print("f1 history",self.f1_history)
        print("oa history",self.oa_history)
        

