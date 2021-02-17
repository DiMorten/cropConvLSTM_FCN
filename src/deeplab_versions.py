#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:52:18 2020

@author: laura
"""

import numpy as np
from models import cnn
import numpy as np
import argparse
import cv2
from classification_models.keras import Classifiers
import tensorflow as tf
import keras
from keras.models import Model
from keras import layers
from keras.layers import Input, Lambda, Activation, Concatenate, UpSampling2D
from keras.layers import Add, Dropout, BatchNormalization, Conv2D
from keras.layers import DepthwiseConv2D, ZeroPadding2D, GlobalAveragePooling2D
from keras import backend as K
import tensorflow as tf


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation("relu")(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation("relu")(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation("relu")(x)

    return x

def DeepLabVersions(input_shp, params, input_tensor=None):        
            
    print("[INFO] loading {}...".format(params.model))
    if params.model != "custom":
        
#        if not params.weights: weights=None
        weights=None

        network, preprocess_input = Classifiers.get(params.model)
        base_model = network(input_shape=input_shp,
                             weights=weights, 
                             include_top=False)
        
#        base_model.trainable = params.trainable
        base_model.trainable = True

        
        # base_model = Model(inputs=net.input,
        #                     outputs=net.get_layer("stage3_unit1_relu1").output,
        #                     name = "base_model")
        
        # our input image is now represented as a NumPy array of shape
        # (inputShape[0], inputShape[1], 3) however we need to expand the
        # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
        # so we can pass it through the network
        
        if input_tensor is None:
            img_input = Input(shape=input_shp)
        else:
            img_input = input_tensor
        
        if params.channels==3:
            inputs = preprocess_input(img_input)
            inputs = np.expand_dims(img_input, 0)
            
    # end of feature extractor
    atrous_rates = [3,6,9]

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    x = base_model.get_layer("stage2_unit1_relu1").output
    shape_before = K.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = K.int_shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(128, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation("relu", name='aspp0_activation')(b0)

    # rate = 6 (12)
    b1 = SepConv_BN(x, 128, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 128, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 128, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # concatenate ASPP branches & project
    x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(128, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation("relu")(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    # Feature projection
    # x4 (x2) block
    skip_size = K.int_shape(base_model.get_layer("relu0").output)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    skip_size[1:3],
                                                    method='bilinear', align_corners=True))(x)

    dec_skip1 = Conv2D(128, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(base_model.get_layer("relu0").output)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 128, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 128, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

    size_before3 = K.int_shape(img_input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    size_before3[1:3],
                                                    method='bilinear', 
                                                    align_corners=True))(x)
    x = Conv2D(128, (1, 1), padding='same', name="last_conv")(x)
    size_before3 = K.int_shape(img_input)
    x = Dropout(0.65)(x)
    out_class = Conv2D(params.classes, (1, 1), padding='same', name="classification")(x)

    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = input_tensor
    else:
        inputs = img_input

    out_class = Activation("softmax", name='cl_output')(out_class)
        
    if not params.add_reg:

        model = Model(base_model.input, out_class, name='deeplabv3plus')
    
    else:

        x_reg = base_model.get_layer("stage2_unit1_relu1").output
        x_reg = Conv2D(24, (1, 1), padding='same', use_bias=False, name='reg_conv')(x_reg)
        x_reg = BatchNormalization(name='reg_conv_BN', epsilon=1e-5)(x_reg)
        x_reg = Activation("relu")(x)
        size_before3 = tf.keras.backend.int_shape(img_input)
        out_reg = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                        size_before3[1:3],
                                                        method='bilinear', 
                                                        align_corners=True))(out_reg)
        x_reg = Conv2D(24, (1, 1), padding='same', use_bias=False, name='reg_conv_1')(x_reg)
        x_reg = BatchNormalization(name='reg_conv_BN_1', epsilon=1e-5)(x)
        x_reg = Activation("relu")(x_reg)
        x_reg = Conv2D(1, (1, 1), padding='same', use_bias=False, name='conv_out')(x_reg)
        out_reg = Activation('sigmoid', name='reg_output')(out_reg)
        
        model = Model(base_model.inputs, [out_class,out_reg], name='deeplabv3plus')
        
            
    return model            
    
        

            
        
        
        
        