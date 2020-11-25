#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:08:43 2020

@author: laura
"""

import numpy as np
import tensorflow as tf
from keras import backend as K

def accuracy_mask():
    def f_acc(y_true, y_pred):
        nb_classes = K.int_shape(y_pred)[-1]
        y_pred = K.reshape(y_pred, (-1, nb_classes))
        
        y_true = tf.cast(y_true,tf.int32)
        y_true = K.one_hot(K.flatten(y_true), nb_classes + 1)
        unpacked = tf.unstack(y_true, axis=-1)
        legal_labels = ~tf.cast(unpacked[-1], tf.bool)
        y_true = tf.stack(unpacked[:-1], axis=-1)

        return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

    return f_acc

def crossentropy_mask(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)
    
    y_true = tf.cast(y_true,tf.int32)
    y_true = K.one_hot(K.flatten(y_true), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def categorical_focal_loss(depth, gamma=2.0, alpha=0.25, class_indexes=None):
    r"""Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)
    Args:
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
    """
    def f_cat(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        
        y_true = tf.cast(y_true,tf.int32)
        y_true = K.one_hot(K.flatten(y_true), K.int_shape(y_pred)[-1]+1)
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)
    
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    
        # Calculate focal loss
        loss = K.mean(- y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred)))
        return loss
    return f_cat

def masked_mse():
    def f_reg(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred =  K.flatten(y_pred)
        error = y_pred - y_true
        error = tf.gather(error, tf.where(tf.not_equal(y_true, 0.0)))
        return K.mean(K.square(error))
    return f_reg

def round_if_needed(x, threshold, **kwargs):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x

def average(x, per_image=False, class_weights=None, **kwargs):

    if per_image:
        x = K.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)


def f_score(depth, beta=1, class_weights=1, 
            class_indexes=None, smooth=1e-5, 
            per_image=False, threshold=None,
            **kwargs):
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:
    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}
    The formula in terms of *Type I* and *Type II* errors:
    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}
    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
    Returns:
        F-score in range [0, 1]
    """
    def f(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true,dtype=tf.uint8), depth)
        y_true = gather_channels(y_true, class_indexes)
        y_pred = gather_channels(y_pred, class_indexes)
        
        axes = [1, 2]
    
        # calculate score
        tp = K.sum(y_true * y_pred, axis=axes)
        fp = K.sum(y_pred, axis=axes) - tp
        fn = K.sum(y_true, axis=axes) - tp
    
        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = average(score, per_image, class_weights, **kwargs)

        return score
    
    return f


def precision(gt, pr, depth, class_weights=1, class_indexes=None, smooth=1e-5, per_image=False, threshold=None, **kwargs):
    r"""Calculate precision between the ground truth (gt) and the prediction (pr).
    .. math:: F_\beta(tp, fp) = \frac{tp} {(tp + fp)}
    where:
         - tp - true positives;
         - fp - false positives;
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.
    Returns:
        float: precision score
    """
    gt = tf.one_hot(tf.cast(y_true,dtype=tf.int8), depth)
    gt = gather_channels(gt, class_indexes)
    pr = gather_channels(pr, class_indexes)
    
    axes = [1, 2]

    # score calculation
    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    
    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


def recall(gt, pr, depth, class_weights=1, class_indexes=None, smooth=1e-5, per_image=False, threshold=None, **kwargs):
    r"""Calculate recall between the ground truth (gt) and the prediction (pr).
    .. math:: F_\beta(tp, fn) = \frac{tp} {(tp + fn)}
    where:
         - tp - true positives;
         - fp - false positives;
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.
    Returns:
        float: recall score
    """
    gt = tf.one_hot(gt, depth)
    gt = gather_channels(gt, class_indexes)
    pr = gather_channels(pr, class_indexes)
    
    axes = [1, 2]

    tp = K.sum(gt * pr, axis=axes)
    fn = K.sum(gt, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score


