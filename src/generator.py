#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:56:07 2020

@author: laura
"""

import numpy as np
import keras
from collections import Counter
from tensorflow.python.keras.utils.data_utils import Sequence
import deb
import pdb
import cv2
from icecream import ic
# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, labels, coords, idx_coord, channels = 25, 
                 patch_size = 15, batch_size=32, dim=(15,15,14), 
                 samp_per_epoch = None, shuffle=False, 
                 use_augm=False):
        '''
        Parameters
        ----------
        data : image with shape (row, col, channels(*seq))
        labels : labels with shape (row, col)
        depth: rgeression image with shape (row, col)
        coords : x,y coordinate for each pixel of interest
        idx_coord : index of coordinates, shape (len(coords),2)
        channels : channels of imput data for each seq. The default is 14.
        patch_size : patch size. The default is 15.
        batch_size : The default is 32.
        dim : input dimension for the CNN model. The default is (15,15,14).
        samp_per_epoch : (optional) # of samples for each epoch. The default is None.
        shuffle : (optional) shuffle after each epoch. The default is False.
        use_augm : (optional) data augmenattion. The default is False.

        Returns
        -------
            Datagenerator

        '''

        self.data = data
        self.label = labels
        #self.depth = depth
        self.dim = dim
        self.batch_size = batch_size
        self.list_coords = idx_coord
        self.coords = coords
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.channels = channels
        self.use_augm = use_augm
        self.samp_per_epoch = samp_per_epoch
        self.on_epoch_end()
        self.single_image_test = True

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.samp_per_epoch:  
            # train over #samp_per_epoch random samples at each epoch
            return int(np.floor(self.samp_per_epoch / self.batch_size))
        else:
            # use all avaliable samples at each epoch
            return int(np.floor(len(self.list_coords) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        idx_tmp = [self.list_coords[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(idx_tmp)

        return X, [Y] 

    def on_epoch_end(self):
        'Updates indexes and list coords after each epoch'
        
        if self.samp_per_epoch:
            self.indexes = np.arange(self.samp_per_epoch)
        else:
            self.indexes = np.arange(len(self.list_coords))
            
        if self.shuffle == True:
            # shuffle indexes we use to iterate on
            np.random.shuffle(self.indexes)
            # shuffle the coordiantes index 
            np.random.shuffle(self.list_coords)

    def __data_generation(self, idx_tmp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        if self.single_image_test == True:
            X = np.empty((self.batch_size, *self.dim[:-1], 2))
        else:
            X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size, self.patch_size,self.patch_size), dtype=np.uint8)
        #D = np.empty((self.batch_size, self.patch_size,self.patch_size,1), dtype=np.float32)

        # Generate data
        for i in range(len(idx_tmp)):
            # get patch
            # idx_tmp --> (index for x y coord, index for the seq (0 for seq 1 and 1 for seq 2) )
            patch_tmp = self.data[self.coords[0][idx_tmp[i]]-self.patch_size//2:self.coords[0][idx_tmp[i]]+self.patch_size//2+self.patch_size%2,
                          self.coords[1][idx_tmp[i]]-self.patch_size//2:self.coords[1][idx_tmp[i]]+self.patch_size//2+self.patch_size%2,:]

            lab_tmp = self.label[self.coords[0][idx_tmp[i]]-self.patch_size//2:self.coords[0][idx_tmp[i]]+self.patch_size//2+self.patch_size%2,
                          self.coords[1][idx_tmp[i]]-self.patch_size//2:self.coords[1][idx_tmp[i]]+self.patch_size//2+self.patch_size%2]

            #print(patch_tmp[...,-2].min(),np.average(patch_tmp[...,-2]),patch_tmp[...,-2].max())
#            np.save("patch_sample.npy",patch_tmp)
            #sample_patch = patch_tmp[...,-2].copy()
            #sample_patch = (sample_patch + 1.3) * 70 / 3.5
            #print(sample_patch.min(),np.average(sample_patch),sample_patch.max())
            #cv2.imwrite("sample_patch.png",sample_patch.astype(np.uint8))
            #cv2.imwrite('lab_tmp.png',lab_tmp*20)
            #pdb.set_trace()
            #depth_tmp = self.depth[self.coords[0][idx_tmp[i]]-self.patch_size//2:self.coords[0][idx_tmp[i]]+self.patch_size//2+self.patch_size%2,
            #              self.coords[1][idx_tmp[i]]-self.patch_size//2:self.coords[1][idx_tmp[i]]+self.patch_size//2+self.patch_size%2]

            idx_tmp = np.array(idx_tmp)
            # Random flips and rotations 
            if self.use_augm:
                transf = np.random.randint(0,6,1)
                if transf == 0:
                    # rot 90
                    patch_tmp = np.rot90(patch_tmp,1,(0,1))
                    lab_tmp = np.rot90(lab_tmp,1,(0,1))
                    #depth_tmp = np.rot90(depth_tmp,1,(0,1))
                    
                elif transf == 1:
                    # rot 180
                    patch_tmp = np.rot90(patch_tmp,2,(0,1))
                    lab_tmp = np.rot90(lab_tmp,2,(0,1))
                    #depth_tmp = np.rot90(depth_tmp,2,(0,1))
                  
                elif transf == 2:
                    # flip horizontal
                    patch_tmp = np.flip(patch_tmp,0)
                    lab_tmp = np.flip(lab_tmp,0)
                    #depth_tmp = np.flip(depth_tmp,0)
                    
                  
                elif transf == 3:
                    # flip vertical
                    patch_tmp = np.flip(patch_tmp,1)
                    lab_tmp = np.flip(lab_tmp,1)
                    #depth_tmp = np.flip(depth_tmp,1)
                  
                elif transf == 4:
                    # rot 270
                    patch_tmp = np.rot90(patch_tmp,3,(0,1))
                    lab_tmp = np.rot90(lab_tmp,3,(0,1))
                    #depth_tmp = np.rot90(depth_tmp,3,(0,1))
                 
            if self.single_image_test == True:                
                X[i,] = patch_tmp[..., -2:] # (t_len, h, w, channels)
            else:
                X[i,] = patch_tmp # (t_len, h, w, channels)    

#            X[i,] = patch_tmp
#            ic(X[i,].shape)
            X[i,...,-1] = lab_tmp.copy() # see if metrics get higher
            X[i,...,0] = lab_tmp.copy() # see if metrics get higher

            Y[i,] = lab_tmp
            #D[i,:,:,0] = depth_tmp
        #deb.prints(Y.shape)
        return X, np.expand_dims(Y,axis=-1)#, D