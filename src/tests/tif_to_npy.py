import cv2
import numpy as np
from pathlib import Path

path=Path('../../data/cv/sar')
im=cv2.imread(str(path/'20151029.tif'),-1)
print(im.shape)

im=np.expand_dims(im,axis=-1)
print(im.shape)

im=np.concatenate((im,im),axis=-1)
print(im.shape)
np.save(path/'im.npy',im)