import numpy as np


class MIM():
    def im_load(self, path,im_names):

        out=[]
        for im_name in im_names:
            im=np.load(path/(im_name+'.npy')).astype(np.float16)
            out.append(im)
        out=np.asarray(out).astype(np.float16)
        print("Loaded images shape", out.shape)
        self.t_len, self.row, self.col, self.bands = out.shape
        return out
class MIMTimeSequence(MIM):
    def __init__(self):
        pass
    def getChannelwiseFlattenedSequence(self, image):
        image=np.moveaxis(image,0,-1)
        image=np.reshape(image,(self.row,self.col,self.bands*self.t_len))
        print("Scaler flattened images shape after t and band concatenation", image.shape)
        return image
    def reshapeForScaler(self, image):
        image = self.getChannelwiseFlattenedSequence(image)
        image = image.reshape(self.row*self.col,self.bands * self.t_len)
        return image
    def getChannelwiseUnflattenedSequence(self, image):
        image=np.reshape(image,(self.row,self.col,self.bands,self.t_len))
        image=np.moveaxis(image,-1,0)
        return image
    def reshapeFromScaler(self, image):
        image = image.reshape(self.row,self.col,self.bands * self.t_len)
        image = self.getChannelwiseUnflattenedSequence(image)
        return image
class MIMStack(MIM):
    def __init__(self):
        pass
    def im_load(self, path,im_names):
        out = super().im_load(path,im_names)
        
        out=np.moveaxis(out,0,-1)
        out=np.reshape(out,(self.row,self.col,self.bands*self.t_len))
        print("Loaded images shape after t and band concatenation", out.shape)
        
        return out
    def getChannelwiseFlattenedSequence(self, image):
        return image
    def reshapeForScaler(self, image):
        image = image.reshape(self.row*self.col,self.bands * self.t_len)
        return image
    def reshapeFromScaler(self, image):
        image = image.reshape(self.row,self.col,self.bands * self.t_len)
        return image

