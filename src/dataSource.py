import numpy as np
from pathlib import Path
import deb
import cv2
class Dataset():
	def imsLoad(self):
		ims=[]
		for im_name in self.ims_list:
			deb.prints(self.path/(im_name+'.npy'))
			im=np.load(self.path/(im_name+'.npy'))
			ims.append(im)
		self.ims=np.asarray(ims)
		deb.prints(self.ims.shape)
		self.t_len, self.row, self.col, self.bands = self.ims.shape
	def imsToTemporalStack(self):
		out = np.moveaxis(self.ims,0,2)
		out = out.reshape((self.row,self.col,self.t_len*self.bands))
		deb.prints(out.shape)
		np.save(self.path/'stack.npy',out)
	def loadIms(self):
		return np.load(self.path/'stack.npy')



class CampoVerdeSAR(Dataset):
	def __init__(self):
		self.path=Path('../data/cv/sar')
		deb.prints(list(self.path.glob('*')))
		self.ims_list=['20151029','20151110','20151122','20151204','20151216','20160121']

if __name__ == "__main__":
	ds = CampoVerdeSAR()
	ds.imsLoad()
	ds.imsToTemporalStack()
