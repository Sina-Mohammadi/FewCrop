import os
from sklearn.utils import shuffle
import torch
import PIL.Image as Image
import numpy as np

__all__ = ['DatasetFolder']





class DatasetFolder(object):

    def __init__(self,split,scenario):
        
        self.images= np.load("../data_fewcrop/"+scenario+"/"+split+"_data.npy",mmap_mode='r')
        self.labels=np.load("../data_fewcrop/"+scenario+"/"+split+"_labels.npy",mmap_mode='r').astype('int')          
        
        
    def __len__(self):
        return len(self.labels)
            

    def __getitem__(self, i):
        
        img=self.images[i].astype('float32')
        img = torch.from_numpy(img)
        target=torch.from_numpy(np.asarray(self.labels[i].astype('int')))
                     
        return img, target,i

