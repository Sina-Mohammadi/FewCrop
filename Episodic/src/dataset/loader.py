import os
from sklearn.utils import shuffle
import torch
import PIL.Image as Image
import numpy as np

__all__ = ['DatasetFolder']





class DatasetFolder(object):

    def __init__(self,split,scenario):
        self.split = split
        self.scenario=scenario
        if self.scenario=="scenario_1":
            if self.split=='train':
                self.images= np.load("/home/jovyan/private/Images-PASTIS-Fold1.npy",mmap_mode='r')
                self.labels=np.load("/home/jovyan/private/Labels-PASTIS-Fold1.npy",mmap_mode='r').astype('int')          
            elif self.split=='val':
                self.images= np.load('/home/jovyan/private/Zueri-Fold3-imgs.npy',mmap_mode='r')     
                self.labels=np.load('/home/jovyan/private/Zueri-Fold3-lbls.npy',mmap_mode='r').astype('int')
            else:
                self.images= np.load('/home/jovyan/private/ImagesGhana.npy',mmap_mode='r')             
                self.labels=np.load('/home/jovyan/private/LabelsGhana.npy',mmap_mode='r').astype('int')
                
        elif self.scenario=="scenario_2":
            if self.split=='train':
                self.images= np.load("/home/jovyan/private/NearDomain-Train-X.npy",mmap_mode='r')                
                self.labels=np.load("/home/jovyan/private/NearDomain-Train-Y.npy",mmap_mode='r').astype('int')          
            elif self.split=='val':
                self.images= np.load('/home/jovyan/private/NearDomain-Val-X.npy',mmap_mode='r')                
                self.labels=np.load('/home/jovyan/private/NearDomain-Val-Y.npy',mmap_mode='r').astype('int')
            else:
                self.images= np.load('/home/jovyan/private/NearDomain-Test-X.npy',mmap_mode='r')                
                self.labels=np.load('/home/jovyan/private/NearDomain-Test-Y.npy',mmap_mode='r').astype('int')
        
        
    def __len__(self):
        return len(self.labels)
            
               


    def __getitem__(self, i):
        
        img=self.images[i].astype('float32')
        img = torch.from_numpy(img)
        target=torch.from_numpy(np.asarray(self.labels[i].astype('int')))
                     
        return img, target,i
