
# Implementation of 3D dataset

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset




class Dataset_3D(Dataset):
    def __init__(self, data_path, data_type = 'train', mask = True, shuffle = True, split = 1):
        super(Dataset_3D, self).__init__()
        
        self.data_path = data_path
        
        if data_type == 'train':
            self.train = True
            self.val = False
            
        elif data_type == 'val':
            self.train = False
            self.val = True
            
              
        self.mask = mask
        self.shuffle = shuffle
        self.split = split
        
        
               
        self.x = np.array(h5py.File(data_path+'x_train.h5','r')['x'])
        self.x_mask = np.array(h5py.File(data_path+'x_train_mask.h5','r')['x_mask'])
        y = np.array(h5py.File(data_path+'y_train.h5','r')['y'])
        self.y = y[:,None,:]
        
        if self.mask:
            samples = np.flatnonzero(self.x_mask[:,0] > 0)
        else:
            samples = np.flatnonzero(self.x_mask[:,0] > -1)
        
        if self.shuffle:
            np.random.seed(0)
            np.random.shuffle(samples)
        
        if self.train:
            self.indices = samples[:int(len(samples)*self.split)]
        elif self.val:
            self.indices = samples[int(len(samples)*self.split):]
            
        
 
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        x_in = torch.FloatTensor(self.x[self.indices[index],:,:])
        y_tg = torch.FloatTensor(self.y[self.indices[index],:,:])
        return x_in, y_tg
    
    
    
    
    
    
    
