
# Implementation of 2D dataset

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset



class Dataset_2D(Dataset):
    def __init__(self, data_path, data_type = 'train', shuffle = True, split = 1):
        super(Dataset_2D, self).__init__()
        
        self.data_path = data_path
        
        if data_type == 'train':
            self.train = True
            self.val = False
            
        elif data_type == 'val':
            self.train = False
            self.val = True
            

        self.shuffle = shuffle
        self.split = split
        
        
        self.x = np.array(h5py.File(data_path+'x_train.h5','r')['x'])
        self.x_mask = np.array(h5py.File(data_path+'x_train_mask.h5','r')['x_mask'])
        self.y = np.array(h5py.File(data_path+'y_train.h5','r')['y'])
        
        samples = np.arange(self.x.shape[0])
        
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
        
        x_in = torch.FloatTensor(self.x[self.indices[index],:,:,:])
        y_tg = torch.FloatTensor(self.y[self.indices[index],:,:,:])
        x_mask = torch.FloatTensor(self.x_mask[self.indices[index],:,:,:])
            
        return x_in, y_tg, x_mask
    
    
    
    
    
    
    
