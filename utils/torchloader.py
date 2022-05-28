import numpy as np
import torch
from torch.utils.data import Dataset

class Data(Dataset):

    def __init__(self, input_path):
        
        #input_path is path to the side and front images numpy array
        self.data = np.load(input_path)
        
        # preprocessing step to convert grey image to mask
        self.data = np.array(self.data,dtype='float')/255.0
        self.data[self.data < 1] = 0
        self.data = 1 - self.data

        self.data = np.expand_dims(self.data, axis = 1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        front_ = self.data[i,:,:,:,0]
        side_ = self.data[i,:,:,:,1]
        
        front_ = torch.from_numpy(front_).float()
        side_ = torch.from_numpy(side_).float()

        return front_, side_


