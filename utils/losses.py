import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MAASE(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(MAASE, self).__init__()
        """Mean Average and Squared Error = 0.5*L1Loss + 0.5*L2Loss"""

    def forward(self, output, target):
        l2_loss = torch.mul(F.mse_loss(output, target), 0.5)
        l1_loss = torch.mul(F.l1_loss(output, target), 0.5)
        return torch.add(l1_loss, l2_loss)

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
    
    def forward(self, output, target):
        output = (output >= 0.5).float()
        accuracy = torch.mean(torch.abs(output - target).sum()/len(torch.flatten(output))).item()
        return np.round(1 - accuracy, 3)



